# curriculum_trainer.py
# Stable curriculum finetuning with OOM-safe hard mining & evaluation

import os, json, random, argparse, tempfile
from typing import List, Dict
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import PeftModel
import evaluate
from inspect import signature

# ======================= Utils & Safety =======================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def set_torch_safety():
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================= Data Helpers =========================
def _detect_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    if {"instruction","input","output"}.issubset(cols.keys()):
        return cols["instruction"], cols["input"], cols["output"]
    if {"source","target"}.issubset(cols.keys()):
        instr = cols["instruction"] if "instruction" in cols else None
        return instr, cols["source"], cols["target"]
    raise ValueError(
        f"Kolom tidak sesuai. Ditemukan: {list(df.columns)} | "
        f"Harus (instruction,input,output) ATAU (source,target [+instruction ops])."
    )

def normalize_df_to_iio(df: pd.DataFrame, default_instruction: str):
    instr_col, in_col, out_col = _detect_cols(df)
    if instr_col is None:
        df = df.rename(columns={in_col:"input", out_col:"output"})
        df["instruction"] = default_instruction
    else:
        df = df.rename(columns={instr_col:"instruction", in_col:"input", out_col:"output"})
    # clean
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["input"] = df["input"].astype(str).str.strip()
    df["output"] = df["output"].astype(str).str.strip()
    df = df[(df["input"]!="") & (df["output"]!="")]
    df = df.drop_duplicates(subset=["instruction","input","output"]).reset_index(drop=True)
    return df[["instruction","input","output"]]

def load_splits_csv(splits_dir: str, default_instruction: str):
    paths = {
        "train": os.path.join(splits_dir, "train.csv"),
        "valid": os.path.join(splits_dir, "valid.csv"),
        "test":  os.path.join(splits_dir, "test.csv"),
    }
    for k,p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split CSV tidak ditemukan: {p}")
    train = normalize_df_to_iio(pd.read_csv(paths["train"]), default_instruction)
    valid = normalize_df_to_iio(pd.read_csv(paths["valid"]), default_instruction)
    test  = normalize_df_to_iio(pd.read_csv(paths["test"]),  default_instruction)
    return train, valid, test

def make_prompt(instr: str, src: str) -> str:
    return f"Instruksi: {instr}\nTeks: {src}\nKeluaran:"

# ======================= Adapter Resolver =====================
def _find_adapter_dir(adapter_dir: str) -> str:
    """
    Pastikan path mengarah ke folder yang berisi adapter_config.json.
    Jika yang diberikan adalah root run, otomatis pilih checkpoint-* terbaru yang valid.
    """
    adapter_dir = os.path.normpath(adapter_dir)
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(cfg):
        return adapter_dir

    if os.path.isdir(adapter_dir):
        cps = [os.path.join(adapter_dir, d) for d in os.listdir(adapter_dir)
               if d.startswith("checkpoint-") and os.path.isdir(os.path.join(adapter_dir, d))]
        def _step(p):
            try: return int(os.path.basename(p).split("-")[-1])
            except: return -1
        cps.sort(key=_step, reverse=True)
        for c in cps:
            if os.path.isfile(os.path.join(c, "adapter_config.json")):
                return c

    raise FileNotFoundError(
        f"adapter_config.json tidak ditemukan di '{adapter_dir}' maupun subfolder checkpoint-*.\n"
        f"Pastikan training menyimpan adapter (trainer.save_model) ATAU arahkan ke folder checkpoint yang benar."
    )

# ======================= Model Loading ========================
def load_tok_and_model(base_dir: str, adapter_dir: str,
                       dtype: torch.dtype = torch.float32, device: str="cuda"):
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    tok.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_dir, use_safetensors=True, torch_dtype=dtype,
        low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    base.config.use_cache = False
    if getattr(base.config,"decoder_start_token_id",None) is None:
        base.config.decoder_start_token_id = tok.pad_token_id
    if getattr(base.config,"eos_token_id",None) is None:
        base.config.eos_token_id = tok.eos_token_id
    if getattr(base.config,"pad_token_id",None) is None:
        base.config.pad_token_id = tok.pad_token_id
    base.config.forced_bos_token_id = None
    base.config.forced_eos_token_id = None

    adapter_dir = _find_adapter_dir(adapter_dir)  # resolve valid folder

    model = PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=True, local_files_only=True,
        torch_dtype=(dtype if dtype!=torch.float32 else None)
    ).to(device)
    model.train()
    return tok, model

# ======================= Hard Set Builder (OOM-safe) ==========
def build_hard_set(
    df_train: pd.DataFrame,
    tok, model,
    top_n: int = 400,
    batch_size: int = 8,
    max_src_len: int = 224,
    max_new_tokens: int = 48
) -> pd.DataFrame:
    """
    Skor kesulitan = 1 - chrF(pred, gold). Ambil TOP-N (terbesar).
    Versi hemat VRAM: use_cache=False, auto backoff batch/length saat OOM.
    """
    chrf = evaluate.load("chrf")  # 0..100
    device = next(model.parameters()).device
    preds = []

    # hemat memori
    model.eval()
    torch.cuda.empty_cache()

    i = 0
    n = len(df_train)
    cur_bsz = max(1, batch_size)

    while i < n:
        end = min(i + cur_bsz, n)
        chunk = df_train.iloc[i:end]

        prompts = [make_prompt(instr, src)
                   for instr, src in zip(chunk["instruction"], chunk["input"])]
        enc = tok(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_src_len
        ).to(device)

        try:
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=False  # kunci hemat VRAM
                )
            outs = tok.batch_decode(gen, skip_special_tokens=True)
            preds.extend([o.strip() for o in outs])
            i = end  # lanjut batch berikutnya

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                if cur_bsz > 1:
                    cur_bsz = max(1, cur_bsz // 2)  # kecilkan batch
                    continue
                # cur_bsz == 1 → kecilkan panjang generate
                if max_new_tokens > 16:
                    max_new_tokens = max(16, max_new_tokens // 2)
                    continue
                # sudah mentok: lempar lagi
                raise
            else:
                raise

    refs = [[t] for t in df_train["output"].tolist()]
    scores = []
    for p, r in zip(preds, refs):
        try:
            s = chrf.compute(predictions=[p], references=[r])["score"]
        except Exception:
            s = 0.0
        scores.append(s)

    df = df_train.copy()
    df["chrf"] = scores
    df["difficulty"] = 1.0 - (df["chrf"]/100.0)
    df = (df.sort_values("difficulty", ascending=False)
            .head(top_n)
            .drop(columns=["chrf","difficulty"])
            .reset_index(drop=True))
    return df

# ======================= HF Dataset Prep ======================
def _dict_to_jsonl(d: Dict[str, List[str]]) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl"); os.close(fd)
    with open(path,"w",encoding="utf-8") as f:
        n = len(next(iter(d.values())))
        for i in range(n):
            row = {k: d[k][i] for k in d}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path

def to_hf_dataset(df: pd.DataFrame, tok, max_src_len=256, max_tgt_len=256):
    tmp = {"instruction": df["instruction"].tolist(),
           "input": df["input"].tolist(),
           "output": df["output"].tolist()}
    ds = DatasetDict({"data": load_dataset("json",
                         data_files={"data": _dict_to_jsonl(tmp)}, split="data")})
    def _preprocess(batch):
        prompts = [make_prompt(i,s) for i,s in zip(batch["instruction"], batch["input"])]
        enc = tok(prompts, max_length=max_src_len, truncation=True, padding=False)
        dec = tok(text_target=batch["output"], max_length=max_tgt_len, truncation=True, padding=False)
        enc["labels"] = dec["input_ids"]
        return enc
    cols_keep = {"input_ids","attention_mask","labels"}
    rm_cols = [c for c in ds["data"].column_names if c not in cols_keep]
    ds_proc = ds["data"].map(_preprocess, batched=True, remove_columns=rm_cols)
    return ds_proc

def data_collator_seq2seq(tok, model):
    return DataCollatorForSeq2Seq(
        tokenizer=tok, model=model,
        padding="longest", label_pad_token_id=-100,
        return_tensors="pt"
    )

# ======================= Train (stabil & OOM-safe eval) =======
def _assert_has_trainables(model):
    any_trainable = False
    for _, p in model.named_parameters():
        if p.requires_grad:
            any_trainable = True
            break
    if not any_trainable:
        raise RuntimeError(
            "Tidak ada parameter yang requires_grad=True. "
            "Pastikan adapter LoRA dimuat dengan benar (is_trainable=True) dan folder adapter valid (punya adapter_config.json)."
        )

def train_once(tok, model, train_ds, valid_ds,
               out_dir: str,
               epochs: int,
               lr: float,
               weight_decay: float = 0.01,
               max_grad_norm: float = 0.3,
               bsz: int = 4,
               eval_bsz: int = 1,
               grad_accum: int = 8,
               eval_accum_steps: int = 1,
               eval_loss_only: bool = True,
               fp16: bool = False,
               bf16: bool = False,
               seed: int = 42,
               scheduler_type: str = "cosine",
               warmup_ratio: float = 0.06,
               warmup_steps: int = 0,
               grad_checkpointing: bool = False,
               label_smoothing: float = 0.1):

    os.makedirs(out_dir, exist_ok=True)

    # stabil + dukung grad ckpt
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    if grad_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    _assert_has_trainables(model)
    torch.set_grad_enabled(True)

    sig = signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters

    kw = dict(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=max(1, eval_bsz),
        gradient_accumulation_steps=grad_accum,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["none"],
        predict_with_generate=False,  # eval tanpa generate (hemat VRAM)
        fp16=False,                   # training stabil
        bf16=bf16,                    # boleh True jika GPU support
        remove_unused_columns=False,
        seed=seed,
        optim="adamw_torch",
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        label_smoothing_factor=label_smoothing,
    )
    # evaluation/save strategy (handle versi beda)
    if "evaluation_strategy" in params: kw["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:     kw["eval_strategy"] = "epoch"
    if "save_strategy" in params:       kw["save_strategy"] = "epoch"
    if "save_safetensors" in params:    kw["save_safetensors"] = True
    if "logging_strategy" in params:    kw["logging_strategy"] = "steps"

    # scheduler + warmup
    if "lr_scheduler_type" in params: kw["lr_scheduler_type"] = scheduler_type
    if "warmup_ratio" in params and warmup_steps == 0: kw["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in params: kw["warmup_steps"] = warmup_steps if warmup_steps>0 else 500

    # gradient checkpointing flag (opsional)
    if "gradient_checkpointing" in params:
        kw["gradient_checkpointing"] = bool(grad_checkpointing)

    # OOM mitigation for EVAL
    if "eval_accumulation_steps" in params:
        kw["eval_accumulation_steps"] = max(1, eval_accum_steps)
    if bf16 and "bf16_full_eval" in params:
        kw["bf16_full_eval"] = True
    elif (not bf16) and "fp16_full_eval" in params:
        kw["fp16_full_eval"] = True
    if "dataloader_pin_memory" in params:
        kw["dataloader_pin_memory"] = True
    if "dataloader_num_workers" in params:
        kw["dataloader_num_workers"] = 0

    args = Seq2SeqTrainingArguments(**kw)
    coll = data_collator_seq2seq(tok, model)

    if eval_loss_only:
        def cm(_): return {}
    else:
        bleu = evaluate.load("sacrebleu"); chrf = evaluate.load("chrf")
        def _sanitize(arr2d, pad_id):
            out=[]
            for row in arr2d:
                row = row.tolist() if hasattr(row,"tolist") else row
                out.append([tid if (isinstance(tid,int) and tid>=0) else pad_id for tid in row])
            return out
        def cm(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple): preds = preds[0]
            preds_clean = _sanitize(preds, tok.pad_token_id)
            labels_clean = _sanitize(labels, tok.pad_token_id)
            preds_str = tok.batch_decode(preds_clean, skip_special_tokens=True)
            labels_str = tok.batch_decode(labels_clean, skip_special_tokens=True)
            preds_str = [p.strip() for p in preds_str]
            refs = [[l.strip()] for l in labels_str]
            try: b = bleu.compute(predictions=preds_str, references=refs)["score"]
            except: b = 0.0
            try: c = chrf.compute(predictions=preds_str, references=refs)["score"]
            except: c = 0.0
            return {"bleu": b, "chrf": c}

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=coll,
        tokenizer=tok,  # deprecation warning aman
        compute_metrics=cm,
    )

    trainer.train()

    # Simpan adapter/LoRA ke root -> memastikan ada adapter_config.json di out_dir
    trainer.save_model()
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(out_dir)

    return out_dir

# ======================= Final Evaluation =====================
def final_eval(tok, model, df,
               beams=6, len_pen=0.95, no_repeat=3,
               min_new=4, max_new=64, batch_size=16):
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    device = next(model.parameters()).device
    preds = []
    refs  = [[t] for t in df["output"].tolist()]
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]
        prompts = [make_prompt(inst, src) for inst,src in zip(chunk["instruction"], chunk["input"])]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                num_beams=beams, length_penalty=len_pen, no_repeat_ngram_size=no_repeat,
                min_new_tokens=min_new, max_new_tokens=max_new, do_sample=False
            )
        outs = tok.batch_decode(gen, skip_special_tokens=True)
        preds.extend([o.strip() for o in outs])
    b = bleu.compute(predictions=preds, references=refs)["score"]
    c = chrf.compute(predictions=preds, references=refs)["score"]
    return {"BLEU": b, "chrF": c}

# ======================= Main =================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--start_adapter", required=True, help="checkpoint adapter awal (boleh root run)")
    parser.add_argument("--splits_dir", required=True, help="folder CSV train/valid/test")
    parser.add_argument("--out_root", required=True, help="folder output semua siklus")
    parser.add_argument("--cycles", type=int, default=1)

    parser.add_argument("--default_instruction", type=str, default="Terjemahkan ke bahasa Indonesia baku.")

    # hard builder sampling/mining
    parser.add_argument("--hard_top_n", type=int, default=250)
    parser.add_argument("--hard_gen_bsz", type=int, default=8)          # batch untuk generate hard-set
    parser.add_argument("--hard_max_src_len", type=int, default=224)     # src len untuk hard-set
    parser.add_argument("--hard_max_new", type=int, default=48)          # max new tokens saat hard-set
    parser.add_argument("--hard_epochs", type=int, default=1)
    parser.add_argument("--hard_lr", type=float, default=2e-6)

    # polish full
    parser.add_argument("--full_epochs", type=int, default=2)
    parser.add_argument("--full_lr", type=float, default=1e-6)

    # sequence & batch sizing
    parser.add_argument("--max_src_len", type=int, default=224)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--eval_bsz", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--eval_accum_steps", type=int, default=1)

    # precision & stability
    parser.add_argument("--fp16", action="store_true", default=False)  # default False (stabil)
    parser.add_argument("--bf16", action="store_true", default=False)  # nyalakan jika GPU support
    parser.add_argument("--grad_ckpt", action="store_true", default=False)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine","linear","polynomial","constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    # decode params for final evals
    parser.add_argument("--beams", type=int, default=6)
    parser.add_argument("--len_pen", type=float, default=0.95)
    parser.add_argument("--no_repeat", type=int, default=3)
    parser.add_argument("--min_new", type=int, default=4)
    parser.add_argument("--max_new", type=int, default=64)

    args = parser.parse_args()

    set_seed(args.seed)
    set_torch_safety()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # load data
    train_df, valid_df, test_df = load_splits_csv(args.splits_dir, args.default_instruction)
    print(f"[INFO] Sizes -> train: {len(train_df)} | valid: {len(valid_df)} | test: {len(test_df)}")

    # start adapter
    cur_adapter = args.start_adapter

    for cycle in range(1, args.cycles+1):
        print(f"\n====== CYCLE {cycle} ======")
        # load model (adapter path bisa root; akan di-resolve)
        tok, model = load_tok_and_model(args.base_dir, cur_adapter, dtype=dtype, device=device)

        # 1) build hard set (OOM-safe)
        print("[INFO] Build hard set …")
        hard_df = build_hard_set(
            train_df, tok, model,
            top_n=args.hard_top_n,
            batch_size=args.hard_gen_bsz,
            max_src_len=args.hard_max_src_len,
            max_new_tokens=args.hard_max_new
        )
        os.makedirs(args.out_root, exist_ok=True)
        hard_csv_path = os.path.join(args.out_root, f"cycle{cycle}_hard_train.csv")
        hard_df.to_csv(hard_csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Hard CSV -> {hard_csv_path} (rows={len(hard_df)})")

        # 2) micro-finetune on hard set
        print("[INFO] Micro-finetune (hard set) …")
        hard_ds_train = to_hf_dataset(hard_df, tok, args.max_src_len, args.max_tgt_len)
        valid_ds      = to_hf_dataset(valid_df, tok, args.max_src_len, args.max_tgt_len)
        out_hard = os.path.join(args.out_root, f"cycle{cycle}_hard_ft")
        best_hard = train_once(tok, model, hard_ds_train, valid_ds,
                               out_dir=out_hard,
                               epochs=args.hard_epochs, lr=args.hard_lr,
                               weight_decay=0.01, max_grad_norm=0.3,
                               bsz=args.bsz, eval_bsz=args.eval_bsz,
                               grad_accum=args.grad_accum,
                               eval_accum_steps=args.eval_accum_steps,
                               eval_loss_only=True,
                               fp16=args.fp16, bf16=args.bf16, seed=args.seed,
                               scheduler_type=args.scheduler,
                               warmup_ratio=args.warmup_ratio,
                               warmup_steps=args.warmup_steps,
                               grad_checkpointing=args.grad_ckpt,
                               label_smoothing=args.label_smoothing)

        # reload dari adapter hasil hard-ft
        del tok, model; torch.cuda.empty_cache()
        tok, model = load_tok_and_model(args.base_dir, best_hard, dtype=dtype, device=device)

        # 3) polish di full train
        print("[INFO] Polish (full train) …")
        full_ds_train = to_hf_dataset(train_df, tok, args.max_src_len, args.max_tgt_len)
        valid_ds      = to_hf_dataset(valid_df, tok, args.max_src_len, args.max_tgt_len)
        out_full = os.path.join(args.out_root, f"cycle{cycle}_full_polish")
        best_full = train_once(tok, model, full_ds_train, valid_ds,
                               out_dir=out_full,
                               epochs=args.full_epochs, lr=args.full_lr,
                               weight_decay=0.01, max_grad_norm=0.3,
                               bsz=args.bsz, eval_bsz=args.eval_bsz,
                               grad_accum=args.grad_accum,
                               eval_accum_steps=args.eval_accum_steps,
                               eval_loss_only=True,
                               fp16=args.fp16, bf16=args.bf16, seed=args.seed,
                               scheduler_type=args.scheduler,
                               warmup_ratio=args.warmup_ratio,
                               warmup_steps=args.warmup_steps,
                               grad_checkpointing=args.grad_ckpt,
                               label_smoothing=args.label_smoothing)

        # evaluasi akhir cycle
        del tok, model; torch.cuda.empty_cache()
        tok, model = load_tok_and_model(args.base_dir, best_full, dtype=dtype, device=device)
        print("[INFO] Final eval (VALID) …")
        ev_v = final_eval(tok, model, valid_df, args.beams, args.len_pen, args.no_repeat, args.min_new, args.max_new)
        print("[FINAL VALID]", ev_v)
        print("[INFO] Final eval (TEST) …")
        ev_t = final_eval(tok, model, test_df, args.beams, args.len_pen, args.no_repeat, args.min_new, args.max_new)
        print("[FINAL TEST]", ev_t)

        # simpan hasil & siapkan adapter untuk cycle berikutnya
        with open(os.path.join(args.out_root, f"cycle{cycle}_report.json"), "w", encoding="utf-8") as f:
            json.dump({"valid":ev_v, "test":ev_t, "hard_csv": hard_csv_path,
                       "best_hard":best_hard, "best_full":best_full}, f, ensure_ascii=False, indent=2)

        cur_adapter = best_full
        del tok, model; torch.cuda.empty_cache()

    print("\n[OK] Selesai semua siklus.")

if __name__ == "__main__":
    main()
