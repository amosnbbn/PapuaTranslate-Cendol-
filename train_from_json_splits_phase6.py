import os, argparse, json, random, numpy as np, torch
from typing import Dict, List
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
)
from peft import PeftModel
import evaluate

# ---------------------- Utils ----------------------
def set_seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_prompt(instruction: str, src: str) -> str:
    return f"Instruksi: {instruction}\nTeks: {src}\nKeluaran:"

def sanitize_for_decode(batch_ids: List[List[int]], pad_id: int) -> List[List[int]]:
    out = []
    for row in batch_ids:
        out.append([tid if isinstance(tid, int) and tid >= 0 else pad_id for tid in row])
    return out

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser("Fine-tune Cendol-mT5 dari JSON splits (instruction/input/output)")
    ap.add_argument("--base_dir", required=True, help="Folder base model, ex: C:/ML/models/cendol-mt5-base-inst")
    ap.add_argument("--adapter_init", default="", help="(Opsional) LoRA adapter untuk starting weights, ex: C:/.../checkpoint-96")
    ap.add_argument("--splits_dir", required=True, help="Folder json_splits/json (berisi train.json, valid.json, test.json)")
    ap.add_argument("--out_dir", required=True, help="Folder output training")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=0.1)
    ap.add_argument("--src_max_len", type=int, default=256)
    ap.add_argument("--tgt_max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Gunakan FP16 (hati2 OOM). Default FP32")
    ap.add_argument("--bf16", action="store_true", help="Gunakan BF16 kalau GPU mendukung")
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()

    # ---- runtime safety
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed_all(args.seed)

    assert torch.cuda.is_available(), "CUDA tidak terdeteksi."
    device = "cuda"

    # ---- load data (JSON)
    files = {
        "train": os.path.join(args.splits_dir, "train.json"),
        "validation": os.path.join(args.splits_dir, "valid.json"),
        "test": os.path.join(args.splits_dir, "test.json"),
    }
    ds = load_dataset("json", data_files=files)

    # sanity kolom
    for split in ["train", "validation", "test"]:
        cols = set(ds[split].column_names)
        for k in ["instruction", "input", "output"]:
            assert k in cols, f"{split} harus punya kolom '{k}', ada: {ds[split].column_names}"

    # ---- tokenizer & base
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True)
    tok.padding_side = "right"

    # dtype
    if args.bf16:
        dtype = torch.bfloat16; use_fp16=False; use_bf16=True
    elif args.fp16:
        dtype = torch.float16;  use_fp16=True;  use_bf16=False
    else:
        dtype = torch.float32;  use_fp16=False; use_bf16=False

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_dir,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype if dtype != torch.float32 else None,
        local_files_only=True
    ).to(device)
    base.config.use_cache = False
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = tok.pad_token_id
    if getattr(base.config, "eos_token_id", None) is None:
        base.config.eos_token_id = tok.eos_token_id
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id
    base.config.forced_bos_token_id = None
    base.config.forced_eos_token_id = None

    # ---- (opsional) init dari LoRA adapter lama
    if args.adapter_init and os.path.isdir(args.adapter_init):
        model = PeftModel.from_pretrained(
            base, args.adapter_init, is_trainable=True, local_files_only=True,
            torch_dtype=dtype if dtype != torch.float32 else None,
        ).to(device)
    else:
        # kalau tidak ada adapter init, tetap training full-LoRA di base (pastikan PEFT config ada di base dir jika ingin LoRA baru)
        model = base  # tetap jalan, tapi tanpa LoRA tambahan
    model.train()
    try:
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    except:
        pass

    # ---- preprocess
    trunc_src = 0; trunc_tgt = 0
    def _prep(batch):
        nonlocal trunc_src, trunc_tgt
        instrs = [str(x) for x in batch["instruction"]]
        srcs   = [str(x) for x in batch["input"]]
        tgts   = [str(x) for x in batch["output"]]

        prompts = [make_prompt(i, s) for i, s in zip(instrs, srcs)]

        enc = tok(prompts, max_length=args.src_max_len, truncation=True, padding=False)
        dec = tok(text_target=tgts, max_length=args.tgt_max_len, truncation=True, padding=False)

        trunc_src += sum(len(ids) >= args.src_max_len for ids in enc["input_ids"])
        trunc_tgt += sum(len(ids) >= args.tgt_max_len for ids in dec["input_ids"])

        enc["labels"] = dec["input_ids"]
        return enc

    keep = {"input_ids","attention_mask","labels"}
    rm_train = [c for c in ds["train"].column_names if c not in keep]
    rm_val   = [c for c in ds["validation"].column_names if c not in keep]
    rm_test  = [c for c in ds["test"].column_names if c not in keep]

    ds_proc = DatasetDict(
        train=ds["train"].map(_prep, batched=True, remove_columns=rm_train),
        validation=ds["validation"].map(_prep, batched=True, remove_columns=rm_val),
        test=ds["test"].map(_prep, batched=True, remove_columns=rm_test),
    )
    print(f"[INFO] Truncated -> src: {trunc_src} | tgt: {trunc_tgt}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100, return_tensors="pt"
    )

    # ---- metrics
    bleu = evaluate.load("sacrebleu"); chrf = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        preds = preds.tolist() if hasattr(preds, "tolist") else preds
        labels = labels.tolist() if hasattr(labels, "tolist") else labels

        preds_clean  = sanitize_for_decode(preds, tok.pad_token_id)
        labels_clean = sanitize_for_decode(labels, tok.pad_token_id)

        preds_str  = [s.strip() for s in tok.batch_decode(preds_clean,  skip_special_tokens=True)]
        labels_str = [s.strip() for s in tok.batch_decode(labels_clean, skip_special_tokens=True)]
        refs = [[l] for l in labels_str]

        try: bleu_s = bleu.compute(predictions=preds_str, references=refs)["score"]
        except: bleu_s = 0.0
        try: chrf_s = chrf.compute(predictions=preds_str, references=refs)["score"]
        except: chrf_s = 0.0
        return {"bleu": bleu_s, "chrf": chrf_s}

    # ---- args
    os.makedirs(args.out_dir, exist_ok=True)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,

        eval_strategy="epoch",      # pakai eval_strategy (lebih baru dari evaluation_strategy)
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,

        predict_with_generate=True,
        generation_max_length=args.tgt_max_len,
        generation_num_beams=args.beam,

        optim="adamw_torch",
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,

        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=False,

        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        seed=args.seed,
        remove_unused_columns=False,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    print("[INFO] Best checkpoint:", trainer.state.best_model_checkpoint)

    # ---- final eval (VALID & TEST) pakai beam argumen yang sama
    for split, key in [("validation","FINAL_VALID"), ("test","FINAL_TEST")]:
        metrics = trainer.evaluate(eval_dataset=ds_proc[split], metric_key_prefix=key.lower())
        print(f"[{key}] {metrics}")

    # ---- save adapter/tokenizer
    print("[INFO] Save adapter/tokenizer")
    try:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(args.out_dir)
    except Exception as e:
        print("[WARN] save adapter failed:", repr(e))
    tok.save_pretrained(args.out_dir)
    print("[OK] Done ->", args.out_dir)

    # ---- sanity generate
    try:
        model.eval()
        samples = [
            "Awam kapito… (contoh kalimat bahasa Papua pendek)",
            "Tabea, mari katong pigi pasar besok…",
        ]
        prompts = [make_prompt("Terjemahkan ke bahasa Indonesia baku.", s) for s in samples]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.src_max_len).to(device)
        with torch.inference_mode():
            gen_ids = model.generate(**enc, max_length=args.tgt_max_len, num_beams=args.beam, do_sample=False)
        outs = tok.batch_decode(gen_ids, skip_special_tokens=True)
        print("\n[GEN SAMPLE]")
        for s, o in zip(samples, outs):
            print("SRC:", s); print("OUT:", o); print("-"*50)
    except Exception as e:
        print("[WARN] sanity generation failed:", repr(e))

if __name__ == "__main__":
    main()
