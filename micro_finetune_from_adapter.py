# micro_finetune_from_adapter.py  (robust + safe-decode)
import os, argparse, random
from typing import Optional, List
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
)
from peft import PeftModel
from sacrebleu.metrics import CHRF

# ====== kolom kandidat (biar fleksibel) ======
COL_IN_CANDS  = ["input","source","src","text"]
COL_OUT_CANDS = ["output","target","tgt","label"]
COL_INS_CANDS = ["instruction","instruksi"]

def pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    low = [c.lower() for c in cols]
    for name in cands:
        if name.lower() in low:
            return cols[low.index(name.lower())]
    return None

def detect_fmt(path: str) -> str:
    p = path.lower()
    if p.endswith(".csv"): return "csv"
    if p.endswith(".tsv") or p.endswith(".tab"): return "tsv"
    if p.endswith(".json") or p.endswith(".jsonl"): return "json"
    raise ValueError(f"Unsupported file: {path}")

def load_one(path: str) -> Dataset:
    fmt = detect_fmt(path)
    if fmt == "csv":
        return load_dataset("csv", data_files={"d": path})["d"]
    if fmt == "tsv":
        return load_dataset("csv", data_files={"d": path}, delimiter="\t")["d"]
    if fmt == "json":
        return load_dataset("json", data_files={"d": path})["d"]
    raise AssertionError

def normalize_split(ds: Dataset) -> Dataset:
    cols = ds.column_names
    cin  = pick_col(cols, COL_IN_CANDS)
    cout = pick_col(cols, COL_OUT_CANDS)
    cins = pick_col(cols, COL_INS_CANDS)  # optional
    if not cin or not cout:
        raise KeyError(f"Kolom input/target tidak ditemukan. Kolom ada: {cols}")
    keep = [cin, cout] + ([cins] if cins else [])
    ds = ds.remove_columns([c for c in cols if c not in keep])
    rename_map = {cin: "input", cout: "output"}
    if cins: rename_map[cins] = "instruction"
    ds = ds.rename_columns(rename_map)
    return ds

def make_prompt(inst: Optional[str], txt: str, default_inst: str) -> str:
    inst = inst.strip() if inst else default_inst
    return f"Instruksi: {inst}\nTeks: {txt}\nKeluaran:"

def _sanitize_ids(nested_ids, pad_id: int):
    """
    Ubah ke list-of-list int & ganti semua nilai < 0 jadi pad_id
    supaya decode tidak error (OverflowError).
    """
    fixed = []
    # nested_ids bisa torch.Tensor ragged; gunakan tolist() bila ada
    outer = nested_ids.tolist() if hasattr(nested_ids, "tolist") else nested_ids
    for row in outer:
        clean = []
        for x in row:
            try:
                xi = int(x)
            except Exception:
                xi = pad_id
            clean.append(xi if xi >= 0 else pad_id)
        fixed.append(clean)
    return fixed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",        type=str, required=True, help="folder model base HF (full, bukan GGUF)")
    ap.add_argument("--adapter_in",  type=str, required=True, help="folder adapter LoRA awal (checkpoint sebelumnya)")
    ap.add_argument("--train",       type=str, required=True, help="CSV/TSV/JSON train (bebas: input/output atau source/target)")
    ap.add_argument("--valid",       type=str, required=True, help="CSV/TSV/JSON valid (skema kolom bebas)")
    ap.add_argument("--out",         type=str, required=True, help="folder output adapter hasil micro-finetune")

    ap.add_argument("--epochs",      type=int, default=2)
    ap.add_argument("--lr",          type=float, default=5e-6)
    ap.add_argument("--bsz",         type=int, default=8)
    ap.add_argument("--grad_accum",  type=int, default=4)
    ap.add_argument("--seed",        type=int, default=42)

    ap.add_argument("--max_src",     type=int, default=256)
    ap.add_argument("--max_tgt",     type=int, default=128)
    ap.add_argument("--instruction", type=str, default="Terjemahkan ke bahasa Indonesia baku.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA tidak tersedia."
    device = "cuda"

    # deterministik ringan
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION","1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # tokenizer + base
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tok.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
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

    # adapter (trainable)
    model = PeftModel.from_pretrained(
        base, args.adapter_in, is_trainable=True, local_files_only=True
    ).to(device)

    # load & normalisasi split (toleran beda kolom)
    ds_tr = normalize_split(load_one(args.train))
    ds_va = normalize_split(load_one(args.valid))

    # filter kosong
    def _ok(ex):
        return bool(str(ex["input"]).strip()) and bool(str(ex["output"]).strip())
    ds_tr = ds_tr.filter(_ok)
    ds_va = ds_va.filter(_ok)

    # preprocess
    def preprocess(batch):
        ins = batch.get("instruction", None)
        if ins is None:
            ins = [""] * len(batch["input"])
        elif not isinstance(ins, list):
            ins = [ins] * len(batch["input"])
        prompts = [make_prompt(i, s, args.instruction) for i, s in zip(ins, batch["input"])]
        enc = tok(prompts, max_length=args.max_src, truncation=True, padding=False)
        dec = tok(text_target=batch["output"], max_length=args.max_tgt, truncation=True, padding=False)
        enc["labels"] = dec["input_ids"]
        return enc

    keep = {"input_ids","attention_mask","labels"}
    ds_proc = DatasetDict(
        train=ds_tr.map(preprocess, batched=True, remove_columns=[c for c in ds_tr.column_names if c not in keep]),
        validation=ds_va.map(preprocess, batched=True, remove_columns=[c for c in ds_va.column_names if c not in keep]),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100
    )

    # metric: chrF++
    chrf_metric = CHRF(word_order=2)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):  # kadang (preds, None)
            preds = preds[0]
        preds_clean  = _sanitize_ids(preds,  tok.pad_token_id)
        labels_clean = _sanitize_ids(labels, tok.pad_token_id)

        preds_str  = [s.strip() for s in tok.batch_decode(preds_clean,  skip_special_tokens=True)]
        labels_str = [s.strip() for s in tok.batch_decode(labels_clean, skip_special_tokens=True)]

        scores = [chrf_metric.sentence_score(preds_str[i], [labels_str[i]]).score
                  for i in range(len(preds_str))]
        return {"chrf": float(np.mean(scores))}

    # train args
    os.makedirs(args.out, exist_ok=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        overwrite_output_dir=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,
        warmup_ratio=0.10,
        weight_decay=0.01,
        max_grad_norm=0.3,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_first_step=True,
        logging_steps=50,

        predict_with_generate=True,
        generation_num_beams=2,           # lebih cepat saat eval
        generation_max_length=args.max_tgt,

        fp16=False, bf16=False,           # mikro-finetune stabil di fp32
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )

    trainer.train()
    print("[INFO] Best checkpoint:", trainer.state.best_model_checkpoint)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("[OK] Saved adapter ->", args.out)

if __name__ == "__main__":
    main()
