#!/usr/bin/env python3
# eval_bleu_chrf_to_csv_with_analysis.py
"""
Evaluate BLEU & chrF for a local base model + LoRA adapter; save per-sample CSV with ground_truth + analysis column,
and a one-row summary CSV.

Usage examples:
 python eval_bleu_chrf_to_csv_with_analysis.py
 python eval_bleu_chrf_to_csv_with_analysis.py --adapter "C:/cendol-mt5/outputs/pap2id_phase8_fast_e10/checkpoint-168" --split test
"""

import os
import argparse
from pathlib import Path
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate

# ---------------- DEFAULTS (sesuaikan bila perlu) ----------------
DEFAULT_BASE    = r"C:\ML\models\cendol-mt5-base-inst"
DEFAULT_ADAPTER = r"C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168"
DEFAULT_SPLITS  = r"C:\cendol-mt5\data_splits_json\json"
DEFAULT_SPLIT   = "test"

DECODE_CFG = dict(
    num_beams=6,
    length_penalty=0.95,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=64,
    do_sample=False
)

# ---------------- Helpers ----------------
def normalize_path(p: str) -> str:
    # convert to Path-friendly string (handles backslashes)
    return str(Path(p))

def check_adapter(adapter_dir: str):
    p = Path(adapter_dir)
    if not p.exists() or not p.is_dir():
        return False, f"Adapter folder tidak ditemukan: {adapter_dir}"
    # common adapter files
    has_config = (p / "adapter_config.json").exists() or (p / "peft_config.json").exists() or (p / "config.json").exists()
    has_weights = any((p / name).exists() for name in ("adapter_model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"))
    if not has_config:
        return False, "Tidak ditemukan adapter_config.json (atau peft_config.json) di folder adapter."
    if not has_weights:
        return False, "Tidak ditemukan file bobot adapter (adapter_model.safetensors atau pytorch_model.bin) di folder adapter."
    return True, "OK"

def make_prompt(inst: str, src: str) -> str:
    if not inst or inst.strip() == "":
        inst = "Terjemahkan ke bahasa Indonesia baku."
    return f"Instruksi: {inst}\nTeks: {src}\nKeluaran:"

def load_json_splits(splits_dir: str):
    files = {
        "train": os.path.join(splits_dir, "train.json"),
        "validation": os.path.join(splits_dir, "valid.json"),
        "test": os.path.join(splits_dir, "test.json"),
    }
    for k,p in files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split file missing: {p}")
    ds = load_dataset("json", data_files=files)
    def _norm(ex):
        inst = ex.get("instruction") or "Terjemahkan ke bahasa Indonesia baku."
        inp  = ex.get("input", "")
        out  = ex.get("output", "")
        return {"instruction": str(inst), "input": str(inp), "output": str(out)}
    ds = ds.map(_norm)
    def _valid(x): return (x["input"].strip()!="") and (x["output"].strip()!="")
    ds["train"] = ds["train"].filter(_valid)
    ds["validation"] = ds["validation"].filter(_valid)
    ds["test"] = ds["test"].filter(_valid)
    return ds

def generate_all(model, tok, dataset_split, device, decode_cfg, batch_size=8, max_source_len=256):
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    model.eval()
    preds, refs, inputs, insts = [], [], [], []
    n = len(dataset_split)
    for i in range(0, n, batch_size):
        batch = dataset_split[i:i+batch_size]
        prompts = [make_prompt(inst, inp) for inst, inp in zip(batch["instruction"], batch["input"])]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_source_len).to(device)
        with torch.no_grad():
            gen_ids = model.generate(**enc, **decode_cfg)
        outs = tok.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend([o.strip() for o in outs])
        refs.extend([[t.strip()] for t in batch["output"]])
        inputs.extend([s.strip() for s in batch["input"]])
        insts.extend([s.strip() for s in batch["instruction"]])
    bleu_s = sacrebleu.compute(predictions=preds, references=refs)["score"]
    chrf_s = chrf.compute(predictions=preds, references=refs)["score"]
    return bleu_s, chrf_s, inputs, preds, refs, insts

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Eval BLEU & chrF and save CSV (base + LoRA adapter) with analysis column")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Path ke folder base model (local)")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Path ke folder adapter LoRA (local)")
    parser.add_argument("--splits", default=DEFAULT_SPLITS, help="Folder yang berisi train.json, valid.json, test.json")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train","validation","test"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_beams", type=int, default=DECODE_CFG["num_beams"])
    parser.add_argument("--max_new_tokens", type=int, default=DECODE_CFG["max_new_tokens"])
    args = parser.parse_args()

    base_dir = normalize_path(args.base)
    adapter_dir = normalize_path(args.adapter)
    splits_dir = normalize_path(args.splits)
    device = args.device

    print("[INFO] paths:")
    print("  base   :", base_dir)
    print("  adapter:", adapter_dir)
    print("  splits :", splits_dir)
    print("  device :", device)

    if not Path(base_dir).is_dir():
        raise FileNotFoundError(f"Base model folder tidak ditemukan: {base_dir}")
    ok,msg = check_adapter(adapter_dir)
    if not ok:
        raise FileNotFoundError(f"Adapter problem: {msg}")
    if not Path(splits_dir).is_dir():
        raise FileNotFoundError(f"Splits folder tidak ditemukan: {splits_dir} (harus berisi train.json valid.json test.json)")

    # load tokenizer and base model
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    print("[INFO] load tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_dir, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "right"

    print("[INFO] load base model ...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_dir, local_files_only=True, torch_dtype=dtype)
    base_model.to(device)
    base_model.config.use_cache = False
    if getattr(base_model.config, "decoder_start_token_id", None) is None:
        base_model.config.decoder_start_token_id = tokenizer.pad_token_id

    print("[INFO] attach LoRA adapter ...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False, local_files_only=True)
    except Exception as e:
        raise RuntimeError("Gagal memuat adapter LoRA. Periksa folder adapter. Error: " + repr(e))
    model.to(device)

    # prepare decode cfg
    decode_cfg = DECODE_CFG.copy()
    decode_cfg["num_beams"] = args.num_beams
    decode_cfg["max_new_tokens"] = args.max_new_tokens

    # load splits
    print("[INFO] load splits ...")
    ds = load_json_splits(splits_dir)
    print(f"[INFO] sizes -> train: {len(ds['train'])} | valid: {len(ds['validation'])} | test: {len(ds['test'])}")

    eval_ds = ds[args.split]
    print(f"[INFO] evaluating split {args.split} (n={len(eval_ds)}) ...")
    bleu_s, chrf_s, inputs, preds, refs, insts = generate_all(model, tokenizer, eval_ds, device, decode_cfg, batch_size=args.batch)

    print("---- Results ----")
    print(f"BLEU ({args.split}): {bleu_s:.2f}")
    print(f"chrF ({args.split}): {chrf_s:.2f}")

    # CSV outputs
    out_dir = Path(splits_dir)
    samples_csv = out_dir / f"eval_samples_{args.split}.csv"
    summary_csv = out_dir / f"eval_summary_{args.split}.csv"

    # write samples CSV with separate ground_truth and analysis column (analysis empty)
    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "instruction", "input", "prediction", "ground_truth", "references", "analysis"])
        for i, (inst, inp, pred, r) in enumerate(zip(insts, inputs, preds, refs), start=1):
            refs_join = " || ".join(r)
            ground_truth = r[0] if len(r) > 0 else ""
            writer.writerow([i, inst, inp, pred, ground_truth, refs_join, ""])
    print("[INFO] Saved per-sample CSV:", str(samples_csv))

    # write summary CSV
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n_samples", "bleu", "chrf"])
        writer.writerow([args.split, len(inputs), f"{bleu_s:.2f}", f"{chrf_s:.2f}"])
    print("[INFO] Saved summary CSV:", str(summary_csv))

if __name__ == "__main__":
    main()
