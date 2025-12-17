# decode_grid_eval.py
import os, json, time
import numpy as np
from typing import Optional, List, Dict, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from peft import PeftModel
import evaluate
import torch

# ========= PATHS (SESUAIKAN) =========
BASE_DIR   = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_IN = "C:\cendol-mt5\outputs\pap2id_clean_polish\checkpoint-144"  # adapter terbaik saat ini
OUT_DIR    = "C:\cendol-mt5\outputs\decode_grid"

VALID_PATH = "data_splits_clean/valid.csv"
TEST_PATH  = "data_splits_clean/test.csv"

# ========= KOLOM =========
COL_INST_OPT = ["instruction", "instruksi"]
COL_IN_OPT   = ["input", "source", "src", "text"]
COL_OUT_OPT  = ["output", "target", "tgt", "label"]

# ========= GRID =========
GRID_NUM_BEAMS       = [4, 6]
GRID_LEN_PENALTY     = [0.90, 0.95, 1.00]
GRID_NGRAM           = [3]
GRID_MIN_NEW_TOKENS  = [4, 6]
MAX_NEW_TOKENS       = 64
BAD_WORDS            = ["Instruksi:", "Teks:", "Keluaran:", "Instruksi"]

BATCH_SIZE_GEN = 8  # naikin kalau VRAM cukup
SEED = 42

def detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".csv"): return "csv"
    if p.endswith(".tsv") or p.endswith(".tab"): return "tsv"
    if p.endswith(".json") or p.endswith(".jsonl"): return "json"
    raise ValueError(f"Format tidak didukung: {path}")

def load_split(path: str) -> Tuple[Dataset, str, str, Optional[str]]:
    fmt = detect_format(path)
    if fmt == "csv":
        ds = load_dataset("csv", data_files={"data": path})["data"]
    elif fmt == "tsv":
        ds = load_dataset("csv", data_files={"data": path}, delimiter="\t")["data"]
    else:
        ds = load_dataset("json", data_files={"data": path})["data"]

    cols = ds.column_names
    def pick(cands):
        low = [c.lower() for c in cols]
        for c in cands:
            if c.lower() in low:
                return cols[low.index(c.lower())]
        return None

    col_in   = pick(COL_IN_OPT)
    col_out  = pick(COL_OUT_OPT)
    col_inst = pick(COL_INST_OPT)
    if not col_in or not col_out:
        raise KeyError(f"Kolom input/target tidak ditemukan. Kolom: {cols}")
    return ds, col_in, col_out, col_inst

def make_prompt(inst: Optional[str], txt: str) -> str:
    if inst is None or str(inst).strip()=="":
        inst = "Terjemahkan ke bahasa Indonesia baku."
    return f"Instruksi: {inst}\nTeks: {txt}\nKeluaran:"

@torch.no_grad()
def batched_generate(model, tok, inputs, gen_kwargs, device):
    outs = []
    for i in range(0, len(inputs), BATCH_SIZE_GEN):
        batch = inputs[i:i+BATCH_SIZE_GEN]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(device) for k,v in enc.items()}
        gen_ids = model.generate(**enc, **gen_kwargs)
        dec = tok.batch_decode(gen_ids, skip_special_tokens=True)
        outs.extend([s.strip() for s in dec])
    return outs

def main():
    assert torch.cuda.is_available(), "CUDA tidak tersedia."
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda"

    print("[INFO] Load tokenizer & base")
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_DIR, use_safetensors=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    base.config.use_cache = False

    print("[INFO] Load adapter")
    model = PeftModel.from_pretrained(base, ADAPTER_IN, is_trainable=False, local_files_only=True).to(device)
    model.eval()

    # bad words ids (opsional)
    bad_ids = [tok.encode(w, add_special_tokens=False) for w in BAD_WORDS if w]

    print("[INFO] Load VALID & TEST")
    ds_val, cin_v, cout_v, cins_v = load_split(VALID_PATH)
    ds_tst, cin_t, cout_t, cins_t = load_split(TEST_PATH)

    # siapkan teks valid
    prompts_val = [make_prompt(ds_val[cins_v][i] if cins_v else None, ds_val[cin_v][i]) for i in range(len(ds_val))]
    refs_val = [[str(ds_val[cout_v][i]).strip()] for i in range(len(ds_val))]

    bleu = evaluate.load("sacrebleu"); chrf = evaluate.load("chrf")

    results = []
    print("[INFO] Start GRID on VALID …")
    for beams in GRID_NUM_BEAMS:
        for lp in GRID_LEN_PENALTY:
            for ng in GRID_NGRAM:
                for mn in GRID_MIN_NEW_TOKENS:
                    gen_kwargs = dict(
                        num_beams=beams,
                        length_penalty=lp,
                        no_repeat_ngram_size=ng,
                        min_new_tokens=mn,
                        max_new_tokens=MAX_NEW_TOKENS,
                        early_stopping=True,
                        bad_words_ids=bad_ids if bad_ids else None,
                    )
                    t0 = time.time()
                    preds = batched_generate(model, tok, prompts_val, gen_kwargs, device)
                    b = bleu.compute(predictions=preds, references=refs_val)["score"]
                    c = chrf.compute(predictions=preds, references=refs_val)["score"]
                    dt = time.time()-t0
                    rec = {"beams":beams,"len_pen":lp,"no_repeat":ng,"min_new":mn,
                           "max_new":MAX_NEW_TOKENS,"BLEU":b,"chrF":c,"sec":dt}
                    results.append(rec)
                    print(rec)

    # sort: BLEU desc, tie by chrF desc
    results_sorted = sorted(results, key=lambda r: (r["BLEU"], r["chrF"]), reverse=True)
    best = results_sorted[0]
    print("\n=== BEST on VALID ===")
    print(best)

    # simpan hasil grid
    with open(os.path.join(OUT_DIR, "decode_grid_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    # run TEST dengan setting terbaik
    print("\n[INFO] Run TEST with best settings …")
    prompts_test = [make_prompt(ds_tst[cins_t][i] if cins_t else None, ds_tst[cin_t][i]) for i in range(len(ds_tst))]
    refs_test = [[str(ds_tst[cout_t][i]).strip()] for i in range(len(ds_tst))]

    best_kwargs = dict(
        num_beams=best["beams"],
        length_penalty=best["len_pen"],
        no_repeat_ngram_size=best["no_repeat"],
        min_new_tokens=best["min_new"],
        max_new_tokens=MAX_NEW_TOKENS,
        early_stopping=True,
        bad_words_ids=bad_ids if bad_ids else None,
    )
    preds_test = batched_generate(model, tok, prompts_test, best_kwargs, device)
    b_t = bleu.compute(predictions=preds_test, references=refs_test)["score"]
    c_t = chrf.compute(predictions=preds_test, references=refs_test)["score"]

    summary = {"best_valid": best, "test_bleu": b_t, "test_chrF": c_t, "N_test": len(ds_tst)}
    print("\n=== SUMMARY ===")
    print(summary)

    with open(os.path.join(OUT_DIR, "decode_grid_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # simpan prediksi test
    pred_path = os.path.join(OUT_DIR, "test_predictions.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in preds_test:
            f.write(p+"\n")
    print("[OK] Saved:", pred_path)

if __name__ == "__main__":
    main()
