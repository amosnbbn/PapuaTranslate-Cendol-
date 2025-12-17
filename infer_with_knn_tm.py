# infer_with_knn_tm.py
import os, json, argparse
import numpy as np
from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from peft import PeftModel
import torch

# ====== PATHS (SESUAIKAN) ======
BASE_DIR   = r"C:\ML\models\cendol-mt5-base-inst"
ADAPTER_IN = r"C:\cendol-mt5\outputs\pap2id_clean_polish\checkpoint-144"  # adapter terbaik
TRAIN_PATH = r"data_splits_clean\train.csv"

# Kolom yang mungkin
COL_INST = ["instruction","instruksi"]
COL_IN   = ["input","source","src","text"]
COL_OUT  = ["output","target","tgt","label"]

# Decoding terbaik dari grid (best BLEU)
GEN_CFG = dict(
    num_beams=6,
    length_penalty=0.95,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=64,
    early_stopping=True,
)
BAD_WORDS = ["Instruksi:", "Teks:", "Keluaran:", "Instruksi"]

# TF-IDF / kNN
TOPK        = 1
THRESH_HIT  = 0.72   # jika cosine >= 0.72, pakai TM; tweak 0.68–0.80 sesuai preferensi
MAX_SRC_LEN = 256

def detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".csv"): return "csv"
    if p.endswith(".tsv") or p.endswith(".tab"): return "tsv"
    if p.endswith(".json") or p.endswith(".jsonl"): return "json"
    raise ValueError(f"Format tidak didukung: {path}")

def pick_col(ds: Dataset, candidates: List[str]) -> Optional[str]:
    cols = ds.column_names
    low = [c.lower() for c in cols]
    for name in candidates:
        if name.lower() in low:
            return cols[low.index(name.lower())]
    return None

def load_train_tm(path: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
    fmt = detect_format(path)
    if fmt == "csv":
        ds = load_dataset("csv", data_files={"data": path})["data"]
    elif fmt == "tsv":
        ds = load_dataset("csv", data_files={"data": path}, delimiter="\t")["data"]
    else:
        ds = load_dataset("json", data_files={"data": path})["data"]
    col_in  = pick_col(ds, COL_IN)
    col_out = pick_col(ds, COL_OUT)
    col_ins = pick_col(ds, COL_INST)
    if not col_in or not col_out:
        raise KeyError(f"Kolom input/target tidak ditemukan. Kolom: {ds.column_names}")
    srcs = [str(x).strip() for x in ds[col_in]]
    tgts = [str(y).strip() for y in ds[col_out]]
    insts = [str(z).strip() if col_ins and z is not None else "" for z in (ds[col_ins] if col_ins else [None]*len(srcs))]
    return srcs, tgts, insts

def build_vectorizer(corpus: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1,2), min_df=1)
    mat = vec.fit_transform(corpus)
    return vec, mat

def knn_tm_predict(vec: TfidfVectorizer, mat, train_src: List[str], train_tgt: List[str], query: str) -> Tuple[str, float]:
    qv = vec.transform([query])
    sims = cosine_similarity(qv, mat)[0]
    idx = int(np.argmax(sims))
    return (train_tgt[idx], float(sims[idx]))

def make_prompt(inst: Optional[str], txt: str) -> str:
    if not inst:
        inst = "Terjemahkan ke bahasa Indonesia baku."
    return f"Instruksi: {inst}\nTeks: {txt}\nKeluaran:"

@torch.no_grad()
def model_generate(model, tok, texts: List[str], device: str) -> List[str]:
    bad_ids = [tok.encode(w, add_special_tokens=False) for w in BAD_WORDS if w]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SRC_LEN).to(device)
    gen_ids = model.generate(
        **enc,
        num_beams=GEN_CFG["num_beams"],
        length_penalty=GEN_CFG["length_penalty"],
        no_repeat_ngram_size=GEN_CFG["no_repeat_ngram_size"],
        min_new_tokens=GEN_CFG["min_new_tokens"],
        max_new_tokens=GEN_CFG["max_new_tokens"],
        early_stopping=True,
        bad_words_ids=bad_ids if bad_ids else None,
    )
    return [s.strip() for s in tok.batch_decode(gen_ids, skip_special_tokens=True)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None, help="Kalimat sumber untuk diterjemahkan")
    ap.add_argument("--input_file", type=str, default=None, help="File txt, 1 kalimat per baris")
    ap.add_argument("--out_json", type=str, default="infer_knn_tm_results.json")
    ap.add_argument("--threshold", type=float, default=THRESH_HIT)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA tidak tersedia."
    device = "cuda"

    # 1) Load TM
    print("[INFO] Load training TM …")
    train_src, train_tgt, train_ins = load_train_tm(TRAIN_PATH)
    print(f"[INFO] TM size: {len(train_src)}")

    # 2) Build TF-IDF
    print("[INFO] Fit TF-IDF …")
    vec, mat = build_vectorizer(train_src)

    # 3) Inputs
    queries = []
    if args.inputs:
        queries.extend(args.inputs)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries.extend([ln.strip() for ln in f if ln.strip()!=""])
    if not queries:
        # Contoh default
        queries = [
            "Tabea, mari katong pigi pasar besok…",
            "Awam kapito… (contoh kalimat bahasa Papua pendek)",
        ]
    print(f"[INFO] #queries = {len(queries)}")

    # 4) Load model
    print("[INFO] Load tokenizer & model …")
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, ADAPTER_IN, is_trainable=False, local_files_only=True).to(device)
    model.eval()

    # 5) Predict
    outputs = []
    to_fallback = []
    fallback_map = {}  # index -> prompt text

    for i, q in enumerate(queries):
        # KNN TM
        tm_pred, score = knn_tm_predict(vec, mat, train_src, train_tgt, q)
        if score >= args.threshold:
            outputs.append({"src": q, "pred": tm_pred, "how": "tm", "score": score})
        else:
            # siapkan prompt + fallback ke model
            inst = ""  # kalau kamu punya default instruksi lain, taruh di sini
            prompt = make_prompt(inst, q)
            to_fallback.append((i, prompt, q, score))

    # 6) Fallback batch ke model
    if to_fallback:
        prompts = [p for _, p, _, _ in to_fallback]
        gens = model_generate(model, tok, prompts, device)
        for (i, _, src, score), pred in zip(to_fallback, gens):
            outputs.append({"src": src, "pred": pred, "how": "model", "score": score})

    # 7) Urutkan sesuai input awal
    # (outputs saat ini tidak dalam urutan input. Kita mapping kembali.)
    index_map = {o["src"]+"__"+str(k): k for k, o in enumerate(outputs)}
    # di sini kita tidak punya unique key selain teks; untuk aman jika ada duplikat query, tambahkan enumerasi
    # sederhana: rebuild sesuai urutan queries
    ordered = []
    used = set()
    for q in queries:
        # cari entri pertama yang cocok
        cand_idx = None
        for idx, o in enumerate(outputs):
            if idx in used: continue
            if o["src"] == q:
                cand_idx = idx; break
        if cand_idx is None:
            ordered.append({"src": q, "pred": "", "how": "missing", "score": 0.0})
        else:
            ordered.append(outputs[cand_idx]); used.add(cand_idx)

    # 8) Simpan
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    print("[OK] Saved ->", args.out_json)

    # 9) Tampilkan ringkas
    for item in ordered[:10]:
        print(f"SRC: {item['src']}\nPRED({item['how']}, {item['score']:.3f}): {item['pred']}\n---")

if __name__ == "__main__":
    main()
