# infer_with_knn_tm_v3.py
import os, json, argparse, re
import numpy as np
from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# ===== Default PATH (bisa override via argumen CLI) =====
DEFAULT_BASE   = "C:\ML\models\cendol-mt5-base-inst"
DEFAULT_ADAPT  = "C:\cendol-mt5\outputs\pap2id_clean_polish\checkpoint-144"
DEFAULT_TRAIN  = "C:\cendol-mt5\data_splits_clean/train.csv"

# ===== Decoding config (hasil grid terbaik BLEU) =====
GEN_CFG = dict(
    num_beams=6,
    length_penalty=0.95,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=64,
    early_stopping=True,
)
BAD_WORDS = ["Instruksi:", "Teks:", "Keluaran:", "Instruksi", "Baiklah", "Baiklah,"]

MAX_SRC_LEN = 256

# ===== Column guesses =====
COL_INST = ["instruction","instruksi"]
COL_IN   = ["input","source","src","text"]
COL_OUT  = ["output","target","tgt","label"]

# ---------- helpers ----------
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

_punct_re = re.compile(r"[^\w\s.,:;!?/\\-]")
_space_re = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _punct_re.sub("", s)
    s = _space_re.sub(" ", s)
    return s

def jaccard_tokens(a: str, b: str) -> float:
    A = set(a.split()); B = set(b.split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def length_ratio(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0: return 0.0
    return min(la, lb) / max(la, lb)

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
    srcs = [normalize_text(str(x)) for x in ds[col_in]]
    tgts = [str(y).strip() for y in ds[col_out]]
    insts = [str(z).strip() if col_ins and z is not None else "" for z in (ds[col_ins] if col_ins else [None]*len(srcs))]
    return srcs, tgts, insts

def build_tfidf_word(corpus: List[str]):
    vec = TfidfVectorizer(lowercase=False, analyzer="word", ngram_range=(1,2), min_df=1)
    mat = vec.fit_transform(corpus)
    return vec, mat

def build_tfidf_char(corpus: List[str]):
    vec = TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(3,5), min_df=1)
    mat = vec.fit_transform(corpus)
    return vec, mat

def knn_top(vec, mat, query_norm: str, topk: int = 1):
    qv = vec.transform([query_norm])
    sims = cosine_similarity(qv, mat)[0]
    idxs = np.argsort(-sims)[:topk]
    return [(int(i), float(sims[i])) for i in idxs]

def make_prompt(inst: Optional[str], txt: str) -> str:
    inst = (inst.strip() if inst else "Terjemahkan ke bahasa Indonesia baku.")
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
    # IO
    ap.add_argument("--base_dir", type=str, default=DEFAULT_BASE)
    ap.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPT)
    ap.add_argument("--train_path", type=str, default=DEFAULT_TRAIN)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--input_file", type=str, default=None)
    ap.add_argument("--out_json", type=str, default="infer_knn_tm_results.json")
    # TM options
    ap.add_argument("--sim", choices=["word","char","both"], default="both")
    ap.add_argument("--threshold", type=float, default=0.72)
    ap.add_argument("--min_jaccard", type=float, default=0.25)
    ap.add_argument("--min_len_ratio", type=float, default=0.60)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # — path hygiene —
    base_dir   = os.path.normpath(args.base_dir)
    adapter_in = os.path.normpath(args.adapter_dir)
    train_path = os.path.normpath(args.train_path)
    if not os.path.isdir(base_dir):   raise FileNotFoundError(f"Base model folder tidak ditemukan: {base_dir}")
    if not os.path.isdir(adapter_in): raise FileNotFoundError(f"Adapter folder tidak ditemukan: {adapter_in}")
    if not os.path.isfile(train_path):raise FileNotFoundError(f"Train file tidak ditemukan: {train_path}")
    assert torch.cuda.is_available(), "CUDA tidak tersedia."

    device = "cuda"

    # 1) Load TM
    print("[INFO] Load training TM …")
    train_src, train_tgt, _ = load_train_tm(train_path)
    print(f"[INFO] TM size: {len(train_src)}")

    # 2) Fit vectorizers
    print(f"[INFO] Fit TF-IDF sim='{args.sim}' …")
    if args.sim in ("word","both"):
        vec_w, mat_w = build_tfidf_word(train_src)
    else:
        vec_w, mat_w = None, None
    if args.sim in ("char","both"):
        vec_c, mat_c = build_tfidf_char(train_src)
    else:
        vec_c, mat_c = None, None

    # 3) Inputs
    queries_raw = []
    if args.inputs:
        queries_raw.extend(args.inputs)
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries_raw.extend([ln.strip() for ln in f if ln.strip()!=""])
    if not queries_raw:
        queries_raw = [
            "Tabea, mari katong pigi pasar besok…",
            "Awam kapito… (contoh kalimat bahasa Papua pendek)",
        ]
    queries_norm = [normalize_text(q) for q in queries_raw]
    print(f"[INFO] #queries = {len(queries_raw)}")

    # 4) Load model
    print("[INFO] Load tokenizer & model …")
    tok = AutoTokenizer.from_pretrained(base_dir, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_in, is_trainable=False, local_files_only=True).to(device)
    model.eval()

    outputs = []
    to_fallback_prompts = []
    fallback_meta = []

    # 5) Inference loop
    for q_raw, q_norm in zip(queries_raw, queries_norm):
        best_idx, best_score = -1, -1.0
        top1_w = top1_c = None

        if args.sim in ("word","both"):
            topk_w = knn_top(vec_w, mat_w, q_norm, topk=args.topk)
            top1_w = topk_w[0]
            best_idx, best_score = top1_w
        if args.sim in ("char","both"):
            topk_c = knn_top(vec_c, mat_c, q_norm, topk=args.topk)
            top1_c = topk_c[0]
            if top1_c[1] > best_score:
                best_idx, best_score = top1_c

        if args.debug:
            print(f"\nQ: {q_raw}")
            if top1_w: print("  top1_word:", top1_w)
            if top1_c: print("  top1_char:", top1_c)
            print(f"  best_idx={best_idx}, score={best_score:.3f}")

        accept_tm = False
        dbg_reason = ""
        if best_score >= args.threshold and best_idx >= 0:
            cand_src_norm = train_src[best_idx]
            jacc = jaccard_tokens(q_norm, cand_src_norm)
            lr   = length_ratio(q_norm, cand_src_norm)
            if args.debug:
                print(f"  jacc={jacc:.3f}, len_ratio={lr:.2f}")
            if jacc >= args.min_jaccard and lr >= args.min_len_ratio:
                accept_tm = True
                dbg_reason = f"tm(score={best_score:.3f}, jacc={jacc:.3f}, lenr={lr:.2f})"
            else:
                dbg_reason = f"fallback(score={best_score:.3f}, jacc={jacc:.3f}, lenr={lr:.2f})"
        else:
            dbg_reason = f"fallback(score={best_score:.3f})"

        if accept_tm:
            pred = train_tgt[best_idx]
            outputs.append({"src": q_raw, "pred": pred, "how": dbg_reason, "score": best_score})
        else:
            prompt = make_prompt("", q_raw)
            to_fallback_prompts.append(prompt)
            fallback_meta.append({"src": q_raw, "score": best_score, "why": dbg_reason})

    # 6) Fallback batch to model
    if to_fallback_prompts:
        gens = model_generate(model, tok, to_fallback_prompts, device)
        for meta, pred in zip(fallback_meta, gens):
            outputs.append({"src": meta["src"], "pred": pred, "how": meta["why"], "score": meta["score"]})

    # 7) Preserve input order
    ordered = []
    used = [False]*len(outputs)
    for q in queries_raw:
        take = None
        for j, o in enumerate(outputs):
            if not used[j] and o["src"] == q:
                take = (j, o); break
        if take is None:
            ordered.append({"src": q, "pred": "", "how": "missing", "score": 0.0})
        else:
            used[take[0]] = True
            ordered.append(take[1])

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    print("[OK] Saved ->", os.path.normpath(args.out_json))

    for item in ordered:
        print(f"SRC: {item['src']}\nPRED({item['how']}): {item['pred']}\n---")

if __name__ == "__main__":
    # pip install scikit-learn datasets transformers peft
    main()
