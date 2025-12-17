# auto_build_hard_train.py  (VERSI FIX & FAST)
import os, csv, argparse, json
from typing import List, Optional, Tuple
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# NEW: gunakan sacrebleu untuk skor chrF per-example
# pip install sacrebleu
from sacrebleu.metrics import CHRF

# ======== Default path (ubah sesuai punyamu) ========
DEFAULT_BASE   = "C:\ML\models\cendol-mt5-base-inst"
DEFAULT_ADAPT  = "C:\cendol-mt5\outputs\pap2id_clean_phase5_booster\checkpoint-96"
DEFAULT_SPLIT_DIR = "C:\cendol-mt5\data_splits_clean"  # berisi train.csv, valid.csv, test.csv
DEFAULT_OUT_CSV   = "C:\cendol-mt5\hard_train.csv"

# Konfigurasi decoding (bisa diringankan via --fast)
GEN_CFG = dict(
    num_beams=4,
    length_penalty=0.95,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=64,
    early_stopping=True,
)

BAD_WORDS = ["Instruksi:", "Teks:", "Keluaran:", "Instruksi", "Baiklah", "Baiklah,"]
DEFAULT_INSTRUCTION = "Terjemahkan ke bahasa Indonesia baku."
MAX_SRC_LEN = 256

COL_IN_CANDS  = ["input", "source", "src", "text"]
COL_OUT_CANDS = ["output", "target", "tgt", "label"]
COL_INS_CANDS = ["instruction", "instruksi"]

def pick_col(ds: Dataset, cands: List[str]) -> Optional[str]:
    cols = ds.column_names
    low = [c.lower() for c in cols]
    for name in cands:
        if name.lower() in low:
            return cols[low.index(name.lower())]
    return None

def detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith(".csv"): return "csv"
    if p.endswith(".tsv") or p.endswith(".tab"): return "tsv"
    if p.endswith(".json") or p.endswith(".jsonl"): return "json"
    raise ValueError(f"Format file tidak didukung: {path}")

def load_splits(split_dir: str) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    def _load_one(pth: str) -> Optional[Dataset]:
        if not os.path.isfile(pth): return None
        fmt = detect_format(pth)
        if fmt == "csv":  ds = load_dataset("csv", data_files={"d": pth})["d"]
        elif fmt == "tsv":ds = load_dataset("csv", data_files={"d": pth}, delimiter="\t")["d"]
        else:             ds = load_dataset("json", data_files={"d": pth})["d"]
        return ds

    train = _load_one(os.path.join(split_dir, "train.csv"))
    valid = _load_one(os.path.join(split_dir, "valid.csv"))
    test  = _load_one(os.path.join(split_dir, "test.csv"))

    if train is None:
        raise FileNotFoundError(f"train.csv tidak ditemukan di {split_dir}")
    return train, valid, test

def make_prompt(inst: Optional[str], txt: str, default_inst: str) -> str:
    inst = (inst.strip() if inst else default_inst)
    return f"Instruksi: {inst}\nTeks: {txt}\nKeluaran:"

def jaccard(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def len_ratio(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0: return 0.0
    return min(la, lb) / max(la, lb)

@torch.no_grad()
def generate_batch(model, tok, texts: List[str], device: str, use_fp16: bool) -> List[str]:
    bad_ids = [tok.encode(w, add_special_tokens=False) for w in BAD_WORDS if w]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SRC_LEN).to(device)
    if use_fp16:
        with torch.autocast("cuda", dtype=torch.float16):
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
    else:
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
    ap.add_argument("--base_dir", type=str, default=DEFAULT_BASE)
    ap.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPT)
    ap.add_argument("--split_dir", type=str, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--out_csv", type=str, default=DEFAULT_OUT_CSV)

    ap.add_argument("--top_n", type=int, default=250, help="berapa banyak hard cases yang diambil")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_test", action="store_true", help="ikutkan test split sebagai kandidat")

    # Heuristik seleksi
    ap.add_argument("--echo_jaccard_min", type=float, default=0.5, help="anggap echo jika Jaccard src-pred >= ini")
    ap.add_argument("--short_src_max_tok", type=int, default=6, help="anggap pendek jika token sumber <= ini")
    ap.add_argument("--chrF_cut", type=float, default=35.0, help="prioritaskan contoh dengan chrF <= ini")

    # NEW: mode cepat
    ap.add_argument("--fast", action="store_true", help="mode cepat: fp16 + beams=2 + max_new=48")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA tidak tersedia."
    device = "cuda"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load splits
    train, valid, test = load_splits(args.split_dir)

    # Deteksi kolom
    def detect_cols(ds: Dataset):
        cin  = pick_col(ds, COL_IN_CANDS)
        cout = pick_col(ds, COL_OUT_CANDS)
        cins = pick_col(ds, COL_INS_CANDS)
        if not cin or not cout:
            raise KeyError(f"Kolom input/target tidak ditemukan. Kolom: {ds.column_names}")
        return cin, cout, cins

    col_in, col_out, col_ins = detect_cols(train)

    # Gabungkan kandidat (train + valid [+ test opsional])
    cand = train
    parts = ["train"]
    if valid is not None:
        cand = concatenate_datasets([cand, valid])
        parts.append("valid")
    if args.include_test and test is not None:
        cand = concatenate_datasets([cand, test])
        parts.append("test")
    print(f"[INFO] Candidate pool from: {', '.join(parts)} | size={len(cand)}")

    # Load model
    print("[INFO] Load tokenizer & model …")
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True)

    use_fp16 = bool(args.fast)  # fast mode: fp16
    dtype = torch.float16 if use_fp16 else torch.float32

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_dir, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
    ).to(device)
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False, local_files_only=True).to(device)
    model.eval()

    # Fast mode: ringankan decoding
    if args.fast:
        GEN_CFG["num_beams"] = 2
        GEN_CFG["max_new_tokens"] = 48

    def batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield i, lst[i:i+bs]

    sources = [str(s) for s in cand[col_in]]
    targets = [str(t) for t in cand[col_out]]
    insts   = [str(z) if (col_ins and z is not None) else "" for z in (cand[col_ins] if col_ins else [None]*len(cand))]

    # Generate predictions
    preds = []
    for _, chunk in batches(list(zip(sources, insts)), args.batch_size):
        prompts = [make_prompt(ins, src, DEFAULT_INSTRUCTION) for src, ins in chunk]
        preds.extend(generate_batch(model, tok, prompts, device, use_fp16))

    # NEW: chrF per-example via sacrebleu
    chrf_metric = CHRF(word_order=2)  # chrF++ (lebih sensitif semantik)
    chrf_scores = np.array([chrf_metric.sentence_score(preds[i], [targets[i]]).score
                            for i in range(len(preds))], dtype=float)

    # Heuristik echo/pendek
    def token_len(x: str) -> int: return len(x.split())
    jac_src_pred = np.array([jaccard(sources[i], preds[i]) for i in range(len(sources))])
    lr_src_pred  = np.array([len_ratio(sources[i], preds[i]) for i in range(len(sources))])
    src_len      = np.array([token_len(s) for s in sources])

    # Skor kesulitan: rendah chrF + echo + pendek → diprioritaskan
    # semakin BESAR score -> semakin diprioritaskan untuk hard set
    score = (
        (args.chrF_cut - np.clip(chrf_scores, 0, args.chrF_cut))  # makin rendah chrF makin besar
        + 10.0 * (jac_src_pred >= args.echo_jaccard_min)          # echo bonus
        + 3.0  * (src_len <= args.short_src_max_tok)              # short bonus
        + 2.0  * (lr_src_pred >= 0.8)                             # mirip panjang → echo-ish
    )

    # Ambil top-N unik berdasarkan skor
    idxs = np.argsort(-score)[:args.top_n]
    picked = set()
    rows = []
    for i in idxs:
        src = sources[i].strip()
        tgt = targets[i].strip()
        if not src or not tgt: continue
        key = (src, tgt)
        if key in picked: continue
        picked.add(key)
        rows.append([src, tgt])

    print(f"[INFO] Selected hard cases: {len(rows)} (top_n={args.top_n})")

    # Simpan CSV
    out_path = os.path.normpath(args.out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input","output"])
        w.writerows(rows)
    print("[OK] Saved hard_train.csv ->", out_path)

    # Simpan ringkasan JSON (opsional)
    summary = {
        "size_pool": len(cand),
        "picked": len(rows),
        "top_n": args.top_n,
        "echo_jaccard_min": args.echo_jaccard_min,
        "short_src_max_tok": args.short_src_max_tok,
        "chrF_cut": args.chrF_cut,
        "fast_mode": args.fast,
        "gen_cfg": GEN_CFG,
    }
    with open(out_path.replace(".csv", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[OK] Saved summary ->", out_path.replace(".csv", "_summary.json"))

if __name__ == "__main__":
    # pip install -q sacrebleu datasets transformers peft evaluate
    main()
