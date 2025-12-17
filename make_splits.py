# make_splits.py
import os, csv, json, hashlib, argparse
from typing import Optional, List, Tuple

def detect_format(path: str) -> str:
    p = path.lower()
    if p.endswith((".tsv", ".tab")): return "tsv"
    if p.endswith(".csv"): return "csv"
    if p.endswith((".jsonl",".json")): return "json"
    raise ValueError(f"Format tidak didukung: {path}")

def load_rows(path: str) -> List[dict]:
    fmt = detect_format(path)
    rows = []
    if fmt in ("csv","tsv"):
        delim = "\t" if fmt=="tsv" else ","
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for r in reader:
                rows.append({k.strip(): (v if v is not None else "") for k,v in r.items()})
    else:  # json/jsonl
        if path.lower().endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows = data
                else:
                    raise ValueError("File JSON harus berupa list objek.")
    return rows

# cari kolom fleksibel
def pick_col(names: List[str], candidates: List[str]) -> Optional[str]:
    low = [n.lower() for n in names]
    for c in candidates:
        if c.lower() in low:
            return names[low.index(c.lower())]
    return None

def normalize_space(s: str) -> str:
    return " ".join(str(s).split())

def hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path ke dataset (csv/tsv/json/jsonl)")
    ap.add_argument("--outdir", default="data_splits", help="Folder output")
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--valid_ratio", type=float, default=0.10)
    ap.add_argument("--min_tgt_words", type=int, default=5)
    args = ap.parse_args()

    assert abs(args.train_ratio + args.valid_ratio - 0.90) <= 1e-6 or args.train_ratio + args.valid_ratio < 1.0, \
        "Sisa akan otomatis jadi test_ratio = 1 - train - valid"

    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    assert test_ratio > 0, "Rasio tidak valid."

    os.makedirs(args.outdir, exist_ok=True)

    rows = load_rows(args.data)
    if not rows:
        raise RuntimeError("Dataset kosong.")

    # deteksi kolom
    names = list(rows[0].keys())
    col_inst = pick_col(names, ["instruction","instruksi"])
    col_in   = pick_col(names, ["input","source","src","text"])
    col_out  = pick_col(names, ["output","target","tgt","label"])
    if col_in is None or col_out is None:
        raise KeyError(f"Kolom input/output tidak ditemukan. Kolom tersedia: {names}")

    # cleaning + dedup (berbasis pasangan (inp,out))
    seen = set()
    clean = []
    for r in rows:
        inp = normalize_space(r.get(col_in, ""))
        out = normalize_space(r.get(col_out, ""))
        inst = normalize_space(r.get(col_inst, "")) if col_inst else ""

        if not inp or not out: 
            continue
        if inp == out:
            continue
        if len(out.split()) < args.min_tgt_words:
            continue

        key = (inp, out)
        if key in seen:
            continue
        seen.add(key)

        clean.append({
            "instruction": inst if inst else "Terjemahkan ke bahasa Indonesia baku.",
            "input": inp,
            "output": out
        })

    if not clean:
        raise RuntimeError("Semua data terfilter; longgarkan aturan atau cek file input.")

    # split deterministik berbasis hash input (mencegah kebocoran & reproducible)
    train, valid, test = [], [], []
    for ex in clean:
        p = hash01(ex["input"])
        if p < args.train_ratio:
            train.append(ex)
        elif p < args.train_ratio + args.valid_ratio:
            valid.append(ex)
        else:
            test.append(ex)

    # simpan CSV
    def save_csv(path: str, rows: List[dict]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["instruction","input","output"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    save_csv(os.path.join(args.outdir, "train.csv"), train)
    save_csv(os.path.join(args.outdir, "valid.csv"), valid)
    save_csv(os.path.join(args.outdir, "test.csv"),  test)

    print(f"[OK] Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
    print(f"[OK] Folder: {args.outdir}")

if __name__ == "__main__":
    main()
