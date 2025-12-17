# audit_clean_split_no_sklearn.py
import os, re, json, argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# === CONFIG: ganti kolom sesuai datamu ===
SRC_COL = "source"
TGT_COL = "target"
INST_COL = "instruction"  # kalau tidak ada, akan diisi default

MIN_TGT_TOKENS = 3
LEN_RATIO_LOW, LEN_RATIO_HIGH = 0.5, 2.0
BAD_PROMPT_TOKENS = ["Instruksi:", "Teks:", "Keluaran:", "Instruksi"]

# proporsi split (total 100%)
TEST_FRAC  = 0.15
VALID_FRAC = 0.10
TRAIN_FRAC = 1.0 - TEST_FRAC - VALID_FRAC  # 0.75

def normalize_punct(s: str) -> str:
    s = s.replace("…", "...").replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_junk(s: str) -> bool:
    if not s or str(s).strip()=="":
        return True
    cleaned = re.sub(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\s\.\,\!\?\-\:\;\'\"\(\)\[\]\/%]", "", str(s))
    return len(cleaned) > len(str(s)) * 0.3

def tokenize_simple(s: str):
    return [t for t in re.split(r"\s+", s.strip()) if t]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # deteksi kolom
    cols_low = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n.lower() in cols_low:
                return cols_low[n.lower()]
        return None
    src = pick([SRC_COL,"src","input","text"])
    tgt = pick([TGT_COL,"tgt","output","label"])
    inst= pick([INST_COL,"instruksi"])
    if not src or not tgt:
        raise ValueError(f"Kolom source/target tidak ditemukan. Kolom: {list(df.columns)}")
    df = df.rename(columns={src:SRC_COL, tgt:TGT_COL})
    if inst and inst != INST_COL:
        df = df.rename(columns={inst:INST_COL})
    if INST_COL not in df.columns:
        df[INST_COL] = "Terjemahkan ke bahasa Indonesia baku."
    return df[[INST_COL, SRC_COL, TGT_COL]]

def stratify_bucket(lengths: np.ndarray):
    """Bucket sederhana berdasar panjang source (token)."""
    bins = np.array([0, 8, 16, 32, 64, 10**9], dtype=np.float32)
    idx = np.digitize(lengths, bins, right=True)  # 0..4
    return idx

def stratified_split(df: pd.DataFrame, valid_frac=VALID_FRAC, test_frac=TEST_FRAC, seed=42):
    """Split manual tanpa sklearn: stratified by source length bucket."""
    rng = np.random.RandomState(seed)
    src_tokens = df[SRC_COL].apply(lambda s: len(tokenize_simple(s))).values.astype(np.int32)
    buckets = stratify_bucket(src_tokens)  # 0..4

    # index per bucket
    bucket_to_idx = defaultdict(list)
    for i, b in enumerate(buckets):
        bucket_to_idx[b].append(i)

    n = len(df)
    all_train, all_valid, all_test = [], [], []

    for b, idxs in bucket_to_idx.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)

        nb = len(idxs)
        n_test  = int(round(nb * test_frac))
        n_valid = int(round(nb * valid_frac))
        n_train = nb - n_test - n_valid
        if n_train < 0:  # fallback jika bucket kecil sekali
            n_train = max(0, nb - n_test - n_valid)
        # potong aman
        n_test = min(n_test, nb)
        n_valid = min(n_valid, nb - n_test)
        n_train = nb - n_test - n_valid

        test_idx  = idxs[:n_test]
        valid_idx = idxs[n_test:n_test+n_valid]
        train_idx = idxs[n_test+n_valid:]

        all_test.append(test_idx)
        all_valid.append(valid_idx)
        all_train.append(train_idx)

    train_idx = np.concatenate(all_train) if all_train else np.array([], dtype=np.int64)
    valid_idx = np.concatenate(all_valid) if all_valid else np.array([], dtype=np.int64)
    test_idx  = np.concatenate(all_test)  if all_test  else np.array([], dtype=np.int64)

    # pastikan tdk overlap
    def unique(a): return np.unique(a)
    train_idx, valid_idx, test_idx = unique(train_idx), unique(valid_idx), unique(test_idx)

    # jaga-jaga drop kebocoran exact source antar split
    def lower_src(df_sub): return set(df_sub[SRC_COL].str.lower())
    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    leak_tv = lower_src(train_df) & lower_src(valid_df)
    leak_tt = lower_src(train_df) & lower_src(test_df)
    leak_vt = lower_src(valid_df) & lower_src(test_df)
    leaks = leak_tv | leak_tt | leak_vt
    if leaks:
        valid_df = valid_df[~valid_df[SRC_COL].str.lower().isin(leaks)]
        test_df  = test_df[~test_df[SRC_COL].str.lower().isin(leaks)]

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = load_csv(args.input)

    n0 = len(df)
    # 1) normalisasi & buang junk
    df[SRC_COL] = df[SRC_COL].astype(str).apply(normalize_punct)
    df[TGT_COL] = df[TGT_COL].astype(str).apply(normalize_punct)
    df = df[~df[SRC_COL].apply(is_junk)]
    df = df[~df[TGT_COL].apply(is_junk)]
    # 2) buang source==target
    df = df[df[SRC_COL].str.lower() != df[TGT_COL].str.lower()]
    # 3) target >= MIN_TGT_TOKENS
    df["tgt_tok"] = df[TGT_COL].apply(tokenize_simple)
    df = df[df["tgt_tok"].apply(lambda t: len(t) >= MIN_TGT_TOKENS)]
    # 4) rasio panjang
    df["src_tok"] = df[SRC_COL].apply(tokenize_simple)
    src_len = df["src_tok"].apply(len).replace(0, 1)
    tgt_len = df["tgt_tok"].apply(len)
    df["len_ratio"] = tgt_len / src_len
    df = df[(df["len_ratio"] >= LEN_RATIO_LOW) & (df["len_ratio"] <= LEN_RATIO_HIGH)]
    # 5) cegah kebocoran prompt di target
    bad_re = re.compile("|".join([re.escape(x) for x in BAD_PROMPT_TOKENS]), re.IGNORECASE)
    df = df[~df[TGT_COL].str.contains(bad_re)]
    # 6) dedup exact
    df = df.drop_duplicates(subset=[SRC_COL, TGT_COL])
    # 7) near-dup: source sama, target beda → pilih yang paling panjang
    df = df.sort_values(by=[SRC_COL, TGT_COL], key=lambda s: s.str.len())
    df = df.drop_duplicates(subset=[SRC_COL], keep="last")
    # bersihkan kolom sementara
    df = df[[INST_COL, SRC_COL, TGT_COL]].reset_index(drop=True)

    n_clean = len(df)
    print(f"[INFO] Cleaned: {n0} -> {n_clean} rows")

    # === Stratified split (tanpa sklearn) ===
    train_df, valid_df, test_df = stratified_split(df, VALID_FRAC, TEST_FRAC, seed=42)

    # simpan
    train_p = os.path.join(args.out_dir, "train.csv")
    valid_p = os.path.join(args.out_dir, "valid.csv")
    test_p  = os.path.join(args.out_dir, "test.csv")
    train_df.to_csv(train_p, index=False)
    valid_df.to_csv(valid_p, index=False)
    test_df.to_csv(test_p, index=False)

    rep = {
        "before_rows": int(n0),
        "after_rows": int(n_clean),
        "train": len(train_df), "valid": len(valid_df), "test": len(test_df),
        "avg_src_len": float(df[SRC_COL].apply(lambda s: len(tokenize_simple(s))).mean()),
        "avg_tgt_len": float(df[TGT_COL].apply(lambda s: len(tokenize_simple(s))).mean()),
    }
    with open(os.path.join(args.out_dir, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("[OK] Saved splits to:", args.out_dir)
    print(rep)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path ke dataset mentah (CSV)")
    ap.add_argument("--out_dir", default="data_splits_clean", help="Folder output splits")
    args = ap.parse_args()
    main(args)
