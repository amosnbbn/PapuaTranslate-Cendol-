import argparse, os, json, random
import pandas as pd
from pathlib import Path

DEF_INSTR = "Terjemahkan ke bahasa Indonesia baku."

REQ_IVO = {"instruction", "input", "output"}
REQ_IO  = {"input", "output"}             # <-- baru: terima input/output saja
REQ_ST  = {"source", "target"}
ALT_INSTR = {"instr", "Instruction", "Instruksi", "instruction_text"}

def normalize_columns(df: pd.DataFrame, default_instruction: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    cols = set(df.columns)

    # A) instruction/input/output lengkap
    if REQ_IVO.issubset(cols):
        pass

    # B) source/target (+opsional instruction alternatif)
    elif REQ_ST.issubset(cols):
        df = df.rename(columns={"source": "input", "target": "output"})
        ins_found = None
        for k in ALT_INSTR:
            if k in df.columns:
                ins_found = k
                break
        if ins_found:
            df = df.rename(columns={ins_found: "instruction"})
        else:
            df["instruction"] = default_instruction

    # C) Hanya input/output (dengan atau tanpa 'id' atau kolom lain)
    elif REQ_IO.issubset(cols):
        if "instruction" not in df.columns:
            df["instruction"] = default_instruction

    else:
        raise ValueError(
            f"Kolom tidak sesuai. Ditemukan: {list(df.columns)}\n"
            f"Harus punya (instruction,input,output) ATAU (source,target [+optional instruction]) "
            f"ATAU minimal (input,output)."
        )

    # Pastikan tiga kolom final ada
    for k in ["instruction", "input", "output"]:
        if k not in df.columns:
            if k == "instruction":
                df["instruction"] = default_instruction
            else:
                raise ValueError(f"Kolom '{k}' tidak ditemukan setelah normalisasi.")

    # Cast string + trimming
    for k in ["instruction", "input", "output"]:
        df[k] = df[k].astype(str).fillna("").str.strip()

    # Buang baris kosong
    before = len(df)
    df = df[(df["input"] != "") & (df["output"] != "")]
    after_nonempty = len(df)

    # Dedup (unik per input; jika ganda ambil satu)
    df = df.drop_duplicates(subset=["input", "output"]).drop_duplicates(subset=["input"])
    after_dedup = len(df)

    print(f"[CLEAN] rows: {before} -> nonempty: {after_nonempty} -> dedup: {after_dedup}")
    return df[["instruction", "input", "output"]]

def save_csv(path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[OK] CSV  : {path}")

def save_json(path: str, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON : {path}")

def save_jsonl(path: str, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] JSONL: {path}")

def split_df(df: pd.DataFrame, seed: int, r_train: float, r_valid: float, r_test: float):
    assert abs((r_train + r_valid + r_test) - 1.0) < 1e-6, "proporsi harus jumlahnya 1.0"
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * r_train)
    n_valid = int(n * r_valid)
    n_test  = n - n_train - n_valid

    train = df.iloc[:n_train].reset_index(drop=True)
    valid = df.iloc[n_train:n_train+n_valid].reset_index(drop=True)
    test  = df.iloc[n_train+n_valid:].reset_index(drop=True)

    print(f"[SPLIT] total={n} -> train={len(train)} | valid={len(valid)} | test={len(test)}")
    return train, valid, test

def main():
    ap = argparse.ArgumentParser("Split dataset 1-file menjadi train/valid/test + ekspor CSV & JSON(JSONL).")
    ap.add_argument("--input_csv", required=True, help="CSV sumber (2300+ baris)")
    ap.add_argument("--out_dir", required=True, help="Folder output (mis. C:/cendol-mt5/data_splits_json)")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio",  type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--default_instruction", type=str, default=DEF_INSTR)
    ap.add_argument("--also_jsonl", action="store_true", help="Simpan juga JSONL")
    args = ap.parse_args()

    # load & normalize
    df = pd.read_csv(args.input_csv, encoding="utf-8", dtype=str)
    df = normalize_columns(df, args.default_instruction)

    # split
    train, valid, test = split_df(
        df, seed=args.seed,
        r_train=args.train_ratio, r_valid=args.valid_ratio, r_test=args.test_ratio
    )

    # path out
    out = Path(args.out_dir)
    out_csv  = out / "csv"
    out_json = out / "json"

    # save CSV
    save_csv(str(out_csv / "train.csv"), train)
    save_csv(str(out_csv / "valid.csv"), valid)
    save_csv(str(out_csv / "test.csv"),  test)

    # save JSON
    save_json(str(out_json / "train.json"), train.to_dict(orient="records"))
    save_json(str(out_json / "valid.json"), valid.to_dict(orient="records"))
    save_json(str(out_json / "test.json"),  test.to_dict(orient="records"))

    # optional JSONL
    if args.also_jsonl:
        out_jsonl = out / "jsonl"
        save_jsonl(str(out_jsonl / "train.jsonl"), train.to_dict(orient="records"))
        save_jsonl(str(out_jsonl / "valid.jsonl"), valid.to_dict(orient="records"))
        save_jsonl(str(out_jsonl / "test.jsonl"),  test.to_dict(orient="records"))

    # ringkasan
    avg_in = train["input"].str.split().apply(len).mean()
    avg_out = train["output"].str.split().apply(len).mean()
    print(f"[STATS] avg_len input(train) ~ {avg_in:.2f} | output(train) ~ {avg_out:.2f}")
    print("[DONE] Dataset siap dipakai untuk fine-tuning.")

if __name__ == "__main__":
    main()
