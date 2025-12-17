# eval_predictions_vs_csv.py
import sys, json
import evaluate
import pandas as pd

CSV_PATH   = r"data\DATASET_CLEAN_NOEN.csv"   # ganti kalau beda
TGT_COL    = "target"                          # kolom referensi
PREDS_PATH = r"predictions.txt"                # file output dari inference

def main():
    df = pd.read_csv(CSV_PATH)
    refs = df[TGT_COL].astype(str).tolist()
    preds = [line.rstrip("\n") for line in open(PREDS_PATH, "r", encoding="utf-8")]

    if len(preds) != len(refs):
        print(f"[WARN] Panjang tidak cocok: preds={len(preds)} vs refs={len(refs)}")
        n = min(len(preds), len(refs))
        preds = preds[:n]; refs = refs[:n]

    bleu = evaluate.load("sacrebleu").compute(
        predictions=preds, references=[[r] for r in refs]
    )["score"]
    chrf = evaluate.load("chrf").compute(
        predictions=preds, references=[[r] for r in refs]
    )["score"]

    print(json.dumps({"BLEU": bleu, "chrF": chrf, "N": len(preds)}, indent=2))

if __name__ == "__main__":
    main()
