# eval_valid_decode_best.py
import os, json, torch
from typing import List
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate
from tqdm import tqdm

# ===== CONFIG (sesuaikan) =====
BASE_DIR    = r"C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = r"C:\cendol-mt5\outputs\pap2id_lora_run2_stable_next"  # adapter terbaik/terbaru kamu

DATA_PATH   = r"data\DATASET_CLEAN_NOEN.csv"
SRC_COL     = "source"
TGT_COL     = "target"

USE_INSTRUCTION_FORMAT = True
INSTRUCTION = "Terjemahkan ke bahasa Indonesia baku."

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

# Decoding terbaik dari grid search kamu
NUM_BEAMS            = 4
LENGTH_PENALTY       = 1.0
NO_REPEAT_NGRAM_SIZE = 2
MIN_NEW_TOKENS       = 6
MAX_NEW_TOKENS       = 128
EARLY_STOPPING       = True

# Runtime (hemat VRAM)
DEVICE = "cuda"
INFER_PRECISION = "fp16"  # "fp16" (disarankan), "bf16" (jika GPU mendukung), atau "fp32"
BATCH_SIZE = 2            # kalau masih OOM → 1

# Validation split sama dengan training
VALID_SPLIT = 0.08
RANDOM_SEED = 42

OUT_DIRNAME = "decode_eval"
OUT_FILE    = "valid_predictions.txt"

def make_prompt(s: str) -> str:
    if USE_INSTRUCTION_FORMAT:
        return f"Instruksi: {INSTRUCTION}\nTeks: {s}\nKeluaran:"
    return s

def load_val_dataset(path: str) -> DatasetDict:
    ext = path.lower()
    if ext.endswith(".csv"):
        ds = load_dataset("csv", data_files={"train": path})
    elif ext.endswith((".tsv", ".tab")):
        ds = load_dataset("csv", data_files={"train": path}, delimiter="\t")
    else:
        ds = load_dataset("json", data_files={"train": path})

    def _ok(ex):
        s, t = ex.get(SRC_COL), ex.get(TGT_COL)
        return s is not None and t is not None and str(s).strip() != "" and str(t).strip() != ""

    ds["train"] = ds["train"].filter(_ok)
    sp = ds["train"].train_test_split(test_size=VALID_SPLIT, seed=RANDOM_SEED)
    return DatasetDict(train=sp["train"], validation=sp["test"])

def chunked(xs: List[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def main():
    assert torch.cuda.is_available(), "CUDA tidak tersedia."
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Precision
    if INFER_PRECISION == "bf16":
        dtype = torch.bfloat16; autocast_dtype = torch.bfloat16
    elif INFER_PRECISION == "fp16":
        dtype = torch.float16;  autocast_dtype = torch.float16
    else:
        dtype = torch.float32;  autocast_dtype = None

    # Tokenizer & base
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    tok.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_DIR, use_safetensors=True, torch_dtype=dtype
    ).to(DEVICE).eval()

    if base.config.decoder_start_token_id is None:
        base.config.decoder_start_token_id = tok.pad_token_id
    if base.config.pad_token_id is None:
        base.config.pad_token_id = tok.pad_token_id
    base.config.forced_bos_token_id = None
    base.config.forced_eos_token_id = None

    model = PeftModel.from_pretrained(base, ADAPTER_DIR, local_files_only=True).to(DEVICE).eval()
    torch.set_grad_enabled(False)

    # Data (validation only)
    ds = load_val_dataset(DATA_PATH)
    val = ds["validation"]
    srcs = [str(x) for x in val[SRC_COL]]
    refs = [[str(y).strip()] for y in val[TGT_COL]]

    # Generate predictions for validation
    preds = []
    for batch in tqdm(list(chunked(srcs, BATCH_SIZE))):
        prompts = [make_prompt(s) for s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SOURCE_LEN).to(DEVICE)

        ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.no_grad()
        with ctx:
            gen_ids = model.generate(
                **enc,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                min_new_tokens=MIN_NEW_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS,
                early_stopping=EARLY_STOPPING
            )

        out = tok.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend([o.strip() for o in out])

        del enc, gen_ids
        torch.cuda.empty_cache()

    # Evaluate BLEU/chrF
    bleu = evaluate.load("sacrebleu").compute(predictions=preds, references=refs)["score"]
    chrf = evaluate.load("chrf").compute(predictions=preds, references=refs)["score"]
    metrics = {"BLEU": bleu, "chrF": chrf, "N": len(preds)}

    # Save predictions + metrics
    out_dir = os.path.join(ADAPTER_DIR, OUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, OUT_FILE)
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in preds: f.write(p + "\n")

    out_json = os.path.join(out_dir, "valid_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[OK] Saved predictions ->", out_txt)
    print("[OK] Saved metrics     ->", out_json)
    print("[RESULT]", metrics)

if __name__ == "__main__":
    main()
