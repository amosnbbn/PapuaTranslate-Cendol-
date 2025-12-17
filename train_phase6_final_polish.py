# train_phase6_final_polish.py  (fixed: LoRA trainable + safe metrics)
import os, numpy as np, torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import PeftModel
import evaluate

# ==============================
# CONFIG
# ==============================
BASE       = "C:/ML/models/cendol-mt5-base-inst"
ADAPTER    = "C:/cendol-mt5/outputs/pap2id_clean_phase5_booster/checkpoint-96"
DATA_CLEAN = "C:/cendol-mt5/data_splits_clean/train.csv"
DATA_HARD  = "C:/cendol-mt5/hard_train.csv"            # hasil auto hard-mining
DATA_VALID = "C:/cendol-mt5/data_splits_clean/valid.csv"
OUTPUT_DIR = "C:/cendol-mt5/outputs/pap2id_phase6_final_polish"

EPOCHS      = 3
LR          = 3e-6
BSZ         = 4          # naikkan ke 8 kalau VRAM muat
GRAD_ACCUM  = 8
MAX_SRC_LEN = 256
MAX_TGT_LEN = 128
INSTRUCTION = "Terjemahkan ke bahasa Indonesia baku."

# ==============================
# HELPERS
# ==============================
def load_csv_any(path):
    ds = load_dataset("csv", data_files=path)["train"]
    cols = [c.lower() for c in ds.column_names]
    # seragamkan ke source/target
    if "input" in cols and "output" in cols:
        ds = ds.rename_columns({"input":"source", "output":"target"})
    return ds

def make_prompt(src: str) -> str:
    return f"Instruksi: {INSTRUCTION}\nTeks: {src}\nKeluaran:"

def sanitize_ids(nested_ids, pad_id: int):
    """Ganti <0 (mis. -100) ke pad_id, paksa int → aman untuk decode."""
    arr = nested_ids.tolist() if hasattr(nested_ids, "tolist") else nested_ids
    fixed = []
    for row in arr:
        clean = []
        for x in row:
            try:
                xi = int(x)
            except Exception:
                xi = pad_id
            clean.append(xi if xi >= 0 else pad_id)
        fixed.append(clean)
    return fixed

# ==============================
# LOAD DATA
# ==============================
train_clean = load_csv_any(DATA_CLEAN)
train_hard  = load_csv_any(DATA_HARD)
train_ds    = concatenate_datasets([train_clean, train_hard]).shuffle(seed=42)
valid_ds    = load_csv_any(DATA_VALID)

print(f"[INFO] Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")

# filter kosong
def _ok(ex):
    return bool(str(ex["source"]).strip()) and bool(str(ex["target"]).strip())
train_ds = train_ds.filter(_ok)
valid_ds = valid_ds.filter(_ok)

# ==============================
# TOKENIZER & MODEL (+LoRA trainable)
# ==============================
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.padding_side = "right"

base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
)
base.config.use_cache = False
if getattr(base.config, "decoder_start_token_id", None) is None:
    base.config.decoder_start_token_id = tok.pad_token_id
if getattr(base.config, "eos_token_id", None) is None:
    base.config.eos_token_id = tok.eos_token_id
if getattr(base.config, "pad_token_id", None) is None:
    base.config.pad_token_id = tok.pad_token_id

# >>> IMPORTANT: is_trainable=True <<<
model = PeftModel.from_pretrained(
    base, ADAPTER, is_trainable=True, local_files_only=True
)
model.train()  # ensure training mode (LoRA layers get grads)

try:
    model.print_trainable_parameters()  # sanity: must NOT be 0%
except Exception:
    pass

# ==============================
# PREPROCESS
# ==============================
def preprocess(batch):
    prompts = [make_prompt(s) for s in batch["source"]]
    enc = tok(prompts, max_length=MAX_SRC_LEN, truncation=True, padding=False)
    dec = tok(text_target=batch["target"], max_length=MAX_TGT_LEN, truncation=True, padding=False)
    enc["labels"] = dec["input_ids"]
    return enc

keep = {"input_ids", "attention_mask", "labels"}
train_tok = train_ds.map(preprocess, batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep])
valid_tok = valid_ds.map(preprocess, batched=True, remove_columns=[c for c in valid_ds.column_names if c not in keep])

data_collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100)

# ==============================
# METRICS (chrF)
# ==============================
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # dengan predict_with_generate=True → preds = token IDs hasil generate
    preds_clean  = sanitize_ids(preds,  tok.pad_token_id)
    labels_clean = sanitize_ids(labels, tok.pad_token_id)

    preds_str  = [s.strip() for s in tok.batch_decode(preds_clean,  skip_special_tokens=True)]
    labels_str = [s.strip() for s in tok.batch_decode(labels_clean, skip_special_tokens=True)]

    # chrf expects list[str], list[str] atau list[list[str]] sebagai refs
    score = chrf.compute(predictions=preds_str, references=labels_str)["score"]
    return {"chrF": score}

# ==============================
# TRAINING ARGS
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BSZ,
    per_device_eval_batch_size=BSZ,
    gradient_accumulation_steps=GRAD_ACCUM,

    warmup_ratio=0.05,
    label_smoothing_factor=0.1,
    weight_decay=0.01,
    max_grad_norm=0.3,

    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_first_step=True,
    logging_steps=50,

    predict_with_generate=True,
    generation_num_beams=6,       # decoding kuat (bisa 6)
    generation_max_length=MAX_TGT_LEN,

    fp16=False,                   # fp32 paling stabil untuk micro FT
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="chrF",
    greater_is_better=True,
    seed=42,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    tokenizer=tok,
    compute_metrics=compute_metrics,
)

# ==============================
# TRAIN
# ==============================
trainer.train()
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"[OK] Training done. Saved to {OUTPUT_DIR}")
