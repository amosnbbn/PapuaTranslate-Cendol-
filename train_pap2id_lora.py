# train_pap2id_lora.py
import os
import numpy as np
from typing import Optional
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
import evaluate

# =======================
# ======  CONFIG  =======
# =======================
BASE_DIR   = "C:\ML\models\cendol-mt5-base-inst"          # folder base model lokal (harus ada model.safetensors)
OUTPUT_DIR = "cendol_mt5base_pap2id_lora_np_CONT-20251019T095051Z-1-001"          # folder hasil adapter (akan berisi checkpoint-*)
TRAIN_PATH = "data\DATASET_CLEAN_NOEN.csv"  # CSV: kolom id,source,target
VALID_PATH = None  # atau isi path valid CSV jika ada

# Nama kolom pada file kamu:
SRC_COL = "source"
TGT_COL = "target"

# Format instruksi (aktif)
USE_INSTRUCTION_FORMAT = True
INSTRUCTION = "Terjemahkan ke bahasa Indonesia baku."

# Hyperparams
MAX_SOURCE_LEN   = 256
MAX_TARGET_LEN   = 256
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS       = 3
LEARNING_RATE    = 2e-4
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.03

# LoRA config (untuk mT5/T5)
LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05
# Substring nama modul di T5/MT5 yang umum untuk LoRA:
TARGET_MODULES = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]

# Mixed precision
FP16 = torch.cuda.is_available()
BF16 = False

# Save/Logging
SAVE_EVERY_STEPS = 1000
EVAL_EVERY_STEPS = 1000
LOGGING_STEPS    = 50

# =======================

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")  # teks saja, tanpa vision


def detect_file_format(path: str) -> str:
    pl = path.lower()
    if pl.endswith(".tsv") or pl.endswith(".tab"):
        return "tsv"
    if pl.endswith(".csv"):
        return "csv"
    if pl.endswith(".jsonl") or pl.endswith(".json"):
        return "json"
    raise ValueError(f"Format file tidak didukung: {path}")


def load_local_dataset(train_path: str, valid_path: Optional[str]) -> DatasetDict:
    fmt = detect_file_format(train_path)
    files = {"train": train_path}
    if valid_path and os.path.exists(valid_path):
        files["validation"] = valid_path

    if fmt == "csv":
        ds = load_dataset("csv", data_files=files)
    elif fmt == "tsv":
        ds = load_dataset("csv", data_files=files, delimiter="\t")
    else:
        ds = load_dataset("json", data_files=files)

    if "validation" not in ds:
        sp = ds["train"].train_test_split(test_size=0.05, seed=42)
        ds = DatasetDict(train=sp["train"], validation=sp["test"])
    return ds


def make_prompt(src: str) -> str:
    if USE_INSTRUCTION_FORMAT:
        return f"Instruksi: {INSTRUCTION}\nTeks: {src}\nKeluaran:"
    return src


def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    print(f"[CHECK] trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.4f}% trainable)")
    return trainable, total


def main():
    assert os.path.isdir(BASE_DIR), f"BASE_DIR not found: {BASE_DIR}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] Load tokenizer from", BASE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)

    print("[INFO] Load base model from", BASE_DIR)
    dtype = torch.float16 if FP16 else torch.float32
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_DIR,
        use_safetensors=True,
        torch_dtype=dtype,          # transformers==4.45.2 → gunakan torch_dtype
        low_cpu_mem_usage=True,
    )

    # --- penting untuk training T5/MT5 ---
    base.config.use_cache = False                    # matikan cache saat training
    base.enable_input_require_grads()                # agar gradient checkpointing tidak warning
    base.gradient_checkpointing_enable()             # sesuai args di bawah

    print("[INFO] Prepare LoRA & make it trainable")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,  # substring match
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(base, lora_cfg)

    # Cek betul2 ada parameter trainable:
    trainable, total = print_trainable_parameters(model)
    assert trainable > 0, (
        "[ERROR] Tidak ada parameter trainable. "
        "Kemungkinan TARGET_MODULES tidak cocok dengan arsitektur model. "
        "Coba ganti ke ['q','k','v','o'] dulu, lalu tambah 'wi_0','wi_1','wo' jika sudah jalan."
    )

    print("[INFO] Load dataset from local files")
    ds = load_local_dataset(TRAIN_PATH, VALID_PATH)

    # Debug kolom
    print("[DEBUG] columns:", ds["train"].column_names)
    if SRC_COL not in ds["train"].column_names or TGT_COL not in ds["train"].column_names:
        raise KeyError(
            f"Kolom '{SRC_COL}' / '{TGT_COL}' tidak ditemukan. Kolom tersedia: {ds['train'].column_names}"
        )
    print("[DEBUG] sample row:", ds["train"][0])

    # ===== Bersihkan data: buang baris kosong/None & cast ke string =====
    def _valid_example(ex):
        s = ex.get(SRC_COL, None)
        t = ex.get(TGT_COL, None)
        if s is None or t is None:
            return False
        s = str(s).strip()
        t = str(t).strip()
        return (len(s) > 0) and (len(t) > 0)

    ds = ds.filter(_valid_example)

    # Preprocessing yang robust
    def preprocess(batch):
        srcs = ["" if x is None else str(x) for x in batch[SRC_COL]]
        tgts = ["" if y is None else str(y) for y in batch[TGT_COL]]
        prompts = [make_prompt(s) for s in srcs]

        model_inputs = tokenizer(
            prompts,
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=tgts,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[INFO] Tokenize dataset")
    ds_proc = ds.map(preprocess, batched=True)

    # Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    # Metrics
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds_str = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # labels bisa list/ndarray → ganti -100 jadi pad_token_id agar bisa decode
        labels_arr = np.array(labels, dtype=object)
        labels_arr = np.where(labels_arr != -100, labels_arr, tokenizer.pad_token_id)
        label_str = tokenizer.batch_decode(labels_arr, skip_special_tokens=True)

        preds_str, label_wrapped = postprocess_text(preds_str, label_str)
        bleu_res = bleu.compute(predictions=preds_str, references=label_wrapped)
        chrf_res = chrf.compute(predictions=preds_str, references=label_wrapped)
        return {"bleu": bleu_res["score"], "chrf": chrf_res["score"]}

    # Training args
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_EVERY_STEPS,
        save_steps=SAVE_EVERY_STEPS,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=True,   # dipasangkan dengan enable_input_require_grads + use_cache=False
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Train...")
    has_ckpt = any(e.is_dir() and e.name.startswith("checkpoint-") for e in os.scandir(OUTPUT_DIR))
    trainer.train(resume_from_checkpoint=has_ckpt)

    print("[INFO] Save adapter (PEFT)")
    model.save_pretrained(OUTPUT_DIR)   # adapter_model.safetensors + adapter_config.json
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("[OK] Done. Adapter saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
