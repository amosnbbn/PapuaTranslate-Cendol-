# train_pap2id_lora_fp32_autotarget_debug.py
import os, gc
import numpy as np
from typing import Optional, List
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model
import evaluate

# =======================
# ======  CONFIG  =======
# =======================
BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"          # base model lokal
OUTPUT_DIR  = "C:\cendol-mt5\outputs\pap2id_lora_fp32"     # folder output baru (boleh kosong)

TRAIN_PATH  = "data\DATASET_CLEAN_NOEN.csv"  # CSV: id,source,target
VALID_PATH  = None  # kalau None → auto-split 8%

SRC_COL = "source"
TGT_COL = "target"

USE_INSTRUCTION_FORMAT = True
INSTRUCTION = "Terjemahkan ke bahasa Indonesia baku."

MAX_SOURCE_LEN   = 256
MAX_TARGET_LEN   = 256
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS       = 3  # mulai pendek dulu

LEARNING_RATE    = 3e-5
WARMUP_STEPS     = 500
LABEL_SMOOTHING  = 0.0
MAX_GRAD_NORM    = 0.3
WEIGHT_DECAY     = 0.01

LORA_R        = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.05

# Kandidat target_modules untuk T5/mT5 di berbagai versi
CANDIDATES: List[List[str]] = [
    ["q", "v"],
    ["q", "k", "v", "o"],
    ["q", "k", "v", "o", "wi", "wo"],
    ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
]

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# ===== Helpers =====
def detect_file_format(path: str) -> str:
    p = path.lower()
    if p.endswith((".tsv", ".tab")): return "tsv"
    if p.endswith(".csv"): return "csv"
    if p.endswith((".jsonl", ".json")): return "json"
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
        sp = ds["train"].train_test_split(test_size=0.08, seed=42)
        ds = DatasetDict(train=sp["train"], validation=sp["test"])
    return ds

def make_prompt(src: str) -> str:
    if USE_INSTRUCTION_FORMAT:
        return f"Instruksi: {INSTRUCTION}\nTeks: {src}\nKeluaran:"
    return src

def build_base_and_tokenizer(device="cuda"):
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_DIR,
        use_safetensors=True,
        torch_dtype=torch.float32,    # <- FP32 untuk debug aman
        low_cpu_mem_usage=True
    ).to(device)
    base.config.use_cache = False
    # untuk T5/mT5: pastikan decoder_start_token_id aman
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = tok.pad_token_id
    return tok, base

def attach_lora(base, target_modules: List[str], device="cuda"):
    lcfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(base, lcfg).to(device)
    model.train()
    return model

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    assert torch.cuda.is_available(), "CUDA tidak terdeteksi. Pastikan torch CUDA & driver NVIDIA sudah benar."
    device = "cuda"

    print("[INFO] Load dataset")
    ds = load_local_dataset(TRAIN_PATH, VALID_PATH)
    print("[DEBUG] columns:", ds["train"].column_names)
    if SRC_COL not in ds["train"].column_names or TGT_COL not in ds["train"].column_names:
        raise KeyError(f"Kolom '{SRC_COL}' / '{TGT_COL}' tidak ada. Kolom: {ds['train'].column_names}")
    print("[DEBUG] sample row:", ds["train"][0])

    # filter kosong
    def _valid_example(ex):
        s, t = ex.get(SRC_COL), ex.get(TGT_COL)
        return (s is not None) and (t is not None) and (str(s).strip() != "") and (str(t).strip() != "")
    ds = ds.filter(_valid_example)

    # tokenizer sementara untuk preprocess
    tok_tmp, _ = build_base_and_tokenizer(device)
    def preprocess(batch):
        srcs = [str(x) for x in batch[SRC_COL]]
        tgts = [str(y) for y in batch[TGT_COL]]
        prompts = [make_prompt(s) for s in srcs]
        enc = tok_tmp(prompts, max_length=MAX_SOURCE_LEN, truncation=True, padding=False)
        dec = tok_tmp(text_target=tgts, max_length=MAX_TARGET_LEN, truncation=True, padding=False)
        enc["labels"] = dec["input_ids"]
        return enc

    # IMPORTANT: buang kolom string agar collator tidak memprosesnya
    cols_to_keep = {"input_ids", "attention_mask", "labels"}
    cols_to_remove = [c for c in ds["train"].column_names if c not in cols_to_keep]
    ds_proc = ds.map(preprocess, batched=True, remove_columns=cols_to_remove)

    print("[DEBUG] after map, columns(train):", ds_proc["train"].column_names)
    ex = ds_proc["train"][0]
    neg100 = sum(1 for t in ex["labels"] if t == -100)
    print("[DEBUG] example labels len:", len(ex["labels"]))
    print("[DEBUG] example labels neg100 frac:", neg100 / max(1, len(ex["labels"])))

    # === Auto-detect target_modules (SANITY di FP32) ===
    chosen = None
    chosen_stats = None
    for cand in CANDIDATES:
        print(f"[PROBE] Trying target_modules={cand} ...")
        tok, base = build_base_and_tokenizer(device)
        model = attach_lora(base, cand, device)

        # dataloader 1 batch (FP32, tanpa autocast)
        collator = DataCollatorForSeq2Seq(
            tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100
        )
        from torch.utils.data import DataLoader
        dl = DataLoader(ds_proc["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
        batch = next(iter(dl))
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss_value = float(loss.detach().cpu())
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norm_value = float(total_norm.detach().cpu())

        # hitung berapa lora param yang grad-nya non-zero
        nonzero = 0
        sample_names = []
        for n, p in model.named_parameters():
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad:
                if p.grad is not None and p.grad.data.abs().sum().item() > 0:
                    nonzero += 1
                    if len(sample_names) < 5:
                        sample_names.append(n)

        print(f"[SANITY] loss={loss_value:.4f} grad_norm={grad_norm_value:.6f} lora_nonzero={nonzero}")
        if sample_names:
            for s in sample_names:
                print("   non-zero grad:", s)

        if nonzero > 0 and grad_norm_value > 0.0 and np.isfinite(loss_value):
            chosen = cand
            chosen_stats = (loss_value, grad_norm_value, nonzero)
            del model, base, tok
            clear_gpu()
            break

        del model, base, tok
        clear_gpu()

    assert chosen is not None, (
        "Gagal menemukan target_modules dengan grad LoRA > 0 (bahkan di FP32).\n"
        "- Pastikan data benar (kolom source/target berisi string wajar).\n"
        "- Coba kecilkan MAX_SOURCE_LEN/MAX_TARGET_LEN ke 192.\n"
        "- Jika masih gagal, kirim log [DEBUG]/[SANITY] di atas."
    )
    print(f"[CHOSEN] target_modules={chosen}  (sanity loss={chosen_stats[0]:.4f}, grad_norm={chosen_stats[1]:.6f}, lora_nonzero={chosen_stats[2]})")

    # === Build final model (FP32 training agar pasti jalan) ===
    tok, base = build_base_and_tokenizer(device)
    model = attach_lora(base, chosen, device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100
    )
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [[l.strip()] for l in labels]
        return preds, labels
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        preds_str = tok.batch_decode(preds, skip_special_tokens=True)
        labels_arr = np.array(labels, dtype=object)
        labels_arr = np.where(labels_arr != -100, labels_arr, tok.pad_token_id)
        label_str = tok.batch_decode(labels_arr, skip_special_tokens=True)
        preds_str, label_wrapped = postprocess_text(preds_str, label_str)
        return {
            "bleu": bleu.compute(predictions=preds_str, references=label_wrapped)["score"],
            "chrf": chrf.compute(predictions=preds_str, references=label_wrapped)["score"],
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    label_smoothing_factor=LABEL_SMOOTHING,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_first_step=True,
    logging_steps=50,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,
    fp16=False,                  # FP32 training (debug)
    bf16=False,
    gradient_checkpointing=False,
    max_grad_norm=MAX_GRAD_NORM,
    optim="adamw_torch",
    weight_decay=WEIGHT_DECAY,
    dataloader_pin_memory=False, # <-- ubah ke False
    dataloader_num_workers=0,    # <-- ubah ke 0 (single-process dataloader)
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    seed=42,
    remove_unused_columns=False,
)


    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Train (FP32, fresh, auto-target)...")
    trainer.train(resume_from_checkpoint=False)

    print("[INFO] Save adapter (PEFT)")
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print("[OK] Done. Adapter saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
