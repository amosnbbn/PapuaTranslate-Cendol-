import os
import torch
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model


# ============================================================
#  HELPERS
# ============================================================

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}")


# ============================================================
#  MAIN TRAIN LOOP
# ============================================================

def main():
    print("[INFO] Load tokenizer & base")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

    model = MT5ForConditionalGeneration.from_pretrained(
        "google/mt5-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # ============================================================
    # LOAD LoRA ADAPTER AWAL
    # ============================================================
    print("[INFO] Load adapter awal (PEFT/LoRA)")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"]
    )
    model = get_peft_model(model, lora_cfg)
    print_trainable_parameters(model)

    # ============================================================
    # LOAD DATASET
    # ============================================================
    print("[INFO] Load datasets (train/valid/test)")
    dataset = load_dataset("json", data_files={
        "train": "data/train.json",
        "valid": "data/valid.json",
        "test": "data/test.json"
    })

    def preprocess(batch):
        src = batch["pap"]
        tgt = batch["id"]

        model_inputs = tokenizer(
            src,
            max_length=128,
            truncation=True
        )

        labels = tokenizer(
            tgt,
            max_length=128,
            truncation=True
        )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs

    dataset = dataset.map(preprocess, batched=True)

    print(f"[INFO] Sizes -> train: {len(dataset['train'])} | valid: {len(dataset['valid'])} | test: {len(dataset['test'])}")

    # ============================================================
    # TRAINING SETUP (FIXED VERSION)
    # ============================================================
    print("[INFO] Train Phase-5 (small LR + cosine + early stopping)")

    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints_run2_fp16",
        overwrite_output_dir=True,

        # MATCHED FIX
        evaluation_strategy="epoch",
        save_strategy="epoch",

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        learning_rate=2e-4,
        num_train_epochs=20,
        weight_decay=0.01,

        logging_dir="logs_run2",
        logging_steps=10,

        predict_with_generate=True,

        # FP16 TRAINING
        fp16=True,

        load_best_model_at_end=True,
        metric_for_best_model="loss",

        report_to="none",
        save_total_limit=3,

        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ============================================================
    # TRAINER
    # ============================================================
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    # ============================================================
    # TRAIN!
    # ============================================================
    trainer.train()

    # ============================================================
    # SAVE MODEL
    # ============================================================
    print("[INFO] Saving final model...")
    trainer.save_model("final_model_run2_fp16")


# ============================================================
#  RUN
# ============================================================
if __name__ == "__main__":
    main()
