# train_phase8_fast_e10.py
import os, json, random, numpy as np, torch
from typing import Dict, Any
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from peft import PeftModel
import evaluate

# ====================== CONFIG ======================
BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_IN  = "C:\cendol-mt5\outputs\pap2id_phase7_e10_wandb\checkpoint-560"
SPLITS_DIR  = "C:\cendol-mt5\data_splits_json\json"
OUT_DIR     = "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10 lanjutan2"

INSTR_FALLBACK = "Terjemahkan ke bahasa Indonesia baku."

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256
BATCH_SIZE     = 4
GRAD_ACCUM     = 8

NUM_EPOCHS     = 10
LR             = 7.5e-10
WARMUP_RATIO   = 0.10
WEIGHT_DECAY   = 0.01
LABEL_SMOOTH   = 0.0
SEED           = 42

DECODE_CFG = dict(
    num_beams=6,
    length_penalty=0.95,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=64,
    do_sample=False
)

# ================== RUNTIME SAFETY ==================
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def set_seed_all(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ================== DATA HELPERS ====================
def load_splits_json(splits_dir: str) -> DatasetDict:
    files = {
        "train": os.path.join(splits_dir, "train.json"),
        "validation": os.path.join(splits_dir, "valid.json"),
        "test": os.path.join(splits_dir, "test.json"),
    }
    for k, p in files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Split file missing: {p}")
    ds = load_dataset("json", data_files=files)
    def _norm(ex):
        inst = ex.get("instruction") or INSTR_FALLBACK
        inp  = ex.get("input", "")
        out  = ex.get("output", "")
        return {"instruction": str(inst), "input": str(inp), "output": str(out)}
    ds = ds.map(_norm)
    def _valid(x): return (x["input"].strip()!="") and (x["output"].strip()!="")
    ds["train"] = ds["train"].filter(_valid)
    ds["validation"] = ds["validation"].filter(_valid)
    ds["test"] = ds["test"].filter(_valid)
    return ds

def make_prompt(inst: str, src: str) -> str:
    
    return f"Instruksi: {inst}\nTeks: {src}\nKeluaran:"

# ================== METRICS (final only) ============
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def eval_with_generation(model, tok, ds_split, device, split_name="valid") -> Dict[str, Any]:
    model.eval()
    preds, refs = [], []
    bs = 8
    for i in range(0, len(ds_split), bs):
        batch = ds_split[i:i+bs]
        prompts = [make_prompt(inst, inp) for inst, inp in zip(batch["instruction"], batch["input"])]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_SOURCE_LEN).to(device)
        with torch.no_grad():
            gen_ids = model.generate(**enc, **DECODE_CFG)
        outs = tok.batch_decode(gen_ids, skip_special_tokens=True)
        preds.extend([o.strip() for o in outs])
        refs.extend([[t.strip()] for t in batch["output"]])
    bleu_s = bleu.compute(predictions=preds, references=refs)["score"]
    chrf_s = chrf.compute(predictions=preds, references=refs)["score"]
    return {f"final_{split_name}_bleu": bleu_s, f"final_{split_name}_chrf": chrf_s}

# ======================= MAIN =======================
def main():
    assert torch.cuda.is_available(), "CUDA tidak terdeteksi."
    device = "cuda"; set_seed_all(SEED)

    print("[INFO] Load splits …")
    ds = load_splits_json(SPLITS_DIR)
    print(f"[INFO] Sizes -> train: {len(ds['train'])} | valid: {len(ds['validation'])} | test: {len(ds['test'])}")

    print("[INFO] Load tokenizer & base …")
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    tok.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_DIR,
        use_safetensors=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True
    ).to(device)
    base.config.use_cache = False
    if getattr(base.config, "decoder_start_token_id", None) is None:
        base.config.decoder_start_token_id = tok.pad_token_id
    if getattr(base.config, "eos_token_id", None) is None:
        base.config.eos_token_id = tok.eos_token_id
    if getattr(base.config, "pad_token_id", None) is None:
        base.config.pad_token_id = tok.pad_token_id
    base.config.forced_bos_token_id = None
    base.config.forced_eos_token_id = None

    print("[INFO] Attach adapter (trainable) …")
    assert os.path.isdir(ADAPTER_IN), f"Adapter folder not found: {ADAPTER_IN}"
    model = PeftModel.from_pretrained(
        base, ADAPTER_IN, is_trainable=True, local_files_only=True, torch_dtype=torch.float16
    ).to(device)
    try: model.print_trainable_parameters()
    except: pass

    trunc_src = 0; trunc_tgt = 0
    def preprocess(batch):
        nonlocal trunc_src, trunc_tgt
        prompts = [make_prompt(inst, src) for inst, src in zip(batch["instruction"], batch["input"])]
        enc = tok(prompts, max_length=MAX_SOURCE_LEN, truncation=True, padding=False)
        dec = tok(text_target=batch["output"], max_length=MAX_TARGET_LEN, truncation=True, padding=False)
        trunc_src += sum(len(x)>=MAX_SOURCE_LEN for x in enc["input_ids"])
        trunc_tgt += sum(len(x)>=MAX_TARGET_LEN for x in dec["input_ids"])
        enc["labels"] = dec["input_ids"]
        return enc

    keep = {"input_ids","attention_mask","labels"}
    ds_proc = DatasetDict(
        train=ds["train"].map(preprocess, batched=True,
                              remove_columns=[c for c in ds["train"].column_names if c not in keep]),
        validation=ds["validation"].map(preprocess, batched=True,
                              remove_columns=[c for c in ds["validation"].column_names if c not in keep]),
        test=ds["test"]  # text form for final eval
    )
    print(f"[INFO] Truncated -> src: {trunc_src} | tgt: {trunc_tgt}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100, return_tensors="pt"
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    print("[INFO] Train FAST (loss-only eval per epoch, 10 epoch) …")
    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,


        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        label_smoothing_factor=LABEL_SMOOTH,
        lr_scheduler_type="cosine",

        
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=200,
        save_total_limit=3,

      
        predict_with_generate=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,

        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        eval_accumulation_steps=8,

        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        data_collator=data_collator,
        tokenizer=tok,      
        compute_metrics=None,
    )

    trainer.train()
    print("[INFO] Best checkpoint:", trainer.state.best_model_checkpoint)

    # =============== FINAL EVAL (generate) ===============
    print("[INFO] Final eval (VALID) with tuned decoding …")
    final_valid = eval_with_generation(model, tok, ds["validation"], device, split_name="valid")
    print("[FINAL_VALID]", final_valid)

    print("[INFO] Final eval (TEST) with tuned decoding …")
    final_test = eval_with_generation(model, tok, ds["test"], device, split_name="test")
    print("[FINAL_TEST]", final_test)

    print("[INFO] Save adapter/tokenizer")
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print("[OK] Done ->", OUT_DIR)

if __name__ == "__main__":
    main()
