import os, argparse, random, numpy as np, torch
from typing import List
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
)
from peft import PeftModel
import evaluate

# ======== utils ========
def set_seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_prompt(instr: str, src: str) -> str:
    return f"Instruksi: {instr}\nTeks: {src}\nKeluaran:"

def sanitize_for_decode(batch_ids: List[List[int]], pad_id: int) -> List[List[int]]:
    out = []
    for row in batch_ids:
        out.append([tid if isinstance(tid, int) and tid >= 0 else pad_id for tid in row])
    return out

# ======== W&B callback untuk bikin tabel per-epoch ========
class WandbEpochTable(TrainerCallback):
    def __init__(self):
        self.rows = []  # (epoch, train_loss, eval_loss, bleu, chrf)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # dipanggil setelah evaluate(); simpan angka yang ada
        epoch = round(state.epoch or 0, 2)
        train_loss = getattr(state, "log_history", [{}])[-1].get("loss", None)
        eval_loss  = (metrics or {}).get("eval_loss", None)
        bleu       = (metrics or {}).get("bleu", (metrics or {}).get("eval_bleu", None))
        chrf       = (metrics or {}).get("chrf", (metrics or {}).get("eval_chrf", None))
        self.rows.append([epoch, train_loss, eval_loss, bleu, chrf])

    def on_train_end(self, args, state, control, **kwargs):
        try:
            import wandb
            table = wandb.Table(columns=["Epoch","Training Loss","Validation Loss","BLEU","chrF"])
            for r in self.rows:
                table.add_data(*r)
            wandb.log({"epoch_table": table})
        except Exception as e:
            print("[WARN] wandb table log failed:", repr(e))

# ======== main ========
def main():
    ap = argparse.ArgumentParser("Fine-tune mT5 LoRA 10 epoch + W&B logging")
    ap.add_argument("--base_dir", required=True, help="Folder model base HF (bukan GGUF)")
    ap.add_argument("--adapter_init", default="", help="Checkpoint LoRA untuk lanjut (opsional)")
    ap.add_argument("--splits_dir", required=True, help="Folder berisi train.json, valid.json, test.json")
    ap.add_argument("--out_dir", required=True)

    # training
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=0.1)
    ap.add_argument("--src_max_len", type=int, default=256)
    ap.add_argument("--tgt_max_len", type=int, default=256)
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    # W&B
    ap.add_argument("--wandb_project", default="mt5-pap2id")
    ap.add_argument("--wandb_run_name", default="phase7_e10")
    ap.add_argument("--wandb_entity", default=None)  # isi kalau kamu pakai team/org

    args = ap.parse_args()

    # === W&B init (opsional: set WANDB_API_KEY di env) ===
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity: os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    except Exception as e:
        print("[WARN] wandb init failed, continue without:", repr(e))

    # Runtime safety (Windows-friendly)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed_all(args.seed)
    assert torch.cuda.is_available(), "CUDA tidak terdeteksi."
    device = "cuda"

    # === dataset (JSON dengan instruction/input/output) ===
    files = {
        "train": os.path.join(args.splits_dir, "train.json"),
        "validation": os.path.join(args.splits_dir, "valid.json"),
        "test": os.path.join(args.splits_dir, "test.json"),
    }
    ds = load_dataset("json", data_files=files)
    for split in ["train","validation","test"]:
        cols = set(ds[split].column_names)
        for k in ["instruction","input","output"]:
            assert k in cols, f"{split} harus punya kolom instruction/input/output. Ditemukan: {ds[split].column_names}"

    # === tokenizer & base model ===
    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True)
    tok.padding_side = "right"

    if args.bf16:
        dtype = torch.bfloat16; use_fp16=False; use_bf16=True
    elif args.fp16:
        dtype = torch.float16;  use_fp16=True;  use_bf16=False
    else:
        dtype = torch.float32;  use_fp16=False; use_bf16=False

    base = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_dir, use_safetensors=True, low_cpu_mem_usage=True,
        torch_dtype=(dtype if dtype!=torch.float32 else None), local_files_only=True
    ).to(device)
    base.config.use_cache = False
    if getattr(base.config,"decoder_start_token_id",None) is None:
        base.config.decoder_start_token_id = tok.pad_token_id
    if getattr(base.config,"eos_token_id",None) is None:
        base.config.eos_token_id = tok.eos_token_id
    if getattr(base.config,"pad_token_id",None) is None:
        base.config.pad_token_id = tok.pad_token_id
    base.config.forced_bos_token_id = None
    base.config.forced_eos_token_id = None

    # === LoRA adapter (lanjut dari checkpoint) ===
    if args.adapter_init and os.path.isdir(args.adapter_init):
        model = PeftModel.from_pretrained(
            base, args.adapter_init, is_trainable=True, local_files_only=True,
            torch_dtype=(dtype if dtype!=torch.float32 else None)
        ).to(device)
    else:
        model = base
    model.train()
    if hasattr(model,"print_trainable_parameters"):
        try: model.print_trainable_parameters()
        except: pass

    # === preprocess ===
    trunc_src=0; trunc_tgt=0
    def _prep(batch):
        nonlocal trunc_src, trunc_tgt
        instrs = [str(x) for x in batch["instruction"]]
        srcs   = [str(x) for x in batch["input"]]
        tgts   = [str(x) for x in batch["output"]]
        prompts = [make_prompt(i,s) for i,s in zip(instrs,srcs)]
        enc = tok(prompts, max_length=args.src_max_len, truncation=True, padding=False)
        dec = tok(text_target=tgts, max_length=args.tgt_max_len, truncation=True, padding=False)
        trunc_src += sum(len(ids) >= args.src_max_len for ids in enc["input_ids"])
        trunc_tgt += sum(len(ids) >= args.tgt_max_len for ids in dec["input_ids"])
        enc["labels"] = dec["input_ids"]
        return enc

    keep={"input_ids","attention_mask","labels"}
    rm_train=[c for c in ds["train"].column_names if c not in keep]
    rm_val  =[c for c in ds["validation"].column_names if c not in keep]
    rm_test =[c for c in ds["test"].column_names if c not in keep]
    ds_proc = DatasetDict(
        train=ds["train"].map(_prep, batched=True, remove_columns=rm_train),
        validation=ds["validation"].map(_prep, batched=True, remove_columns=rm_val),
        test=ds["test"].map(_prep, batched=True, remove_columns=rm_test),
    )
    print(f"[INFO] Truncated -> src: {trunc_src} | tgt: {trunc_tgt}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=model, padding="longest", label_pad_token_id=-100, return_tensors="pt"
    )

    # === metrics ===
    bleu=evaluate.load("sacrebleu"); chrf=evaluate.load("chrf")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        preds = preds.tolist() if hasattr(preds,"tolist") else preds
        labels = labels.tolist() if hasattr(labels,"tolist") else labels
        preds_clean  = sanitize_for_decode(preds, tok.pad_token_id)
        labels_clean = sanitize_for_decode(labels, tok.pad_token_id)
        preds_str  = [s.strip() for s in tok.batch_decode(preds_clean,  skip_special_tokens=True)]
        labels_str = [s.strip() for s in tok.batch_decode(labels_clean, skip_special_tokens=True)]
        refs = [[l] for l in labels_str]
        try: bleu_s = bleu.compute(predictions=preds_str, references=refs)["score"]
        except: bleu_s = 0.0
        try: chrf_s = chrf.compute(predictions=preds_str, references=refs)["score"]
        except: chrf_s = 0.0
        return {"bleu": bleu_s, "chrf": chrf_s}

    # === training args (10 epoch penuh) ===
    os.makedirs(args.out_dir, exist_ok=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir, overwrite_output_dir=True,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="bleu", greater_is_better=True,
        save_total_limit=3, logging_steps=50,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",

        predict_with_generate=True, generation_max_length=args.tgt_max_len,
        generation_num_beams=args.beam,

        optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        fp16=args.fp16, bf16=args.bf16, gradient_checkpointing=False,

        dataloader_pin_memory=False, dataloader_num_workers=0,
        report_to=["wandb"],  # <— aktifkan W&B
        seed=args.seed, remove_unused_columns=False, save_safetensors=True,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["validation"],
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        callbacks=[WandbEpochTable()],  # tabel per-epoch
    )

    trainer.train()
    print("[INFO] Best checkpoint:", trainer.state.best_model_checkpoint)

    # Final eval (VALID & TEST)
    for split, key in [("validation","FINAL_VALID"), ("test","FINAL_TEST")]:
        metrics = trainer.evaluate(eval_dataset=ds_proc[split], metric_key_prefix=key.lower())
        print(f"[{key}] {metrics}")

    # Save adapter/tokenizer
    print("[INFO] Save adapter/tokenizer")
    try: model.save_pretrained(args.out_dir)
    except Exception as e: print("[WARN] save adapter failed:", repr(e))
    tok.save_pretrained(args.out_dir)
    print("[OK] Done ->", args.out_dir)

if __name__ == "__main__":
    main()
