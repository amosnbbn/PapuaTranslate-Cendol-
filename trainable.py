# check_trainable_standalone.py
import torch
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR  = "C:\ML\models\cendol-mt5-base-inst"                    # base full model
BEST_CKPT = "cendol_mt5base_pap2id_lora_np_CONT-20251019T095051Z-1-001"    # ganti ke ckpt terbaik

device = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_DIR, use_safetensors=True, torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

model = PeftModel.from_pretrained(
    base, BEST_CKPT, is_trainable=True, local_files_only=True,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

model.train()
try:
    model.print_trainable_parameters()
except Exception as e:
    print("[WARN] can't print trainable params:", e)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"[CHECK] trainable params: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")
