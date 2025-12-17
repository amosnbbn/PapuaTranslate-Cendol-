# fix_safetensors.py
import os, torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM

# === GANTI jika lokasi model base beda ===
BASE_DIR = r"C:\ML\models\cendol-mt5-base-inst"
# ========================================

# Hindari import modul vision yang tak perlu
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

# 1) Bangun arsitektur dari config (tanpa bobot)
print("[INFO] build model from config.json")
cfg = AutoConfig.from_pretrained(BASE_DIR)
model = AutoModelForSeq2SeqLM.from_config(cfg)

# 2) Load state_dict dari 'pytorch_model.bin' (langsung via torch.load)
bin_path = os.path.join(BASE_DIR, "pytorch_model.bin")
print("[INFO] load state_dict from:", bin_path)
sd = torch.load(bin_path, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
    sd = sd["state_dict"]

missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[STATE] missing={len(missing)} unexpected={len(unexpected)}")

# 3) Tulis ulang ke safetensors dg metadata framework yang benar
print("[INFO] save_pretrained(..., safe_serialization=True)")
model.save_pretrained(BASE_DIR, safe_serialization=True)
print("[OK] wrote:", os.path.join(BASE_DIR, "model.safetensors"))
