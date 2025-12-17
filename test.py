# test.py — load base Cendol mT5 + adapter LoRA lokal
import os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# === GANTI path sesuai punyamu ===
BASE_DIR    = r"C:\ML\models\cendol-mt5-base-inst"  # folder base model (berisi model.safetensors)
ADAPTER_DIR = r"C:\cendol-mt5\cendol_mt5base_pap2id_lora_np_CONT-20251019T095051Z-1-001\cendol_mt5base_pap2id_lora_np_CONT"
# Jika ingin pakai checkpoint tertentu (mis. 712), ganti ADAPTER_DIR ke folder checkpoint-712

# --- Guard supaya ketahuan kalau path salah
assert os.path.isdir(ADAPTER_DIR), f"Folder adapter tidak ditemukan: {ADAPTER_DIR}"
assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json")), "adapter_config.json TIDAK ADA di ADAPTER_DIR"
assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors")), "adapter_model.safetensors TIDAK ADA di ADAPTER_DIR"

# --- Device & dtype (kamu saat ini di transformers 4.45.2 → pakai torch_dtype)
use_gpu = torch.cuda.is_available()
device  = "cuda" if use_gpu else "cpu"
dtype   = torch.float16 if use_gpu else torch.float32

# --- Tokenizer ambil dari BASE (jangan dari folder adapter)
tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)

# --- Load base model dari .safetensors
base = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_DIR,
    use_safetensors=True,      # pakai file model.safetensors yang sudah kita buat
    torch_dtype=dtype,         # transformers==4.45.2 → gunakan torch_dtype
    low_cpu_mem_usage=True,
).to(device)

# --- Tempel LoRA dari folder lokal (hindari akses HF)
model = PeftModel.from_pretrained(
    base,
    ADAPTER_DIR,
    local_files_only=True,     # penting: baca lokal saja
).to(device).eval()

def translate_papua_to_id(text, instruction="Terjemahkan ke bahasa Indonesia baku."):
    prompt = f"{instruction}\n\n{text}"
    enc = tok([prompt], return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=128,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            remove_invalid_values=True,
            decoder_start_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    kalimat = "sa"
    print("INPUT :", kalimat)
    print("OUTPUT:", translate_papua_to_id(kalimat))
