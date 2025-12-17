# try_infer_use_best.py
import os, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168"  # <- pakai ini

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_prompt(src: str) -> str:
    # Harus PERSIS seperti saat training (pakai "Keluaran:")
    return f"Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks: {src}\nKeluaran:"

def bersih(t: str) -> str:
    t = t.replace("Keluaran:", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# 1) load tokenizer & base
tok  = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR)

# Fix token penting untuk mT5
if base.config.decoder_start_token_id is None:
    base.config.decoder_start_token_id = tok.pad_token_id  # BOS mT5 = <pad>
if base.config.eos_token_id is None:
    base.config.eos_token_id = tok.eos_token_id
if base.config.pad_token_id is None:
    base.config.pad_token_id = tok.pad_token_id
base.config.forced_bos_token_id = None
base.config.forced_eos_token_id = None
base.config.use_cache = False

base = base.to(device).eval()

# 2) tempel adapter
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device).eval()
for p in model.parameters(): p.requires_grad = False

print("[INFO] Adapter terpasang:", hasattr(model, "peft_config"))

GEN = dict(
    num_beams=4,              # stabil
    no_repeat_ngram_size=3,
    length_penalty=1.0,
    min_new_tokens=3,
    max_new_tokens=64,
    do_sample=False,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)

tests = [
    "Ade ko cantik, pipi congka, hitam manis",
    "Andai mo bilang tapi skarang su berbeda",
    "Stengah mati sa su jaga buat koi",
    "Trada makanan di meja",
]

prompts = [make_prompt(s) for s in tests]
enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

with torch.inference_mode():
    out = model.generate(**enc, **GEN)

preds = [bersih(x) for x in tok.batch_decode(out, skip_special_tokens=True)]
for s, p in zip(tests, preds):
    print("SRC:", s)
    print("MT :", p)
    print("-"*60)
