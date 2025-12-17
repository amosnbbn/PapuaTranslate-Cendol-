# try_infer_debug.py
import os, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ==== GANTI KE PUNYAMU ====
BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = "C:/cendol-mt5/outputs/pap2id_phase7_e10_wandb/checkpoint-560"   # folder adapter hasil training

assert os.path.isdir(ADAPTER_DIR), f"Adapter folder not found: {ADAPTER_DIR}"
assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json")), "adapter_config.json tidak ada"
assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors")), "adapter_model.safetensors tidak ada"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok   = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
base  = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR).to(device)

# === load adapter ===
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
model.eval()
for p in model.parameters(): p.requires_grad = False

# Info adapter loaded
try:
    model.print_trainable_parameters()
except Exception as e:
    print("[WARN] print_trainable_parameters failed:", repr(e))
print("[INFO] PEFT configs:", getattr(model, "peft_config", None))

def make_prompt(instr: str, src: str) -> str:
    # HINDARI 'Keluaran:' agar tidak bocor
    return f"Instruksi: {instr}\nTeks: {src}\nJawaban:"

def clean(text: str) -> str:
    text = re.sub(r"\b(Keluaran:?)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

instr = "Terjemahkan ke bahasa Indonesia baku."
sources = [
    "Ade ko cantik, pipi congka, hitam manis",
    "Andai mo bilang tapi skarang su berbeda",
    "Stengah mati sa su jaga buat koi",
    "Trada makanan di meja",
]

gen_kwargs = dict(
    num_beams=6,
    length_penalty=1.07,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=96,
    do_sample=False,
)

for s in sources:
    prompt = make_prompt(instr, s)
    print("\n==== PROMPT ====\n", prompt)
    enc = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    pred = clean(tok.decode(out[0], skip_special_tokens=True))
    print("SRC:", s)
    print("MT :", pred)
    print("-"*60)
