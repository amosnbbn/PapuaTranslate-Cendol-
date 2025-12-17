# try_infer_simple_safe.py
import os, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok   = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
base  = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR).to(device)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
model.eval()
for p in model.parameters(): p.requires_grad = False

def make_prompt(instr: str, src: str) -> str:
    # SAMA seperti training: "Keluaran:"
    return f"Instruksi: {instr}\nTeks: {src}\nKeluaran:"

def clean(t: str) -> str:
    t = t.replace("Keluaran:", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# (opsional) pre-normalisasi input supaya slang lebih aman
PRE_NORM = {
    r"\bpipi congka\b": "lesung pipi",
    r"\bsa\b": "saya",
    r"\bko\b": "kamu",
    r"\bmo\b": "mau",
    r"\bsu\b": "sudah",
    r"\bkoi\b": "kamu",
    r"\btrada\b": "tidak ada",
}
def prenorm(s: str, enable=True) -> str:
    if not enable: return s
    for pat,rep in PRE_NORM.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

# KALAU versi transformers-mu lama, JANGAN pakai force_words_ids/constraints
BAD = ["ayam", "Peralatan makan", "meletakkan", "Jawaban", "Jawaban:"]
bad_words_ids = [tok(w, add_special_tokens=False).input_ids for w in BAD]
bad_words_ids = [ids for ids in bad_words_ids if isinstance(ids, list) and len(ids) > 0]

gen_args = dict(
    num_beams=5,
    length_penalty=1.07,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=96,
    do_sample=False,
)
if bad_words_ids:
    gen_args["bad_words_ids"] = bad_words_ids

instr = "Terjemahkan ke bahasa Indonesia baku."
samples = [
    "Ade ko cantik, pipi congka, hitam manis",
    "Andai mo bilang tapi skarang su berbeda",
    "Stengah mati sa su jaga buat koi",
    "Trada makanan di meja",
]

for s in samples:
    s_in = prenorm(s, enable=True)   # set False kalau mau “murni”
    prompt = make_prompt(instr, s_in)
    assert "Keluaran:" in prompt and "Jawaban" not in prompt
    enc = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.inference_mode():
        out = model.generate(**enc, **gen_args)
    pred = clean(tok.decode(out[0], skip_special_tokens=True))
    print("SRC:", s)
    print("MT :", pred)
    print("-"*60)
