from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from peft import PeftModel
import torch

BASE_DIR   = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_IN = "C:\cendol-mt5\outputs\pap2id_clean_polish\checkpoint-144"

tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR, torch_dtype=torch.float32).cuda()
model = PeftModel.from_pretrained(base, ADAPTER_IN).cuda().eval()

gen_cfg = GenerationConfig(
    num_beams=6,
    length_penalty=0.95,
    no_repeat_ngram_size=3,
    min_new_tokens=4,
    max_new_tokens=64,
    early_stopping=True,
    # Kalau masih suka keluar “Instruksi/Teks/Keluaran”, aktifkan bad_words optional di bawah:
    # bad_words_ids=[tok.encode(w, add_special_tokens=False) for w in ["Instruksi:", "Teks:", "Keluaran:"]]
)

def translate(texts):
    with torch.inference_mode():
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        ids = model.generate(**enc, generation_config=gen_cfg)
        return tok.batch_decode(ids, skip_special_tokens=True)

print(translate(["Tabea, mari katong pigi pasar besok…"]))
