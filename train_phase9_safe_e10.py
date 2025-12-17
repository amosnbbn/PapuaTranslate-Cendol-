# test_infer_pap2id.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE = "C:/ML/models/cendol-mt5-base-inst"
ADAPTER = "C:/cendol-mt5/outputs/pap2id_phase7_e10_wandb/checkpoint-560"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

examples = [
    "Tabea, mari katong pigi pasar besok…",
    "Ade ko cantik, pipi congka, hitam manis",
    "Stengah mati sa su jaga buat koi",
]

for src in examples:
    print(f"SRC: {src}")
    print(f"MT : {translate(src)}")
    print("-" * 60)
