# contoh inferensi bersih
from transformers import AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("C:\ML\models\cendol-mt5-base-inst", use_fast=True)
prompt = "Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks:  ko su makan…\nKeluaran:"
enc = tok([prompt], return_tensors="pt").to("cuda")

# larang token kata2 template (opsional)
bad_words = ["Instruksi:", "Teks:", "Keluaran:"]
bad_ids = [tok.encode(w, add_special_tokens=False) for w in bad_words]

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM
base = AutoModelForSeq2SeqLM.from_pretrained("C:\ML\models\cendol-mt5-base-inst").to("cuda")
model = PeftModel.from_pretrained(base, "C:\cendol-mt5\outputs\pap2id_split_phase2b").to("cuda").eval()

with torch.inference_mode():
    gen = model.generate(
        **enc,
        num_beams=4,
        length_penalty=1.05,
        no_repeat_ngram_size=3,
        min_new_tokens=8,
        max_new_tokens=128,
        bad_words_ids=bad_ids  # cegah template bocor
    )
print(tok.batch_decode(gen, skip_special_tokens=True))
