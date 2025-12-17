# debug_compare_base_adapter_id.py
# ------------------------------------------------------------
# Bandingkan keluaran BASE vs BASE+ADAPTER (PEFT) vs MERGED.
# - Pakai prompt yang sama seperti saat training: "Keluaran:"
# - Set token khusus mT5 (decoder_start_token_id = pad_token_id)
# - Hitung delta logit langkah-pertama (indikasi adapter berpengaruh)
# - Coba beams=4 (beam search) dan beams=1 (greedy)
# ------------------------------------------------------------

import os, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ====== GANTI KE PUNYAMU (PAKAI RAW STRING r"...") ======
BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168"
# Contoh adapter lain yang dulu bagus (uji juga ini jika perlu):
# ADAPTER_DIR = r"C:\cendol-mt5\outputs\pap2id_phase7_e10_wandb\checkpoint-560"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Util ----------
def fix_special_tokens(model, tok):
    """Pastikan ID token khusus mT5 benar, dan nonaktifkan forced BOS/EOS."""
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tok.pad_token_id  # T5/mT5
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tok.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.forced_bos_token_id = None
    model.config.forced_eos_token_id = None
    model.config.use_cache = False

def make_prompt(src: str) -> str:
    # Harus konsisten seperti saat training
    return f"Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks: {src}\nKeluaran:"

def bersih(teks: str) -> str:
    teks = teks.replace("Keluaran:", "")
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks

def generate(model, tok, teks_list, beams=4):
    """Generate terjemahan untuk daftar kalimat sumber."""
    prompts = [make_prompt(t) for t in teks_list]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            num_beams=beams,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            min_new_tokens=3,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return [bersih(x) for x in tok.batch_decode(out, skip_special_tokens=True)]

def first_step_logit_delta(base_model, peft_model, tok, teks):
    """
    Bandingkan logit langkah-pertama decoder antara BASE dan PEFT.
    Jika delta ~0 untuk banyak input, besar kemungkinan adapter tidak berpengaruh
    (base mismatch / ckpt salah / tokenizer atau prompt tidak cocok).
    """
    inp = tok(
        make_prompt(teks),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    # Untuk T5/mT5: decoder dimulai dengan token pad sebagai BOS
    decoder_start = torch.tensor([[tok.pad_token_id]], device=device)

    with torch.inference_mode():
        base_out = base_model(
            **inp,
            decoder_input_ids=decoder_start,
            use_cache=False,
        )
        peft_out = peft_model(
            **inp,
            decoder_input_ids=decoder_start,
            use_cache=False,
        )

    b = base_out.logits[0, 0].float().cpu()
    p = peft_out.logits[0, 0].float().cpu()
    return torch.nn.functional.l1_loss(b, p).item()

# ---------- Main ----------
def main():
    print("[INFO] Cek keberadaan adapter…")
    assert os.path.isdir(ADAPTER_DIR), f"Folder adapter tidak ditemukan: {ADAPTER_DIR}"
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json")), "adapter_config.json tidak ada"
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors")), "adapter_model.safetensors tidak ada"

    print("[INFO] Load tokenizer & base…")
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    tok.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR).to(device)
    fix_special_tokens(base, tok)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    print("[INFO] Tempel adapter PEFT ke base…")
    peft = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device)
    fix_special_tokens(peft, tok)
    peft.eval()
    for p in peft.parameters():
        p.requires_grad = False

    print("[PEFT] attached:", hasattr(peft, "peft_config"))
    try:
        peft.print_trainable_parameters()
    except Exception:
        pass

    # Informasi kecocokan base
    try:
        print("[CFG] base runtime (atribut pada objek):", getattr(peft, "base_model_name_or_path", None))
        print("[CFG] base pada peft_config             :", peft.peft_config["default"].base_model_name_or_path)
    except Exception as e:
        print("[WARN] Tidak bisa membaca peft_config:", repr(e))

    # Daftar uji (bisa kamu ganti)
    tests = [
        "ko su deng dia kah",
        "sa deng dia tadi",
        "sa kenapa ee",
        "ko lagi apa ya",
    ]

    print("\n=== A) BASE ONLY (beams=4) ===")
    out_base = generate(base, tok, tests, beams=4)
    for s, o in zip(tests, out_base):
        print(f"SRC: {s}\nBASE: {o}\n" + "-"*60)

    print("\n=== B) BASE+ADAPTER (PEFT) (beams=4) ===")
    out_peft = generate(peft, tok, tests, beams=4)
    for s, o in zip(tests, out_peft):
        print(f"SRC: {s}\nPEFT: {o}\n" + "-"*60)

    print("\n=== C) Delta logit langkah-pertama (indikasi adapter aktif) ===")
    for s in tests:
        try:
            delta = first_step_logit_delta(base, peft, tok, s)
        except Exception as e:
            delta = None
            print(f"[ERR] delta untuk {s!r} gagal:", repr(e))
        print(f"{s!r}: delta={delta}")

    print("\n[INFO] merge_and_unload (gabung LoRA ke base untuk inferensi)…")
    merged = peft.merge_and_unload()
    fix_special_tokens(merged, tok)
    merged.eval()
    for p in merged.parameters():
        p.requires_grad = False

    print("\n=== D) MERGED (beams=4) ===")
    out_merged = generate(merged, tok, tests, beams=4)
    for s, o in zip(tests, out_merged):
        print(f"SRC: {s}\nMERG: {o}\n" + "-"*60)

    print("\n=== E) PEFT GREEDY (beams=1) ===")
    out_peft_greedy = generate(peft, tok, tests, beams=1)
    for s, o in zip(tests, out_peft_greedy):
        print(f"SRC: {s}\nPEFT_G: {o}\n" + "-"*60)

    print("\n[SELESAI] Jika BASE≈PEFT≈MERG dan delta≈0: kuat indikasi adapter tidak berpengaruh.")
    print("Cek: BASE_DIR sama persis dengan base saat training adapter itu, tokenizer sama, dan prompt 'Keluaran:'.")

if __name__ == "__main__":
    main()
