# pap2id_cli.py
# REPL interaktif uji terjemahan dialek Papua -> Indonesia (mT5 + LoRA)
import os, re, sys, torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ================== KONFIG ==================
BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168"
# Jika adapter lain yang lebih bagus:
# ADAPTER_DIR = r"C:\cendol-mt5\outputs\pap2id_phase7_e10_wandb\checkpoint-560"

USE_PRENORM = True  # normalisasi ringan dialek
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== UTIL ==================
def fix_special_tokens(model, tok):
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tok.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tok.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.forced_bos_token_id = None
    model.config.forced_eos_token_id = None
    model.config.use_cache = False

def make_prompt(src: str) -> str:
    # HARUS sama dengan format training: gunakan “Keluaran:”
    return f"Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks: {src}\nKeluaran:"

def strip_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean_after_decode(t: str) -> str:
    return strip_spaces(t.replace("Keluaran:", ""))

# ===== pra-normalisasi ringan =====
LEX = {
    r"\bko\b": "kamu",
    r"\bngana\b": "kamu",
    r"\bsa\b": "saya",
    r"\bkam\b": "kalian",
    r"\btrada\b": "tidak ada",
    r"\btra\b": "tidak",
    r"\bsu\b": "sudah",
    r"\bdeng\b": "dengan",
    r"\bpigi\b": "pergi",
    r"\bskarang\b": "sekarang",
    r"\bstengah\b": "setengah",
    r"\bkoi\b": "kamu",
    r"\bpung\b": "punya",
    r"\bcongka\b": "lesung pipit",
}
def prenorm(s: str) -> str:
    out = s
    for pat, rep in LEX.items():
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return strip_spaces(out)

# ===== blacklist frasa narasi/artefak =====
BAD_PHRASES = [
    "Baik.", "baiklah.", "sudah begitu", "apakah ini benar", "sedang.",
    "ini lagi apa ya",
    "Jawab", "jawab", "Kata", "kata", "berkata", "katanya", "mengatakan", "mengucap", "lalu", "bahwa",
    "keluar", "keluaran", "keluarannya", "keluari",
    "“", "”", "\"", "(", ")", "…", "‘", "’", "：", "；", ";", ":",
    "Ia ", " ia ",
]
def make_bad_words_ids(tok) -> List[List[int]]:
    ids = []
    for phrase in BAD_PHRASES:
        toks = tok(phrase, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks)
    return ids

# ===== ambil kalimat pertama + buang akar "keluar" =====
END_PUNCTS = r"[.!?…]|。|？|！"
def first_sentence(s: str) -> str:
    s = strip_spaces(s)
    s = re.split(r'\b(kata|jawab|berkata|katanya|mengatakan|mengucap|lalu|bahwa)\b', s, flags=re.IGNORECASE)[0]
    s = re.split(r'["“”\(\)]', s)[0]
    m = re.search(END_PUNCTS, s)
    if m:
        s = s[:m.end()]
    words = s.split()
    if len(words) > 20:
        s = " ".join(words[:20])
    s = re.sub(r"\bkeluar\w*\b", "", s, flags=re.IGNORECASE)
    return strip_spaces(s)

def encode(tok, prompts: List[str], max_len=256):
    return tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEVICE)

def decode(tok, out_ids):
    return [clean_after_decode(x) for x in tok.batch_decode(out_ids, skip_special_tokens=True)]

# ===== rules super-sering (jalan duluan) =====
RULES = [
    (r"^sa tra tau\.?$",                           "Saya tidak tahu."),
    (r"^sa tra tau.*",                             "Saya tidak tahu."),
    (r"^ko lagi apa(?: ya)?\??$",                  "Kamu lagi apa?"),
    (r"^ko lagi bikin\.?$",                        "Kamu sedang membuat apa?"),
    (r"^ko su makan\??$",                          "Kamu sudah makan?"),
    (r"^ko su deng dia kah\??$",                   "Apakah kamu sudah bersamanya?"),
    (r"^sa kenapa ee\.?$",                         "Saya kenapa ya?"),
    (r"^sa deng dia tadi\.?$",                     "Saya bersama dia tadi."),
    (r"^kam dimana\??$",                           "Kalian di mana?"),
    (r"^ko dimana\??$",                            "Kamu di mana?"),
    (r"^trada\b.*",                                None),  # akan ditangani fallback rule di bawah
]
def try_rules(raw: str) -> str | None:
    s = raw.strip().lower()
    for pat, out in RULES:
        if re.match(pat, s, flags=re.IGNORECASE):
            if out is None:
                # contoh "trada ..." -> ubah “trada” ke “Tidak ada”
                return re.sub(r"^trada", "Tidak ada", raw, flags=re.IGNORECASE).strip()
            return out
    return None

# ===== generate =====
def generate_once(model, tok, sources: List[str], beams: int, max_new: int,
                  bad_ids=None, rep_penalty=1.12, no_repeat=3, enc_no_repeat=0):
    prompts = [make_prompt(s) for s in sources]
    enc = encode(tok, prompts)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            num_beams=beams,
            do_sample=False,
            no_repeat_ngram_size=no_repeat,
            encoder_no_repeat_ngram_size=enc_no_repeat,
            repetition_penalty=rep_penalty,
            min_new_tokens=1,
            max_new_tokens=max_new,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            bad_words_ids=bad_ids,
        )
    return decode(tok, out)

def badish(s: str) -> bool:
    if not s.strip(): return True
    low = s.strip().lower()
    if re.search(r"\bkeluar\w*\b", low): return True
    if any(low.startswith(p.lower()) for p in ["baik", "sudah begitu", "apakah ini benar", "sedang"]):
        return True
    return False

def translate_one(model, tok, text: str, mode="auto", max_new=24):
    # 1) rules duluan
    ruled = try_rules(text)
    if ruled is not None:
        return ruled

    # 2) prenorm opsional
    src = prenorm(text) if USE_PRENORM else text

    # 3) pilih strategi
    bad_ids = make_bad_words_ids(tok)
    n_words = len(src.split())
    if mode == "greedy" or (mode == "auto" and n_words <= 5):
        outs = generate_once(model, tok, [src], beams=1, max_new=min(max_new, 18),
                             bad_ids=bad_ids, rep_penalty=1.25, no_repeat=2, enc_no_repeat=1)[0]
        out = first_sentence(outs)
        if badish(out):
            # fallback greedy tanpa blacklist
            outs = generate_once(model, tok, [src], beams=1, max_new=min(max_new, 18),
                                 bad_ids=None, rep_penalty=1.25, no_repeat=2, enc_no_repeat=1)[0]
            out = first_sentence(outs)
    elif mode == "beam2" or (mode == "auto" and n_words <= 9):
        outs = generate_once(model, tok, [src], beams=2, max_new=min(max_new, 32),
                             bad_ids=bad_ids, rep_penalty=1.15, no_repeat=3, enc_no_repeat=0)[0]
        out = first_sentence(outs)
    else:  # beam4
        outs = generate_once(model, tok, [src], beams=4, max_new=max_new,
                             bad_ids=bad_ids, rep_penalty=1.12, no_repeat=3, enc_no_repeat=0)[0]
        out = first_sentence(outs)

    # 4) fallback terakhir bila masih kosong/aneh
    if not out.strip():
        low = text.strip().lower()
        if re.match(r"^trada\b", low):
            out = re.sub(r"^trada", "Tidak ada", text, flags=re.IGNORECASE).strip()
        elif re.match(r"^ko lagi\b", low):
            out = re.sub(r"^ko lagi", "Kamu sedang", text, flags=re.IGNORECASE).strip()
        elif re.match(r"^ko su\b", low):
            out = (re.sub(r"^ko su", "Kamu sudah", text, flags=re.IGNORECASE).strip() + "?").replace("??","?")
        elif re.match(r"^sa tra tau\b", low):
            out = "Saya tidak tahu."
        else:
            out = first_sentence(text)
    return out

# ================== LOAD MODEL ==================
def load_models():
    assert os.path.isdir(ADAPTER_DIR), f"Adapter tidak ditemukan: {ADAPTER_DIR}"
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json"))

    tok  = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    tok.padding_side = "right"

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR).to(DEVICE).eval()
    fix_special_tokens(base, tok)
    for p in base.parameters(): p.requires_grad = False

    model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()
    fix_special_tokens(model, tok)
    for p in model.parameters(): p.requires_grad = False

    return tok, model

# ================== REPL ==================
HELP = """Perintah:
  /mode auto|greedy|beam2|beam4   -> ganti strategi decoding (default: auto)
  /max <angka>                    -> set max_new_tokens (default: 24)
  /help                           -> bantuan ini
  /quit                           -> keluar
Cukup ketik kalimat dialek Papua, ENTER untuk terjemahan.
"""

def main():
    tok, model = load_models()
    print("[INFO] Adapter terpasang:", hasattr(model, "peft_config"))
    mode = "auto"
    max_new = 24
    print(">> Mode interaktif. Ketik kalimat dialek Papua, kosong untuk keluar.")
    print(HELP)

    while True:
        try:
            src = input("SRC> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[SELESAI]")
            break

        if not src:
            print("[SELESAI]")
            break

        # perintah
        if src.startswith("/"):
            cmd = src.lower().split()
            if cmd[0] == "/help":
                print(HELP)
            elif cmd[0] == "/mode" and len(cmd) >= 2:
                m = cmd[1]
                if m in {"auto","greedy","beam2","beam4"}:
                    mode = m
                    print(f"[OK] mode -> {mode}")
                else:
                    print("[ERR] mode tidak dikenal. Pilih: auto|greedy|beam2|beam4")
            elif cmd[0] == "/max" and len(cmd) >= 2:
                try:
                    val = int(cmd[1])
                    max_new = max(8, min(96, val))
                    print(f"[OK] max_new_tokens -> {max_new}")
                except:
                    print("[ERR] /max <angka>")
            elif cmd[0] == "/quit":
                print("[SELESAI]")
                break
            else:
                print("[ERR] perintah tidak dikenal. /help untuk bantuan.")
            continue

        # terjemahkan
        mt = translate_one(model, tok, src, mode=mode, max_new=max_new)
        print("MT >", mt)

if __name__ == "__main__":
    main()
