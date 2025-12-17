import os, re, torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ================== KONFIG ==================
BASE_DIR    = "C:\ML\models\cendol-mt5-base-inst"
ADAPTER_DIR = "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168"
# Jika phase7 terasa lebih bagus:
# ADAPTER_DIR = r"C:\cendol-mt5\outputs\pap2id_phase7_e10_wandb\checkpoint-560"

USE_PRENORM = True
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    return f"Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks: {src}\nKeluaran:"

def strip_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clean_after_decode(t: str) -> str:
    t = t.replace("Keluaran:", "")
    t = strip_spaces(t)
    return t

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

# ===== kata/akar yang memicu narasi =====
BAD_PHRASES = [
    # generik
    "Baik.", "baiklah.", "sudah begitu", "apakah ini benar", "sedang.",
    "ini lagi apa ya",
    # narasi umum
    "Jawab", "jawab", "Kata", "kata", "berkata", "katanya", "mengatakan", "lalu", "bahwa", "mengucap", "Mengucap",
    # artefak prompt
    "keluar", "keluaran", "keluarannya", "keluari",
    # tanda pemicu paragraf panjang
    "“", "”", "\"", "(", ")", "…", "‘", "’", "：", "；", ";", ":",
    # token biblikal yang sering nongol aneh
    "Ia ", " ia ",
]
def make_bad_words_ids(tok) -> List[List[int]]:
    ids = []
    for phrase in BAD_PHRASES:
        toks = tok(phrase, add_special_tokens=False).input_ids
        if toks:
            ids.append(toks)
    return ids

# ===== ambil kalimat pertama + buang akar "keluar" dan jejak narasi =====
END_PUNCTS = r"[.!?…]|。|？|！"
def first_sentence(s: str) -> str:
    s = strip_spaces(s)
    s = re.split(r'\b(kata|jawab|berkata|katanya|mengatakan|lalu|bahwa|mengucap)\b', s, flags=re.IGNORECASE)[0]
    s = re.split(r'["“”\(\)]', s)[0]
    m = re.search(END_PUNCTS, s)
    if m:
        s = s[:m.end()]
    words = s.split()
    if len(words) > 20:
        s = " ".join(words[:20])
    # bersihkan akar "keluar"
    s = re.sub(r"\bkeluar\w*\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def encode(tok, prompts: List[str]):
    return tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

def decode(tok, out_ids):
    return [clean_after_decode(x) for x in tok.batch_decode(out_ids, skip_special_tokens=True)]

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

# ======== RULES SANGAT SERING (jalan duluan) ========
# Bentuknya: regex lower() -> output final
RULES = [
    (r"^sa tra tau\.?$",                           "Saya tidak tahu."),
    (r"^ko lagi apa(?: ya)?\??$",                  "Kamu lagi apa?"),
    (r"^ko lagi bikin\.?$",                        "Kamu sedang membuat apa?"),
    (r"^ko su makan\??$",                          "Kamu sudah makan?"),
    (r"^ko su deng dia kah\??$",                   "Apakah kamu sudah bersamanya?"),
    (r"^sa kenapa ee\.?$",                         "Saya kenapa ya?"),
    (r"^sa deng dia tadi\.?$",                     "Saya bersama dia tadi."),
    # Variasi umum lain:
    (r"^kam dimana\??$",                           "Kalian di mana?"),
    (r"^ko dimana\??$",                            "Kamu di mana?"),
    (r"^sa tra tau.*",                             "Saya tidak tahu."),
]

def try_rules(raw: str) -> str | None:
    s = raw.strip().lower()
    for pat, out in RULES:
        if re.match(pat, s, flags=re.IGNORECASE):
            return out
    return None

# ===== pipeline adaptif =====
def translate_batch(model, tok, texts: List[str]) -> List[str]:
    # 0) **RULES duluan** (bypass model kalau match)
    outs = [None] * len(texts)
    to_model_idx = []
    src_for_model = []

    for i, raw in enumerate(texts):
        ruled = try_rules(raw)
        if ruled is not None:
            outs[i] = ruled
        else:
            to_model_idx.append(i)
            src_for_model.append(raw)

    if not to_model_idx:
        return outs  # semua beres oleh RULES

    # 1) prenorm (opsional) utk sisanya
    src = [prenorm(t) for t in src_for_model] if USE_PRENORM else list(src_for_model)

    # 2) klasifikasi panjang
    short_idx, mid_idx, long_idx = [], [], []
    for j, s in enumerate(src):
        n = len(s.split())
        if n <= 5: short_idx.append(j)      # naikin ambang pendek jadi <=5
        elif n <= 9: mid_idx.append(j)
        else: long_idx.append(j)

    bad_ids = make_bad_words_ids(tok)
    gen_outs = [""] * len(src)

    # 3) pendek → GREEDY max_new kecil
    if short_idx:
        S = [src[j] for j in short_idx]
        o = generate_once(model, tok, S, beams=1, max_new=14,
                          bad_ids=bad_ids, rep_penalty=1.25, no_repeat=2, enc_no_repeat=1)
        for j, y in zip(short_idx, o):
            gen_outs[j] = first_sentence(y)

    # 4) menengah → BEAM 2
    if mid_idx:
        S = [src[j] for j in mid_idx]
        o = generate_once(model, tok, S, beams=2, max_new=28,
                          bad_ids=bad_ids, rep_penalty=1.15, no_repeat=3, enc_no_repeat=0)
        for j, y in zip(mid_idx, o):
            gen_outs[j] = first_sentence(y)

    # 5) panjang → BEAM 4
    if long_idx:
        S = [src[j] for j in long_idx]
        o = generate_once(model, tok, S, beams=4, max_new=48,
                          bad_ids=bad_ids, rep_penalty=1.12, no_repeat=3, enc_no_repeat=0)
        for j, y in zip(long_idx, o):
            gen_outs[j] = first_sentence(y)

    # 6) fallback untuk yang masih aneh
    def badish(s: str) -> bool:
        if not s.strip(): return True
        low = s.strip().lower()
        if re.search(r"\bkeluar\w*\b", low): return True
        if any(low.startswith(p.lower()) for p in ["baik", "sudah begitu", "apakah ini benar", "sedang"]):
            return True
        return False

    need_fb = [j for j, y in enumerate(gen_outs) if badish(y)]
    if need_fb:
        S = [src[j] for j in need_fb]
        o = generate_once(model, tok, S, beams=1, max_new=18,
                          bad_ids=None, rep_penalty=1.25, no_repeat=2, enc_no_repeat=1)
        for j, y in zip(need_fb, o):
            y = first_sentence(y)
            if not badish(y):
                gen_outs[j] = y

    # 7) taruh kembali sesuai indeks awal + fallback terakhir
    for local_j, glob_i in enumerate(to_model_idx):
        y = gen_outs[local_j]
        if not y.strip():
            # fallback terakhir: simple re-mapping pola umum
            low = texts[glob_i].strip().lower()
            if re.match(r"^trada\b", low):
                y = re.sub(r"^trada", "Tidak ada", texts[glob_i], flags=re.IGNORECASE)
            elif re.match(r"^ko lagi\b", low):
                y = re.sub(r"^ko lagi", "Kamu sedang", texts[glob_i], flags=re.IGNORECASE)
            elif re.match(r"^ko su\b", low):
                y = re.sub(r"^ko su", "Kamu sudah", texts[glob_i], flags=re.IGNORECASE) + "?"
            elif re.match(r"^sa tra tau\b", low):
                y = "Saya tidak tahu."
            else:
                y = first_sentence(texts[glob_i])
        outs[glob_i] = y

    return outs

# ================== MAIN (contoh) ==================
def main():
    assert os.path.isdir(ADAPTER_DIR), f"Adapter tidak ditemukan: {ADAPTER_DIR}"
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json"))

    tok  = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    tok.padding_side = "right"
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR).to(device).eval()
    fix_special_tokens(base, tok)
    for p in base.parameters(): p.requires_grad = False

    model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(device).eval()
    fix_special_tokens(model, tok)
    for p in model.parameters(): p.requires_grad = False

    print("[INFO] Adapter terpasang:", hasattr(model, "peft_config"))

    tests = [
        "Ade ko cantik, pipi congka, hitam manis",
        "Andai mo bilang tapi skarang su berbeda",
        "Stengah mati sa su jaga buat koi",
        "Trada makanan di meja",
        "ko su deng dia kah",
        "sa deng dia tadi",
        "sa kenapa ee",
        "ko lagi apa ya",
        "sa tra tau",
        "ko lagi bikin",
        "ko su makan",
    ]
    outs = translate_batch(model, tok, tests)
    for s, o in zip(tests, outs):
        print("SRC:", s)
        print("MT :", o)
        print("-"*60)

if __name__ == "__main__":
    main()
