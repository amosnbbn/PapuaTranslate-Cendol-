# app.py
import os, re, torch
from typing import List, Optional, Tuple
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ================== KONFIGURASI ==================
BASE_DIR    = os.getenv("BASE_DIR",    "C:\\ML\\models\\cendol-mt5-base-inst")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "C:\\cendol-mt5\\outputs\\pap2id_phase8_fast_e10\\checkpoint-168")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_PRENORM = True
BEAMS  = 6
N_BEST = 6
MAX_NEW_DEFAULT = 24

# ================== UTIL RINGAN ==================
def strip_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def make_prompt(src: str) -> str:
    return f"Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks: {src}\nKeluaran:"

def clean_after_decode(t: str) -> str:
    t = re.sub(r"\bKeluaran\b\s*:?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*(Terjemahan\s+Bahasa\s+Indonesia\s*:)\s*", "", t, flags=re.IGNORECASE)
    t = strip_spaces(t)
    return t

def fix_special_tokens(model, tok):
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tok.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tok.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.forced_bos_token_id = None
    model.config.forced_eos_token_id = None
    model.config.use_cache = True

# ================== PRENORM ==================
PRE_NORM = [
    (r"\bngana\b",  "kamu"),
    (r"\bkoi\b",    "kamu"),
    (r"\bdorang\b", "mereka"),
    (r"\btong\b",   "kita"),
    (r"\bgosi\b",   "kemaluan pria"),
    (r"\bpace\b",   "pace"),
    (r"\bde\b",     "dia"),
    (r"\bsa\b",     "saya"),
    (r"\bko\b",     "kamu"),
    (r"\bkam\b",    "kalian"),
    (r"\bmo\b",     "mau"),
    (r"\bsu\b",     "sudah"),
    (r"\btra\b",    "tidak"),
    (r"\btrada\b",  "tidak ada"),
    (r"\btrakan\b", "tidak akan"),
    (r"\bbikin\b",  "buat"),
    (r"\bpigi\b",   "pergi"),
    (r"\bdeng\b",   "dengan"),
    (r"\bdimana\b", "di mana"),
    (r"\bsampe\b",  "sampai"),
    (r"\bstlelan\b","setelan"),
    (r"\bpipi\s+congka\b", "lesung pipi"),
    (r"\btoki\b",   "ikuti"),
    (r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+e\b", r"\1"),
]

def prenorm(text: str, enable=True) -> str:
    if not enable:
        return strip_spaces(text)
    out = text
    for pat, rep in PRE_NORM:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return strip_spaces(out)

# ================== REWRITE KEPEMILIKAN ==================
def possessive_rewrite(src: str) -> str:
    pat = re.compile(r"\b(sa|saya|ko|kamu|kita|kam|kalian|dia|mereka)\s+p(?:u|ung)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)(?:\s+(.*))?", flags=re.IGNORECASE)
    def repl(m):
        pron = m.group(1).lower()
        noun = m.group(2)
        tail = m.group(3) or ""
        owner = {
            "sa":"saya","saya":"saya","ko":"kamu","kamu":"kamu",
            "kita":"kita","kam":"kalian","kalian":"kalian",
            "dia":"dia","mereka":"mereka",
        }.get(pron, pron)
        out = f"{noun} {owner}".strip()
        if tail: out += " " + tail
        return out
    return re.sub(pat, repl, src)

# ================== HEURISTIK PRIORITAS TINGGI ==================
def heuristic_override(raw_src: str) -> Optional[str]:
    s = strip_spaces(raw_src.lower())
    if re.fullmatch(r"(ko|kamu)\s+(sudah|su)\s+di\s*mana\??", s):
        return "Kamu sudah di mana?"
    if re.fullmatch(r"(ko|kamu)\s+(sudah|su)\s+makan\??", s):
        return "Kamu sudah makan?"
    if re.fullmatch(r"(ko|kamu)\s+lagi\s+apa(?:\s*ya)?\??", s):
        return "Kamu lagi apa?"
    if re.fullmatch(r"(ko|kamu)\s+lagi\s+bikin\.?", s):
        return "Kamu sedang membuat apa?"
    if re.fullmatch(r"(ko|kamu)\s+(sudah|su)\s+dengan\s+dia\s+kah\??", s):
        return "Apakah kamu sudah bersamanya?"
    if re.fullmatch(r"(sa|saya)\s+(tra|tidak)\s+tau\.?", s):
        return "Saya tidak tahu."
    if re.fullmatch(r"(sa|saya)\s+(tra|tidak)\s+ada\s+uang\.?", s) or re.fullmatch(r"(sa|saya)\s+trada\s+uang\.?", s):
        return "Saya tidak ada uang."
    if re.fullmatch(r"(kam|kalian)\s+di\s*mana\??", s):
        return "Kalian di mana?"
    if re.fullmatch(r"(ko|kamu)\s+di\s*mana\??", s):
        return "Kamu di mana?"
    if re.fullmatch(r"pace\s+(ko|kamu)\s+di\s*mana\??", s):
        return "Pace, kamu di mana?"
    if re.match(r"^trada\b", s):
        out = re.sub(r"^trada", "Tidak ada", raw_src, flags=re.IGNORECASE).strip()
        if out and out[-1].isalnum(): out += "."
        return out
    return None

# ================== GUARDRAILS & RERANK ==================
BAD_PHRASES = [
    "berikut", "artikel", "seri ini", "pelajaran", "pengantar", "menurut",
    "dalam perjalananku", "cerita", "kisah", "bab", "paragraf",
    "air anggur", "alkohol", "anggur", "pesta",
    "jawab", "jawaban", "berkata", "katanya", "mengatakan", "mengucap", "lalu", "bahwa",
    "keluar", "keluaran", "keluarannya", "keluari",
    "Pak ", "Bu ", "Tuan ", "Nyonya ",
    "Terima kasih", "Maaf",
]
PREFERRED_START = [
    "saya", "aku", "kamu", "anda", "dia", "kita", "kalian", "mereka", "apakah", "tidak", "ada", "jangan", "seperti"
]

def make_bad_words_ids(tok) -> List[List[int]]:
    ids = []
    for phrase in BAD_PHRASES:
        toks = tok(phrase, add_special_tokens=False).input_ids
        if toks: ids.append(toks)
    return ids

END_PUNCTS = r"[.!?…]|。|？|！"

def first_sentence(s: str, max_words: int = 20) -> str:
    s = strip_spaces(s)
    s = re.split(r'\b(kata|jawab|berkata|katanya|mengatakan|mengucap|lalu|bahwa)\b', s, flags=re.IGNORECASE)[0]
    s = re.split(r'[""()]', s)[0]
    m = re.search(END_PUNCTS, s)
    if m: s = s[:m.end()]
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
        if not re.search(END_PUNCTS + r"$", s):
            s += "."
    s = re.sub(r"\bkeluar\w*\b", "", s, flags=re.IGNORECASE)
    return strip_spaces(s)

def score_candidate(src_raw: str, cand: str) -> Tuple[int, int, int, int]:
    low = cand.lower()
    bad_hits = sum(1 for w in BAD_PHRASES if w.lower() in low)
    length = len(low.split())
    pref_bonus = -2 if any(low.startswith(p + " ") for p in PREFERRED_START) else 0
    quote_pen = 1 if any(ch in cand for ch in ['"', '(', ')', ':', ';']) else 0
    return (bad_hits, length, pref_bonus, quote_pen)

def rerank_nbest(src_raw: str, cands: List[str]) -> str:
    pruned = [first_sentence(clean_after_decode(c)) for c in cands]
    scored = sorted(pruned, key=lambda x: score_candidate(src_raw, x))
    out = scored[0].strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if out and out[-1].isalnum():
        out += "."
    out = out.replace("..", ".")
    return out

SUBJ_MAP = {
    "saya":"Saya", "sa":"Saya",
    "kamu":"Kamu", "ko":"Kamu",
    "kita":"Kita", "tong":"Kita",
    "kalian":"Kalian", "kam":"Kalian",
    "dia":"Dia", "de":"Dia",
    "mereka":"Mereka", "dorang":"Mereka",
}
VERB_OR_ADV_START = re.compile(r"^(sedang|sudah|telah|akan|mau|ingin|harus|bisa|boleh|jangan|segera|lagi|mulai|ayo|tolong|cepat|perlahan|diam|tenang|hati-hati|masak|memasak|buat|makan|minum|pergi|ikut|ikuti|ganggu|tunggu|jaga|jalan|belajar|baca|menulis|tulis|lihat|dengar|dengarkan)\b", re.IGNORECASE)

def detect_subject_from_src(src_norm: str) -> Optional[str]:
    tokens = src_norm.lower().split()
    for t in tokens[:4]:
        if t in SUBJ_MAP:
            return SUBJ_MAP[t]
    return None

def ensure_subject(src_norm: str, out: str) -> str:
    subj = detect_subject_from_src(src_norm)
    if not subj: return out
    low = out.lower()
    if low.startswith(("saya ","aku ","kamu ","anda ","dia ","kita ","kalian ","mereka ","apakah ","tidak ","ada ","jangan ","seperti ")):
        return out
    if VERB_OR_ADV_START.match(low):
        return f"{subj} {out[0].lower()+out[1:] if out else ''}".strip()
    return out

# ================== PEMUATAN MODEL ==================
def load_models():
    assert os.path.isdir(ADAPTER_DIR), f"Adapter tidak ditemukan: {ADAPTER_DIR}"
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors"))
    assert os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_config.json"))

    tok   = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    tok.padding_side = "right"
    base  = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR).to(DEVICE).eval()
    fix_special_tokens(base, tok)
    for p in base.parameters(): p.requires_grad = False

    model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()
    fix_special_tokens(model, tok)
    for p in model.parameters(): p.requires_grad = False
    return tok, model

TOK, MODEL = load_models()

def generate_nbest(model, tok, source: str, *, beams=BEAMS, nbest=N_BEST,
                   max_new=MAX_NEW_DEFAULT, rep_penalty=1.12, no_repeat=4, bad_ids=None) -> List[str]:
    prompt = make_prompt(source)
    enc = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            num_beams=beams,
            num_return_sequences=nbest,
            do_sample=False,
            no_repeat_ngram_size=no_repeat,
            repetition_penalty=rep_penalty,
            min_new_tokens=1,
            max_new_tokens=max_new,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            bad_words_ids=bad_ids,
        )
    return tok.batch_decode(out, skip_special_tokens=True)

def translate_one(raw_text: str, max_new: int = MAX_NEW_DEFAULT) -> str:
    src_step = possessive_rewrite(raw_text)
    ruled = heuristic_override(src_step)
    if ruled is not None:
        return ruled
    src_norm = prenorm(src_step, enable=USE_PRENORM)

    bad_ids = make_bad_words_ids(TOK)
    cands = generate_nbest(MODEL, TOK, src_norm, beams=BEAMS, nbest=N_BEST, max_new=max_new,
                           rep_penalty=1.12, no_repeat=4, bad_ids=bad_ids)
    out = rerank_nbest(raw_text, cands)

    out = ensure_subject(src_norm, out)

    out = strip_spaces(out)
    if raw_text.strip().endswith("?"):
        if not out.endswith("?"): out = out.rstrip(".") + "?"
    else:
        if out and out[-1].isalnum(): out += "."

    out = out.replace(" .", ".")
    out = out.replace(",,", ",")
    out = re.sub(r"\s+\?", "?", out)
    return out

# ================== FLASK ==================
app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "base_dir": BASE_DIR,
        "adapter_dir": ADAPTER_DIR
    })

@app.post("/translate")
def translate_api():
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "field 'text' wajib diisi"}), 400
    max_new = int(data.get("max_new_tokens", MAX_NEW_DEFAULT))
    max_new = max(8, min(64, max_new))
    out = translate_one(text, max_new=max_new)
    return jsonify({"src": text, "mt": out})

# HTML Template MODERN yang sudah dikoreksi
HTML_MODERN = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PapuaTranslate - Translasi Teks Bahasa Papua</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        /* Header Styles */
        header {
            background-color: #fff;
            color: #000;
            padding: 15px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo h1 {
            font-size: 24px;
            font-weight: 700;
            color: #000;
        }
        
        nav ul {
            display: flex;
            list-style: none;
            gap: 25px;
        }
        
        nav a {
            color: #000;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
            padding: 5px 10px;
        }
        
        nav a:hover {
            background-color: #000;
            color: #fff;
            border-radius: 4px;
        }
        
        /* Hero Section */
        .hero {
            background-color: #fff;
            padding: 60px 0 40px;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
        }
        
        .hero h2 {
            font-size: 32px;
            margin-bottom: 15px;
            color: #000;
        }
        
        .hero p {
            font-size: 18px;
            max-width: 800px;
            margin: 0 auto;
            color: #444;
        }
        
        /* Translation Section */
        .translation {
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 40px;
        }
        
        .translation h3 {
            font-size: 22px;
            margin-bottom: 20px;
            color: #000;
            text-align: center;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
            font-size: 16px;
        }
        
        .input-group textarea {
            width: 100%;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            transition: border-color 0.3s;
            background-color: #f9f9f9;
            min-height: 150px;
            resize: vertical;
        }
        
        .input-group textarea:focus {
            border-color: #555;
            outline: none;
            background-color: #fff;
        }
        
        .language-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .language-label span {
            font-size: 14px;
            color: #666;
        }
        
        .translate-button {
            width: 100%;
            padding: 15px;
            background-color: #000;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.3s;
            margin: 10px 0 20px;
        }
        
        .translate-button:hover {
            background-color: #333;
        }
        
        .translate-button:disabled {
            background-color: #666;
            cursor: not-allowed;
        }
        
        /* Result Section */
        .result {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #eee;
            min-height: 100px;
        }
        
        .result h4 {
            font-size: 16px;
            color: #666;
            margin-bottom: 10px;
        }
        
        .result-text {
            font-size: 16px;
            line-height: 1.6;
            color: #000;
            font-weight: 500;
        }
        
        .loading {
            color: #666;
            font-style: italic;
        }
        
        /* Info Section */
        .info {
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 40px;
        }
        
        .info h3 {
            font-size: 22px;
            margin-bottom: 20px;
            color: #000;
        }
        
        .info p {
            margin-bottom: 15px;
            font-size: 16px;
            color: #444;
        }
        
        /* Footer */
        footer {
            background-color: #fff;
            color: #000;
            padding: 30px 0;
            border-top: 1px solid #eee;
        }
        
        .footer-content {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        
        .footer-section {
            flex-basis: 48%;
            margin-bottom: 20px;
        }
        
        .footer-section h3 {
            font-size: 18px;
            margin-bottom: 15px;
            border-bottom: 2px solid #000;
            padding-bottom: 5px;
            display: inline-block;
        }
        
        .footer-bottom {
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 14px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 15px;
            }
            
            nav ul {
                gap: 15px;
            }
            
            .hero h2 {
                font-size: 28px;
            }
            
            .footer-section {
                flex-basis: 100%;
            }
            
            .translation, .info {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header>
        <div class="container header-content">
            <div class="logo">
                <h1>PapuaTranslate</h1>
            </div>
            <nav>
                <ul>
                    <li><a href="#">About</a></li>
                    <li><a href="#">History</a></li>
                </ul>
            </nav>
        </div>
    </header>

    <!-- Hero Section -->
    <section class="hero">
        <div class="container">
            <h2>Translasi Teks Bahasa Papua ke Bahasa Indonesia Baku</h2>
            <p>Terjemahkan Bahasa Papua ke dalam Bahasa Indonesia yang formal dengan mudah dan akurat menggunakan teknologi AI</p>
        </div>
    </section>

    <!-- Translation Section -->
    <section class="translation">
        <div class="container">
            <h3>Alat Translasi</h3>
            
            <div class="input-group">
                <div class="language-label">
                    <label for="papua-input">Bahasa Papua</label>
                    <span>Masukkan teks dengan Bahasa Papua</span>
                </div>
                <textarea id="papua-input" placeholder="Contoh: 'sa tra tau' akan menjadi 'Saya tidak tahu.'"></textarea>
            </div>
            
            <button class="translate-button" onclick="translateText()" id="translate-btn">Terjemahkan</button>
            
            <div class="input-group">
                <div class="language-label">
                    <label for="indonesia-output">Bahasa Indonesia Baku</label>
                    <span>Hasil terjemahan akan muncul di sini</span>
                </div>
                <div class="result">
                    <h4>Terjemahan:</h4>
                    <div id="indonesia-output" class="result-text">Hasil terjemahan akan ditampilkan di sini...</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Info Section -->
    <section class="info">
        <div class="container">
            <h3>Tentang Aplikasi</h3>
            <p>PapuaTranslate adalah aplikasi yang membantu menerjemahkan logat Papua ke dalam Bahasa Indonesia baku. Aplikasi ini menggunakan teknologi Large Language Model (LLM) yang telah disesuaikan khusus untuk memahami nuansa dan kekhasan logat Papua.</p>
            <p>Dengan PapuaTranslate, Anda dapat dengan mudah memahami makna dari percakapan atau teks berlogat Papua dan mengubahnya menjadi Bahasa Indonesia formal yang lebih mudah dipahami oleh masyarakat luas.</p>
            <p>Aplikasi ini terus berkembang dan belajar dari interaksi pengguna untuk semakin meningkatkan kualitas translasi dari waktu ke waktu.</p>
        </div>
    </section>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>PapuaTranslate</h3>
                    <p>Aplikasi translasi logat Papua ke Bahasa Indonesia baku yang membantu komunikasi dan pemahaman antar budaya.</p>
                </div>
                
                <div class="footer-section">
                    <h3>Informasi Teknis</h3>
                    <p>Model: mT5 Base + LoRA Adapter</p>
                    <p>Device: {{ device }}</p>
                </div>
            </div>
            
            <div class="footer-bottom">
                <p>&copy; 2023 PapuaTranslate. All rights reserved.</p>
            </div>
        </div>
    </footer>

    <script>
        async function translateText() {
            const inputText = document.getElementById('papua-input').value.trim();
            const outputElement = document.getElementById('indonesia-output');
            const translateBtn = document.getElementById('translate-btn');
            
            if (!inputText) {
                outputElement.innerHTML = "Silakan masukkan teks logat Papua untuk diterjemahkan.";
                return;
            }
            
            // Tampilkan loading
            outputElement.innerHTML = '<span class="loading">Menerjemahkan...</span>';
            translateBtn.disabled = true;
            translateBtn.textContent = 'Menerjemahkan...';
            
            try {
                const response = await fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: inputText,
                        max_new_tokens: 24
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    outputElement.innerHTML = data.mt;
                } else {
                    outputElement.innerHTML = 'Error: ' + (data.error || 'Terjadi kesalahan');
                }
            } catch (error) {
                outputElement.innerHTML = 'Error: Gagal terhubung ke server';
                console.error('Error:', error);
            } finally {
                translateBtn.disabled = false;
                translateBtn.textContent = 'Terjemahkan';
            }
        }
        
        // Event listener untuk textarea (Enter + Ctrl untuk menerjemahkan)
        document.getElementById('papua-input').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                translateText();
            }
        });
        
        // Auto-focus pada textarea
        document.getElementById('papua-input').focus();
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_MODERN, device=DEVICE)

if __name__ == "__main__":
    print("=" * 60)
    print("PAPUA TRANSLATE FLASK APP - MODERN UI")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Base model: {BASE_DIR}")
    print(f"Adapter: {ADAPTER_DIR}")
    print("Server running on: http://localhost:8000")
    print("=" * 60)
    print("Pastikan untuk melakukan HARD REFRESH di browser: Ctrl+F5")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=8000, debug=True)