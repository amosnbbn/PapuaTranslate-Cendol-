# app.py
import os, re
from typing import Optional, List, Tuple

from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required
)
from werkzeug.security import generate_password_hash, check_password_hash

# ====== DB (PostgreSQL via SQLAlchemy) ======
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, func, text as sql_text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, scoped_session

# ====== Model MT (mT5 + LoRA) ======
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ---------------------------------------------------------
# KONFIG APLIKASI (ENV-DRIVEN)
# ---------------------------------------------------------
# DB: pakai Postgres lokal by default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://papua_user:nababan04@127.0.0.1:5432/papua_db"
)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-please-change")

# Model lokal (nyalakan dengan LOAD_MODEL=1)
LOAD_MODEL   = os.getenv("LOAD_MODEL", "0") in ("1", "true", "True")
FORCE_CPU    = os.getenv("FORCE_CPU",  "0") in ("1", "true", "True")  # paksa CPU (debug)
BASE_DIR     = os.getenv("BASE_DIR",    "C:\ML\models\cendol-mt5-base-inst")
ADAPTER_DIR  = os.getenv("ADAPTER_DIR", "C:\cendol-mt5\outputs\pap2id_phase8_fast_e10\checkpoint-168")

# Debug: ijinkan akses translate tanpa login (untuk cek cepat di lokal)
DEBUG_ALLOW_OPEN = os.getenv("DEBUG_ALLOW_OPEN", "0") in ("1", "true", "True")

# Device
DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"

# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="static"
)
app.secret_key = SECRET_KEY
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False  # set True bila full HTTPS
)

# ---------------------------------------------------------
# DB SETUP
# ---------------------------------------------------------
# Catatan: untuk lokal gunakan pool default.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

class User(Base, UserMixin):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True)
    email      = Column(String(255), unique=True, nullable=False, index=True)
    pass_hash  = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    histories  = relationship("History", back_populates="user", cascade="all,delete")

class History(Base):
    __tablename__ = "history"
    id         = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    src        = Column(Text, nullable=False)
    mt         = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    user       = relationship("User", back_populates="histories")

def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Ready: {DATABASE_URL}")

def db():
    return SessionLocal()

@app.teardown_appcontext
def remove_scoped_session(_exc=None):
    SessionLocal.remove()

# ---------------------------------------------------------
# LOGIN MANAGER
# ---------------------------------------------------------
login_manager = LoginManager()
login_manager.login_view = "login_get"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    s = db()
    try:
        return s.get(User, int(user_id))
    finally:
        s.close()

# ---------------------------------------------------------
# UTIL RESPON
# ---------------------------------------------------------
def wants_json_response() -> bool:
    if request.is_json:
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept.lower()

# ---------------------------------------------------------
# TRANSLATION UTIL (prenorm, prompt, rerank)
# ---------------------------------------------------------
def strip_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def make_prompt(src: str) -> str:
    return f"Instruksi: Terjemahkan ke bahasa Indonesia baku.\nTeks: {src}\nKeluaran:"

PRE_NORM = [
    (r"\bngana\b", "kamu"), (r"\bkoi\b", "kamu"), (r"\bdorang\b", "mereka"),
    (r"\btong\b", "kita"),  (r"\bgosi\b", "kemaluan pria"),
    (r"\bpace\b", "pace"),  (r"\bde\b", "dia"),   (r"\bsa\b", "saya"),
    (r"\bko\b", "kamu"),    (r"\bkam\b", "kalian"), (r"\bmo\b", "mau"),
    (r"\bsu\b", "sudah"),   (r"\btra\b", "tidak"),  (r"\btrada\b", "tidak ada"),
    (r"\btrakan\b", "tidak akan"), (r"\bbikin\b", "buat"),
    (r"\bpigi\b", "pergi"), (r"\bdeng\b", "dengan"), (r"\bdimana\b", "di mana"),
    (r"\bsampe\b", "sampai"), (r"\bstlelan\b", "setelan"),
    (r"\bpipi\s+congka\b", "lesung pipi"),
    (r"\btoki\b", "ikuti"),
    (r"\b([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+e\b", r"\1"),
]

def prenorm(text: str) -> str:
    out = text
    for pat, rep in PRE_NORM:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return strip_spaces(out)

def possessive_rewrite(src: str) -> str:
    pat = re.compile(r"\b(sa|saya|ko|kamu|kita|kam|kalian|dia|mereka)\s+p(?:u|ung)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)(?:\s+(.*))?", flags=re.IGNORECASE)
    def repl(m):
        pron = m.group(1).lower()
        noun = m.group(2)
        tail = m.group(3) or ""
        owner = {
            "sa": "saya", "saya": "saya", "ko": "kamu", "kamu": "kamu",
            "kita": "kita", "kam": "kalian", "kalian": "kalian",
            "dia": "dia", "mereka": "mereka",
        }.get(pron, pron)
        out = f"{noun} {owner}".strip()
        if tail: out += " " + tail
        return out
    return re.sub(pat, repl, src)

BAD_PHRASES = [
    "berikut","artikel","seri ini","pelajaran","pengantar","menurut",
    "dalam perjalananku","cerita","kisah","bab","paragraf",
    "air anggur","alkohol","anggur","pesta",
    "jawab","jawaban","berkata","katanya","mengatakan","mengucap","lalu","bahwa",
    "keluar","keluaran","keluarannya","keluari",
    "Pak ","Bu ","Tuan ","Nyonya ",
    "Terima kasih","Maaf",
]
PREFERRED_START = ["saya","aku","kamu","anda","dia","kita","kalian","mereka","apakah","tidak","ada","jangan","seperti"]

def make_bad_words_ids(tok) -> List[List[int]]:
    ids = []
    for phrase in BAD_PHRASES:
        toks = tok(phrase, add_special_tokens=False).input_ids
        if toks: ids.append(toks)
    return ids

def clean_after_decode(t: str) -> str:
    t = re.sub(r"\bKeluaran\b\s*:?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*(Terjemahan\s+Bahasa\s+Indonesia\s*:)\s*", "", t, flags=re.IGNORECASE)
    return strip_spaces(t)

def first_sentence(s: str, max_words: int = 20) -> str:
    s = strip_spaces(s)
    s = re.split(r'\b(kata|jawab|berkata|katanya|mengatakan|mengucap|lalu|bahwa)\b', s, flags=re.IGNORECASE)[0]
    s = re.split(r'[""()]', s)[0]
    m = re.search(r"[.!?…]|。|？|！", s)
    if m: s = s[:m.end()]
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])
        if not re.search(r"[.!?…]$", s): s += "."
    s = re.sub(r"\bkeluar\w*\b", "", s, flags=re.IGNORECASE)
    return strip_spaces(s)

def score_candidate(src_raw: str, cand: str) -> Tuple[int,int,int,int]:
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
    if out and out[0].islower(): out = out[0].upper() + out[1:]
    if out and out[-1].isalnum(): out += "."
    return out.replace("..", ".")

# ---------------------------------------------------------
# LOAD MODEL (opsional; matikan di Vercel)
# ---------------------------------------------------------
TOK: Optional[AutoTokenizer] = None
MODEL: Optional[AutoModelForSeq2SeqLM] = None

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

def load_models_if_requested():
    global TOK, MODEL
    if not LOAD_MODEL:
        print("[INFO] LOAD_MODEL=0 → model tidak diload (mode serverless/production).")
        return
    print("------------------------------------------------------------")
    print("[MODEL] Starting load…")
    print(f"[MODEL] Device     : {DEVICE} | FORCE_CPU={FORCE_CPU}")
    print(f"[MODEL] Base dir   : {BASE_DIR}")
    print(f"[MODEL] Adapter dir: {ADAPTER_DIR}")

    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_DIR)
    base = base.to(DEVICE).eval()
    fix_special_tokens(base, tok)
    for p in base.parameters(): p.requires_grad = False

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.to(DEVICE).eval()
    fix_special_tokens(model, tok)
    for p in model.parameters(): p.requires_grad = False

    TOK, MODEL = tok, model
    print("[MODEL] Loaded OK")

def generate_nbest(source: str, *, beams=6, nbest=6, max_new=24,
                   rep_penalty=1.12, no_repeat=4, bad_ids=None) -> List[str]:
    prompt = make_prompt(source)
    enc = TOK([prompt], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.inference_mode():
        out = MODEL.generate(
            **enc,
            num_beams=beams,
            num_return_sequences=nbest,
            do_sample=False,
            no_repeat_ngram_size=no_repeat,
            repetition_penalty=rep_penalty,
            min_new_tokens=1,
            max_new_tokens=max_new,
            eos_token_id=TOK.eos_token_id,
            pad_token_id=TOK.pad_token_id,
            bad_words_ids=bad_ids,
        )
    return TOK.batch_decode(out, skip_special_tokens=True)

def translate_one(raw_text: str, max_new: int = 24) -> str:
    if TOK is None or MODEL is None:
        raise RuntimeError("Model tidak tersedia (LOAD_MODEL=0).")
    src0 = possessive_rewrite(raw_text)
    src  = prenorm(src0)
    bad_ids = make_bad_words_ids(TOK)
    cands = generate_nbest(src, beams=6, nbest=6, max_new=max_new,
                           rep_penalty=1.12, no_repeat=4, bad_ids=bad_ids)
    out = rerank_nbest(raw_text, cands)
    if raw_text.strip().endswith("?") and not out.endswith("?"):
        out = out.rstrip(".") + "?"
    return out

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.before_request
def force_login_first():
    """Kalau belum login, paksa ke /login kecuali beberapa route publik."""
    if DEBUG_ALLOW_OPEN:
        return  # skip guard di mode debug

    public_paths = {"/login", "/register", "/about", "/health"}
    if request.path.startswith("/static/"):   # izinkan file statis
        return
    if request.path in public_paths:
        return
    if not current_user.is_authenticated:
        return redirect(url_for("login_get"))

@app.get("/")
def home():
    recent = []
    if current_user.is_authenticated:
        s = db()
        try:
            rows = (
                s.query(History)
                .filter(History.user_id == current_user.id)
                .order_by(History.created_at.desc())
                .limit(10).all()
            )
            recent = [
                {"src": h.src, "mt": h.mt, "created_at": h.created_at.strftime("%Y-%m-%d %H:%M")}
                for h in rows
            ]
        finally:
            s.close()
    return render_template(
        "index.html",
        device=DEVICE,
        logged_in=current_user.is_authenticated,
        user_email=(current_user.email if current_user.is_authenticated else ""),
        recent=recent
    )

@app.get("/about")
def about():
    return render_template("about.html", device=DEVICE)

# ---------- AUTH ----------
@app.get("/login")
def login_get():
    return render_template("login.html")

@app.post("/login")
def login_post():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    if not email or not password:
        if wants_json_response(): return jsonify({"error": "Email & password wajib diisi"}), 400
        return redirect(url_for("login_get"))

    s = db()
    try:
        u = s.query(User).filter_by(email=email).first()
        if not u or not check_password_hash(u.pass_hash, password):
            if wants_json_response(): return jsonify({"error": "Email atau password salah"}), 401
            return redirect(url_for("login_get"))
        login_user(u)
    finally:
        s.close()

    if wants_json_response(): return jsonify({"ok": True})
    return redirect(url_for("home"))

@app.get("/register")
def register_get():
    return render_template("register.html")

@app.post("/register")
def register_post():
    data = request.get_json(silent=True) or request.form
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    if not email or not password:
        if wants_json_response(): return jsonify({"error": "Email & password wajib diisi"}), 400
        return redirect(url_for("register_get"))

    s = db()
    try:
        exists = s.query(User).filter_by(email=email).first()
        if exists:
            if wants_json_response(): return jsonify({"error": "Email sudah terdaftar"}), 400
            return redirect(url_for("login_get"))
        u = User(email=email, pass_hash=generate_password_hash(password))
        s.add(u); s.commit()
        login_user(u)
    finally:
        s.close()

    if wants_json_response(): return jsonify({"ok": True})
    return redirect(url_for("home"))

@app.get("/logout")
def logout():
    if current_user.is_authenticated:
        logout_user()
    return redirect(url_for("login_get"))

# ---------- API ----------
@app.post("/translate")
@login_required
def translate_api():
    data = request.get_json(force=True, silent=True) or {}
    text_in = (data.get("text") or "").strip()
    if not text_in:
        return jsonify({"error": "field 'text' wajib diisi"}), 400

    max_new = int(data.get("max_new_tokens", 24))
    max_new = max(8, min(64, max_new))

    try:
        mt = translate_one(text_in, max_new=max_new)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"gagal generate: {e}"}), 500

    s = db()
    try:
        s.add(History(user_id=current_user.id, src=text_in, mt=mt))
        s.commit()
    finally:
        s.close()

    return jsonify({"src": text_in, "mt": mt})

@app.get("/history")
@login_required
def history_api():
    s = db()
    try:
        items = (
            s.query(History)
            .filter(History.user_id == current_user.id)
            .order_by(History.created_at.desc())
            .limit(50).all()
        )
        out = [{"src": h.src, "mt": h.mt, "created_at": h.created_at.strftime("%Y-%m-%d %H:%M")} for h in items]
        return jsonify({"items": out})
    finally:
        s.close()

# Endpoint debug: tanpa login (gunakan untuk cek cepat dari index saat DEBUG_ALLOW_OPEN=1)
@app.post("/debug/translate")
def debug_translate_open():
    if not DEBUG_ALLOW_OPEN:
        return jsonify({"error": "disabled"}), 403
    data = request.get_json(force=True, silent=True) or {}
    text_in = (data.get("text") or "").strip()
    if not text_in:
        return jsonify({"error": "field 'text' wajib diisi"}), 400
    try:
        mt = translate_one(text_in, max_new=int(data.get("max_new_tokens", 24)))
    except Exception as e:
        return jsonify({"error": f"gagal generate: {e}"}), 500
    return jsonify({"src": text_in, "mt": mt})

@app.get("/health")
def health():
    # Redaksi DATABASE_URL (jangan bocor password)
    redacted_db = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", DATABASE_URL)
    ok = True; err = None
    try:
        with engine.connect() as c:
            c.execute(sql_text("select 1"))
    except Exception as e:
        ok, err = False, str(e)

    return jsonify({
        "ok": ok,
        "err": err,
        "device": DEVICE,
        "db": redacted_db,
        "model_flags": {"LOAD_MODEL": LOAD_MODEL, "FORCE_CPU": FORCE_CPU},
        "model_loaded": LOAD_MODEL and (TOK is not None and MODEL is not None),
        "debug_allow_open": DEBUG_ALLOW_OPEN,
        "logged_in": current_user.is_authenticated
    })

# ---------------------------------------------------------
# MAIN (local only)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("============================================================")
    print("PapuaTranslate (Flask 3.x) — LOCAL MODE")
    print("============================================================")
    print(f"DEVICE    : {DEVICE}")
    print(f"LOAD_MODEL: {LOAD_MODEL} | FORCE_CPU: {FORCE_CPU}")
    print(f"BASE_DIR  : {BASE_DIR}")
    print(f"ADAPTER   : {ADAPTER_DIR}")
    init_db()
    load_models_if_requested()
    print("Server: http://localhost:8000")
    print("Templates dir:", os.path.abspath(app.template_folder))
    print("Static dir   :", os.path.abspath(app.static_folder))
    app.run(host="0.0.0.0", port=8000, debug=True)
