# inspect_db.py
import sqlite3, os, sys

DB_PATH = "C:\cendol-mt5\pap2id.db"
if not os.path.exists(DB_PATH):
    print(f"[ERR] DB tidak ditemukan: {DB_PATH}")
    sys.exit(1)

con = sqlite3.connect(DB_PATH)
con.row_factory = sqlite3.Row
cur = con.cursor()

print("[OK] Terkoneksi ke:", DB_PATH)

# 1) Daftar tabel
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = [r["name"] for r in cur.fetchall()]
print("\n[TABLES]")
for t in tables:
    print(" -", t)

# 2) Skema tabel penting
def show_schema(tname):
    print(f"\n[SCHEMA] {tname}")
    for col in con.execute(f"PRAGMA table_info({tname});"):
        # cid, name, type, notnull, dflt_value, pk
        print(f"  - {col[1]}  {col[2]}  NOTNULL={col[3]}  PK={col[5]}")

for t in ("users","history","sessions"):
    try:
        show_schema(t)
    except Exception as e:
        print(f"  (skip {t}): {e}")

# 3) Sampel isi
def sample_rows(tname, n=5):
    try:
        rows = con.execute(f"SELECT * FROM {tname} ORDER BY 1 DESC LIMIT {n};").fetchall()
        print(f"\n[SAMPLE {tname} ({len(rows)} rows)]")
        for r in rows:
            d = dict(r)
            # jangan tampilkan hash
            if "pass_hash" in d: d["pass_hash"] = "***"
            print(d)
    except Exception as e:
        print(f"  (skip {tname}): {e}")

for t in ("users","history","sessions"):
    sample_rows(t)

# 4) Hitung ringkas
print("\n[COUNTS]")
for t in tables:
    try:
        c = con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        print(f" - {t}: {c}")
    except:
        pass

con.close()
