import os, json, argparse
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_values

def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("File JSON harus berupa array of objects.")
    return data

def main():
    ap = argparse.ArgumentParser("Load JSON dataset ke PostgreSQL")
    ap.add_argument("--host", default=os.getenv("PGHOST","localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PGPORT","5432")))
    ap.add_argument("--user", default=os.getenv("PGUSER","papua_user"))
    ap.add_argument("--password", default=os.getenv("PGPASSWORD",""))
    ap.add_argument("--db", default=os.getenv("PGDATABASE","papua_db"))
    ap.add_argument("--file", required=True, help="Path JSON (array of {instruction,input,output[,split]})")
    ap.add_argument("--name", required=True, help="Nama dataset (misal: pap2id_v1)")
    ap.add_argument("--desc", default="", help="Deskripsi dataset")
    ap.add_argument("--default_split", default="train", help="Split default bila tidak ada di JSON")
    ap.add_argument("--batch", type=int, default=1000, help="Ukuran batch insert")
    args = ap.parse_args()

    rows = load_json_array(args.file)
    print(f"[INFO] Load JSON: {args.file} ({len(rows)} rows)")

    conn = psycopg2.connect(
        host=args.host, port=args.port, user=args.user, password=args.password, dbname=args.db
    )
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # upsert sederhana: cek apakah nama dataset sudah ada
            cur.execute("SELECT id FROM datasets WHERE name=%s", (args.name,))
            r = cur.fetchone()
            if r:
                dataset_id = r[0]
                print(f"[INFO] Dataset '{args.name}' sudah ada (id={dataset_id})")
            else:
                cur.execute(
                    "INSERT INTO datasets (name, description) VALUES (%s,%s) RETURNING id",
                    (args.name, args.desc)
                )
                dataset_id = cur.fetchone()[0]
                print(f"[INFO] Dataset baru dibuat: id={dataset_id}")

            # siapkan data untuk insert
            batch = []
            total = 0
            for obj in rows:
                instr = (obj.get("instruction") or "").strip()
                src   = (obj.get("input") or "").strip()
                tgt   = (obj.get("output") or "").strip()
                sp    = (obj.get("split") or args.default_split).strip() or args.default_split

                if not src or not tgt:
                    continue  # skip yang kosong
                raw = json.dumps(obj, ensure_ascii=False)

                batch.append((dataset_id, sp, instr if instr else None, src, tgt, raw))

                if len(batch) >= args.batch:
                    execute_values(cur,
                        """
                        INSERT INTO dataset_items (dataset_id, split, instruction, input, output, raw)
                        VALUES %s
                        """,
                        batch
                    )
                    total += len(batch)
                    print(f"[INFO] Inserted {total} rows …")
                    batch.clear()

            if batch:
                execute_values(cur,
                    """
                    INSERT INTO dataset_items (dataset_id, split, instruction, input, output, raw)
                    VALUES %s
                    """,
                    batch
                )
                total += len(batch)

            print(f"[OK] Selesai insert {total} rows ke dataset_items (dataset_id={dataset_id})")
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("[ERR] Gagal insert:", repr(e))
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
