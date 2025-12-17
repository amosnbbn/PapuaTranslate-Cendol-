-- 01_schema_papua.sql
CREATE TABLE IF NOT EXISTS datasets (
  id           BIGSERIAL PRIMARY KEY,
  name         TEXT NOT NULL,
  description  TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_items (
  id           BIGSERIAL PRIMARY KEY,
  dataset_id   BIGINT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  split        TEXT NOT NULL,            -- train / validation / test
  instruction  TEXT,
  input        TEXT NOT NULL,
  output       TEXT NOT NULL,
  raw          JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_items_dataset ON dataset_items(dataset_id);
CREATE INDEX IF NOT EXISTS idx_items_split   ON dataset_items(split);

CREATE TABLE IF NOT EXISTS models (
  id           BIGSERIAL PRIMARY KEY,
  name         TEXT NOT NULL,            -- misal: "phase8_ckpt168"
  base_dir     TEXT NOT NULL,
  adapter_path TEXT NOT NULL,
  sha256       TEXT NOT NULL,
  bleu         DOUBLE PRECISION,
  chrf         DOUBLE PRECISION,
  notes        TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_blobs (
  id           BIGSERIAL PRIMARY KEY,
  model_id     BIGINT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  filename     TEXT NOT NULL,
  content      BYTEA NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
