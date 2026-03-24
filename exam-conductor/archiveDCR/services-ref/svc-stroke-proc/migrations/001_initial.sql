-- svc-stroke-proc: initial schema — TimescaleDB hypertable for processed strokes
-- Owner: svc-stroke-proc (STATE_OWNERSHIP_MAP: "Server strokes")
--
-- Requires: TimescaleDB extension enabled on the database.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Processed strokes table
-- One row per stroke per chunk.  All strokes from a single chunk share
-- the same idempotency_key, enabling atomic dedup.
CREATE TABLE IF NOT EXISTS processed_strokes (
    id               BIGSERIAL       NOT NULL,
    exam_id          UUID            NOT NULL,
    pen_mac          TEXT            NOT NULL,
    chunk_index      INTEGER         NOT NULL,
    idempotency_key  TEXT            NOT NULL,
    stroke_id        TEXT            NOT NULL,
    page_number      INTEGER         NOT NULL DEFAULT 0,
    question_id      TEXT,
    normalized_points JSONB          NOT NULL DEFAULT '[]'::jsonb,
    book_type        TEXT            NOT NULL DEFAULT 'LS',
    created_at       TIMESTAMPTZ     NOT NULL DEFAULT now(),

    PRIMARY KEY (id, created_at)
);

-- Convert to TimescaleDB hypertable, partitioned by created_at
SELECT create_hypertable(
    'processed_strokes',
    'created_at',
    if_not_exists => TRUE
);

-- Per-stroke uniqueness: a chunk contains multiple strokes that share
-- the same idempotency_key.  The unique constraint is therefore on the
-- (idempotency_key, stroke_id) pair so every stroke inserts, while a
-- re-delivered chunk (same key + same stroke_ids) is safely rejected.
CREATE UNIQUE INDEX IF NOT EXISTS idx_processed_strokes_idem
    ON processed_strokes (idempotency_key, stroke_id);

-- Query index: per-exam per-pen lookups for downstream consumers
CREATE INDEX IF NOT EXISTS idx_processed_strokes_exam_pen
    ON processed_strokes (exam_id, pen_mac, created_at DESC);

-- Query index: per-exam per-page for doc-assembly
CREATE INDEX IF NOT EXISTS idx_processed_strokes_exam_page
    ON processed_strokes (exam_id, page_number, created_at DESC);
