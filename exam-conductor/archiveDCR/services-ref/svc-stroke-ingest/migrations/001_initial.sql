-- svc-stroke-ingest: initial schema
-- Tables: idempotency_keys (backup/audit only; hot path uses Redis),
--         upload_progress (per-pen per-exam chunk tracking)

-- Idempotency keys table (backup; Redis is the hot-path store)
-- Retained for audit trail and disaster recovery if Redis is flushed.
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key         TEXT        PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL DEFAULT now() + INTERVAL '7 days'
);

CREATE INDEX IF NOT EXISTS idx_idem_expires
    ON idempotency_keys (expires_at);

-- Upload progress: one row per acknowledged chunk
CREATE TABLE IF NOT EXISTS upload_progress (
    exam_id      UUID        NOT NULL,
    pen_mac      TEXT        NOT NULL,
    chunk_index  INTEGER     NOT NULL,
    total_chunks INTEGER     NOT NULL,
    received_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (exam_id, pen_mac, chunk_index)
);

-- Index for reconciliation queries (per exam)
CREATE INDEX IF NOT EXISTS idx_upload_progress_exam
    ON upload_progress (exam_id);

-- RLS policy (tenant isolation enforced at application layer via
-- svc-auth JWT; stroke-ingest does not use RLS directly because
-- the upload_progress table is scoped by exam_id, not tenant_id.
-- Tenant-scoping is enforced upstream by svc-exam-orch.)
