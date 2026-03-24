-- 001_initial.sql
-- Create assembled_pages table for svc-doc-assembly.
-- Owns: page images metadata, miss indicator auto_state.
-- Miss indicator override_state is written by svc-score-engine (separate column).

CREATE TABLE IF NOT EXISTS assembled_pages (
    page_id         TEXT        PRIMARY KEY,
    exam_id         UUID        NOT NULL,
    student_id      TEXT        NOT NULL,
    page_number     INTEGER     NOT NULL,
    s3_uri          TEXT        NOT NULL,

    -- Per-question miss indicators stored as JSONB array.
    -- Each element: {"question_id": str, "auto_state": str, "override_state": str|null}
    -- auto_state: written by svc-doc-assembly (this service)
    -- override_state: written by svc-score-engine (teacher action)
    question_results JSONB      NOT NULL DEFAULT '[]'::jsonb,

    page_width_mm   REAL        NOT NULL DEFAULT 210.0,
    page_height_mm  REAL        NOT NULL DEFAULT 297.0,

    assembled_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Natural uniqueness: one assembled page per exam+student+page
    CONSTRAINT uq_assembled_page UNIQUE (exam_id, student_id, page_number)
);

-- Index for querying all pages of a student's exam
CREATE INDEX IF NOT EXISTS idx_assembled_pages_exam_student
    ON assembled_pages (exam_id, student_id);

-- Index for querying pages needing miss review (non-answered auto_state)
CREATE INDEX IF NOT EXISTS idx_assembled_pages_miss_review
    ON assembled_pages USING gin (question_results);

-- RLS policy placeholder (tenant_id to be added when multi-tenant schema finalised)
-- ALTER TABLE assembled_pages ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation ON assembled_pages
--     USING (tenant_id = current_setting('app.current_tenant')::uuid);
