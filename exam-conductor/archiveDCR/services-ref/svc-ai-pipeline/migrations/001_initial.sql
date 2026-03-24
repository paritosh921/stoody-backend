-- svc-ai-pipeline: Initial schema for AI recognition results.
--
-- Key constraints (from STATE_OWNERSHIP_MAP):
--   - model_version stored with every result
--   - Re-running AI creates a new row; old rows are never overwritten
--   - Only svc-ai-pipeline writes to this table

CREATE TABLE IF NOT EXISTS ai_results (
    id              BIGSERIAL       PRIMARY KEY,
    event_id        TEXT            NOT NULL UNIQUE,
    exam_id         UUID            NOT NULL,
    student_id      TEXT            NOT NULL,
    model_version   TEXT            NOT NULL,
    source_type     TEXT            NOT NULL DEFAULT 'strokes'
                    CHECK (source_type IN ('strokes', 'copy_image')),
    question_results JSONB          NOT NULL DEFAULT '[]'::jsonb,
    occurred_at     TIMESTAMPTZ     NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Index for the most common query: latest results for a student in an exam.
CREATE INDEX idx_ai_results_exam_student
    ON ai_results (exam_id, student_id, occurred_at DESC);

-- Index for re-run queries: find all versions for a given exam+student.
CREATE INDEX idx_ai_results_model_version
    ON ai_results (exam_id, student_id, model_version);

-- RLS policy placeholder (tenant_id to be added when multi-tenant RLS is wired).
-- ALTER TABLE ai_results ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY ai_results_tenant ON ai_results
--     USING (tenant_id = current_setting('app.current_tenant')::uuid);
