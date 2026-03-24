-- svc-review: Initial schema — objections table.
-- Owner: svc-review (per STATE_OWNERSHIP_MAP.md)
-- FSM states: filed -> assigned -> reviewing -> resolved | escalated

CREATE TABLE IF NOT EXISTS objections (
    objection_id      UUID         PRIMARY KEY,
    tenant_id         TEXT         NOT NULL,
    exam_id           UUID         NOT NULL,
    student_id        TEXT         NOT NULL,
    question_id       TEXT         NOT NULL,
    objection_text    TEXT         NOT NULL,
    status            TEXT         NOT NULL DEFAULT 'filed'
                      CHECK (status IN ('filed', 'assigned', 'reviewing', 'resolved', 'escalated')),
    filed_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    assigned_to       TEXT,
    resolution        TEXT         CHECK (resolution IN ('approved', 'rejected')),
    resolution_reason TEXT,
    score_delta       NUMERIC,
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Enforce: one objection per student per question per exam.
CREATE UNIQUE INDEX IF NOT EXISTS uq_objections_student_question
    ON objections (student_id, exam_id, question_id);

-- Query patterns: filter by exam, filter by status.
CREATE INDEX IF NOT EXISTS idx_objections_exam
    ON objections (exam_id);

CREATE INDEX IF NOT EXISTS idx_objections_status
    ON objections (status);

CREATE INDEX IF NOT EXISTS idx_objections_tenant
    ON objections (tenant_id);

CREATE INDEX IF NOT EXISTS idx_objections_filed_at
    ON objections (filed_at DESC);

-- Enable RLS for multi-tenant isolation.
ALTER TABLE objections ENABLE ROW LEVEL SECURITY;

-- Tenant isolation: only rows matching the current tenant are visible/writable.
CREATE POLICY tenant_isolation_objections ON objections
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

-- Force RLS for the owning role as well.
ALTER TABLE objections FORCE ROW LEVEL SECURITY;

-- Superuser/migration bypass: when app.current_tenant is not set,
-- allow full access for migration scripts and super_admin operations.
CREATE POLICY bypass_when_no_tenant_objections ON objections
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

-- Auto-update updated_at on any row change.
CREATE OR REPLACE FUNCTION update_objections_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_objections_updated_at
    BEFORE UPDATE ON objections
    FOR EACH ROW
    EXECUTE FUNCTION update_objections_updated_at();
