-- svc-exam-orch initial schema
-- Creates: exams, exam_questions, rubrics, question_regions,
--          pen_bindings, assignments

BEGIN;

-- Exams table — core exam definition + FSM state
CREATE TABLE IF NOT EXISTS exams (
    exam_id       UUID PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    title         TEXT NOT NULL,
    subject_id    TEXT NOT NULL,
    class_id      TEXT NOT NULL,
    section_id    TEXT NOT NULL,
    scheduled_at  TIMESTAMPTZ NOT NULL,
    duration_min  INTEGER NOT NULL CHECK (duration_min > 0),
    question_count INTEGER NOT NULL CHECK (question_count > 0),
    total_marks   NUMERIC(8,2) NOT NULL CHECK (total_marks > 0),
    negative_marking BOOLEAN NOT NULL DEFAULT FALSE,
    variants      TEXT[] NOT NULL DEFAULT '{}',
    state         TEXT NOT NULL DEFAULT 'created'
                  CHECK (state IN (
                      'created','armed','timer_running','sync_pending',
                      'scoring','finalized','published','locked','cancelled'
                  )),
    late_entry_cutoff_min INTEGER,
    objection_window_days INTEGER,
    rubric            JSONB NOT NULL DEFAULT '{"questions":[],"confidence_threshold":0.85}',
    question_regions  JSONB NOT NULL DEFAULT '[]',
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_exams_tenant ON exams (tenant_id);
CREATE INDEX idx_exams_state  ON exams (state);
CREATE INDEX idx_exams_scheduled ON exams (scheduled_at);

-- Row-level security
ALTER TABLE exams ENABLE ROW LEVEL SECURITY;
CREATE POLICY exams_tenant_isolation ON exams
    USING (tenant_id = current_setting('app.current_tenant'));

-- Rubrics — per-question marks breakdown
CREATE TABLE IF NOT EXISTS rubrics (
    rubric_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id       UUID NOT NULL REFERENCES exams(exam_id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL,
    question_number INTEGER NOT NULL CHECK (question_number > 0),
    max_marks     NUMERIC(6,2) NOT NULL CHECK (max_marks > 0),
    step_breakdown NUMERIC(6,2)[] NOT NULL DEFAULT '{}',
    expected_answer_type TEXT NOT NULL DEFAULT 'text'
                  CHECK (expected_answer_type IN ('text','formula','diagram','mixed')),
    confidence_threshold NUMERIC(3,2) NOT NULL DEFAULT 0.70
                  CHECK (confidence_threshold BETWEEN 0 AND 1),
    UNIQUE (exam_id, question_number)
);

ALTER TABLE rubrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY rubrics_tenant_isolation ON rubrics
    USING (tenant_id = current_setting('app.current_tenant'));

-- Question regions — bounding boxes on answer sheet
CREATE TABLE IF NOT EXISTS question_regions (
    region_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id       UUID NOT NULL REFERENCES exams(exam_id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL,
    question_number INTEGER NOT NULL CHECK (question_number > 0),
    x             NUMERIC(10,2) NOT NULL CHECK (x >= 0),
    y             NUMERIC(10,2) NOT NULL CHECK (y >= 0),
    width         NUMERIC(10,2) NOT NULL CHECK (width > 0),
    height        NUMERIC(10,2) NOT NULL CHECK (height > 0),
    page          INTEGER NOT NULL DEFAULT 1 CHECK (page > 0),
    UNIQUE (exam_id, question_number)
);

ALTER TABLE question_regions ENABLE ROW LEVEL SECURITY;
CREATE POLICY regions_tenant_isolation ON question_regions
    USING (tenant_id = current_setting('app.current_tenant'));

-- Pen bindings — pen MAC <-> student mapping
CREATE TABLE IF NOT EXISTS pen_bindings (
    exam_id       UUID NOT NULL REFERENCES exams(exam_id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL,
    pen_mac       TEXT NOT NULL,
    student_id    TEXT NOT NULL,
    student_name  TEXT,
    student_roll  TEXT,
    status        TEXT NOT NULL DEFAULT 'provisional'
                  CHECK (status IN ('provisional','confirmed','rejected')),
    source        TEXT NOT NULL
                  CHECK (source IN ('registration_scan','manual_register','server_sync')),
    bound_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    server_confirmed_at TIMESTAMPTZ,
    rejection_reason TEXT,
    PRIMARY KEY (exam_id, pen_mac)
);

CREATE INDEX idx_bindings_student ON pen_bindings (exam_id, student_id);

ALTER TABLE pen_bindings ENABLE ROW LEVEL SECURITY;
CREATE POLICY bindings_tenant_isolation ON pen_bindings
    USING (tenant_id = current_setting('app.current_tenant'));

-- Assignments — invigilators and evaluators
CREATE TABLE IF NOT EXISTS assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_id       UUID NOT NULL REFERENCES exams(exam_id) ON DELETE CASCADE,
    tenant_id     TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('invigilator','evaluator')),
    assigned_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (exam_id, user_id, role)
);

ALTER TABLE assignments ENABLE ROW LEVEL SECURITY;
CREATE POLICY assignments_tenant_isolation ON assignments
    USING (tenant_id = current_setting('app.current_tenant'));

COMMIT;
