-- svc-plagiarism: initial schema for plagiarism flags and teacher verdicts.
-- Owner: svc-plagiarism (single writable owner per STATE_OWNERSHIP_MAP).

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE plagiarism_flags (
    flag_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exam_id          UUID        NOT NULL,
    student_a_id     TEXT        NOT NULL,
    student_b_id     TEXT        NOT NULL,
    question_id      TEXT        NOT NULL,

    -- Composite score dimensions
    text_sim         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    structural_sim   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    temporal_corr    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    proximity_score  DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    composite_score  DOUBLE PRECISION NOT NULL,

    -- Severity: review_recommended | strong_match
    severity         TEXT        NOT NULL
        CHECK (severity IN ('review_recommended', 'strong_match')),

    -- Evidence: raw answer texts for teacher review
    student_a_text   TEXT        NOT NULL DEFAULT '',
    student_b_text   TEXT        NOT NULL DEFAULT '',

    -- Teacher verdict columns (svc-plagiarism owns both flags AND verdicts)
    teacher_verdict  TEXT        NOT NULL DEFAULT 'pending'
        CHECK (teacher_verdict IN ('pending', 'confirmed_plagiarism', 'dismissed')),
    verdict_reason   TEXT,
    verdict_by       TEXT,
    verdict_at       TIMESTAMPTZ,

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Prevent duplicate flags for the same student pair + question
    CONSTRAINT uq_pair_question UNIQUE (exam_id, student_a_id, student_b_id, question_id)
);

-- Query patterns: list by exam, lookup by flag_id (PK), filter by severity
CREATE INDEX idx_flags_exam_id     ON plagiarism_flags (exam_id);
CREATE INDEX idx_flags_severity    ON plagiarism_flags (exam_id, severity);
CREATE INDEX idx_flags_verdict     ON plagiarism_flags (exam_id, teacher_verdict);

-- Row-Level Security (multi-tenant)
ALTER TABLE plagiarism_flags ENABLE ROW LEVEL SECURITY;

-- RLS policy: service role has full access; tenant-scoped access
-- enforced at the application layer via exam_id ownership checks.
CREATE POLICY plagiarism_flags_service_policy
    ON plagiarism_flags
    FOR ALL
    USING (true);
