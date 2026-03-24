-- 001_initial.sql
-- svc-score-engine schema: event store, materialised view, rubrics.
-- Owner: svc-score-engine (sole writer).

BEGIN;

-- =============================================================================
-- 1. Score events — append-only event store
-- =============================================================================
CREATE TABLE IF NOT EXISTS score_events (
    event_id        TEXT        PRIMARY KEY,
    exam_id         UUID        NOT NULL,
    student_id      TEXT        NOT NULL,
    question_id     TEXT,                       -- NULL for exam-level lifecycle events
    event_type      TEXT        NOT NULL,       -- ai_draft_created | override_applied | finalized | published | objection_rescored
    old_value       NUMERIC,                    -- NULL on first draft
    new_value       NUMERIC     NOT NULL,
    actor_id        TEXT        NOT NULL,       -- 'ai_pipeline' | teacher user id
    reason          TEXT        NOT NULL,
    metadata        TEXT,                       -- JSON blob (step_scores, rubric_version, confidence)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Query patterns: by exam+student (student detail), by exam (class overview).
CREATE INDEX idx_score_events_exam_student
    ON score_events (exam_id, student_id, created_at);

CREATE INDEX idx_score_events_exam
    ON score_events (exam_id, created_at);

-- Append-only enforcement: no UPDATE or DELETE via RLS.
-- (Application-level; RLS policy added in tenant migration.)

-- =============================================================================
-- 2. Materialised view — fast read projection of current scores
-- =============================================================================
CREATE TABLE IF NOT EXISTS score_materialized (
    exam_id         UUID        NOT NULL,
    student_id      TEXT        NOT NULL,
    question_id     TEXT        NOT NULL,       -- '__exam__' for lifecycle-level row
    current_score   NUMERIC     NOT NULL DEFAULT 0,
    lifecycle_state TEXT        NOT NULL DEFAULT 'ai_draft',
    rubric_version  INTEGER,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exam_id, student_id, question_id)
);

CREATE INDEX idx_score_mat_exam
    ON score_materialized (exam_id);

-- =============================================================================
-- 3. Rubrics — versioned marking schemes
-- =============================================================================
CREATE TABLE IF NOT EXISTS rubrics (
    question_id     TEXT        NOT NULL,
    version         INTEGER     NOT NULL GENERATED ALWAYS AS IDENTITY,
    body            TEXT        NOT NULL,       -- JSON: {steps: [{label, max_marks, keywords}], negative_marking, negative_factor}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (question_id, version)
);

CREATE INDEX idx_rubrics_question
    ON rubrics (question_id, version DESC);

COMMIT;
