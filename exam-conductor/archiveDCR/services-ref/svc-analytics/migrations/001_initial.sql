-- svc-analytics: Initial schema — percentiles, leaderboard cache, score cache.
-- Owner: svc-analytics (per STATE_OWNERSHIP_MAP.md)
--
-- svc-analytics is the ONLY writer of percentile data.
-- No other service computes or writes percentiles.

-- =========================================================================
-- exam_score_cache — local cache of scores received via score.updated events
-- =========================================================================

CREATE TABLE IF NOT EXISTS exam_score_cache (
    exam_id      UUID        NOT NULL,
    student_id   TEXT        NOT NULL,
    student_name TEXT        NOT NULL DEFAULT '',
    total_score  NUMERIC     NOT NULL,
    tenant_id    TEXT        NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exam_id, student_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_score_cache_exam
    ON exam_score_cache (exam_id);

CREATE INDEX IF NOT EXISTS idx_exam_score_cache_student
    ON exam_score_cache (student_id);

CREATE INDEX IF NOT EXISTS idx_exam_score_cache_tenant
    ON exam_score_cache (tenant_id);

-- =========================================================================
-- exam_percentiles — materialized percentile per student per exam
-- =========================================================================

CREATE TABLE IF NOT EXISTS exam_percentiles (
    id           UUID        PRIMARY KEY,
    exam_id      UUID        NOT NULL,
    student_id   TEXT        NOT NULL,
    percentile   NUMERIC     NOT NULL CHECK (percentile >= 0 AND percentile <= 100),
    tenant_id    TEXT        NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_exam_percentiles_exam_student
    ON exam_percentiles (exam_id, student_id);

CREATE INDEX IF NOT EXISTS idx_exam_percentiles_student
    ON exam_percentiles (student_id);

CREATE INDEX IF NOT EXISTS idx_exam_percentiles_tenant
    ON exam_percentiles (tenant_id);

-- =========================================================================
-- leaderboard_cache — materialized leaderboard rows per exam
-- =========================================================================

CREATE TABLE IF NOT EXISTS leaderboard_cache (
    id           UUID        PRIMARY KEY,
    exam_id      UUID        NOT NULL,
    student_id   TEXT        NOT NULL,
    student_name TEXT        NOT NULL DEFAULT '',
    rank         INTEGER     NOT NULL,
    score        NUMERIC     NOT NULL,
    percentile   NUMERIC     NOT NULL,
    tenant_id    TEXT        NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_leaderboard_cache_exam_student
    ON leaderboard_cache (exam_id, student_id);

CREATE INDEX IF NOT EXISTS idx_leaderboard_cache_exam_rank
    ON leaderboard_cache (exam_id, rank);

CREATE INDEX IF NOT EXISTS idx_leaderboard_cache_tenant
    ON leaderboard_cache (tenant_id);

-- =========================================================================
-- question_response_cache — per-question per-student response data
-- =========================================================================

CREATE TABLE IF NOT EXISTS question_response_cache (
    exam_id      UUID        NOT NULL,
    student_id   TEXT        NOT NULL,
    question_id  TEXT        NOT NULL,
    score        NUMERIC     NOT NULL DEFAULT 0,
    max_score    NUMERIC     NOT NULL DEFAULT 0,
    attempted    BOOLEAN     NOT NULL DEFAULT false,
    tenant_id    TEXT        NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exam_id, student_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_question_response_cache_exam
    ON question_response_cache (exam_id);

CREATE INDEX IF NOT EXISTS idx_question_response_cache_tenant
    ON question_response_cache (tenant_id);

-- =========================================================================
-- Row-Level Security — tenant isolation
-- =========================================================================

ALTER TABLE exam_score_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE exam_percentiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE leaderboard_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE question_response_cache ENABLE ROW LEVEL SECURITY;

ALTER TABLE exam_score_cache FORCE ROW LEVEL SECURITY;
ALTER TABLE exam_percentiles FORCE ROW LEVEL SECURITY;
ALTER TABLE leaderboard_cache FORCE ROW LEVEL SECURITY;
ALTER TABLE question_response_cache FORCE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY tenant_isolation_exam_score_cache ON exam_score_cache
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

CREATE POLICY tenant_isolation_exam_percentiles ON exam_percentiles
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

CREATE POLICY tenant_isolation_leaderboard_cache ON leaderboard_cache
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

CREATE POLICY tenant_isolation_question_response_cache ON question_response_cache
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

-- Bypass policies when tenant context is not set (migrations, super_admin)
CREATE POLICY bypass_when_no_tenant_exam_score_cache ON exam_score_cache
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

CREATE POLICY bypass_when_no_tenant_exam_percentiles ON exam_percentiles
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

CREATE POLICY bypass_when_no_tenant_leaderboard_cache ON leaderboard_cache
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

CREATE POLICY bypass_when_no_tenant_question_response_cache ON question_response_cache
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);
