-- svc-copy-upload: initial schema
-- Copy images are fallback data — never overwrite stroke-derived page images.

CREATE TABLE IF NOT EXISTS copy_images (
    exam_id       UUID        NOT NULL,
    student_id    TEXT        NOT NULL,
    page_number   INTEGER     NOT NULL CHECK (page_number >= 1),
    tenant_id     TEXT        NOT NULL,
    s3_path       TEXT        NOT NULL,
    content_type  TEXT        NOT NULL,
    file_size     BIGINT      NOT NULL CHECK (file_size > 0),
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    uploaded_by   TEXT        NOT NULL,

    PRIMARY KEY (exam_id, student_id, page_number)
);

CREATE INDEX IF NOT EXISTS idx_copy_images_exam
    ON copy_images (exam_id);

CREATE INDEX IF NOT EXISTS idx_copy_images_exam_student
    ON copy_images (exam_id, student_id);

CREATE INDEX IF NOT EXISTS idx_copy_images_tenant
    ON copy_images (tenant_id);

-- Enable RLS for tenant isolation.
ALTER TABLE copy_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE copy_images FORCE ROW LEVEL SECURITY;

-- Tenant isolation: only rows matching the current tenant are visible.
CREATE POLICY tenant_isolation_copy_images ON copy_images
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

-- Bypass policy for migrations and super_admin operations
-- (when app.current_tenant is not set).
CREATE POLICY bypass_when_no_tenant_copy_images ON copy_images
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

COMMENT ON TABLE copy_images IS
    'Fallback photographed answer pages. S3 write first, PG second.';
