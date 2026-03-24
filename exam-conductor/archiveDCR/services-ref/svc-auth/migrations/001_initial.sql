-- svc-auth: Initial schema — revocations and role_mappings tables.
-- Owner: svc-auth (per STATE_OWNERSHIP_MAP.md)

CREATE TABLE IF NOT EXISTS revocations (
    jti             TEXT        PRIMARY KEY,
    tenant_id       TEXT        NOT NULL,
    subject_user_id TEXT,
    revoked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason          TEXT        NOT NULL,
    revoked_by      TEXT        NOT NULL,
    expires_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_revocations_tenant
    ON revocations (tenant_id);

CREATE INDEX IF NOT EXISTS idx_revocations_subject
    ON revocations (subject_user_id);

CREATE INDEX IF NOT EXISTS idx_revocations_expires
    ON revocations (expires_at)
    WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS role_mappings (
    stoody_role   TEXT        NOT NULL,
    tenant_id     TEXT        NOT NULL,
    exampen_roles TEXT[]      NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stoody_role, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_role_mappings_tenant
    ON role_mappings (tenant_id);

-- Seed default role mappings (global / empty tenant placeholder).
-- Tenants without overrides fall through to these defaults.
INSERT INTO role_mappings (stoody_role, tenant_id, exampen_roles) VALUES
    ('admin',       '',  ARRAY['principal']),
    ('super_admin', '',  ARRAY['super_admin']),
    ('principal',   '',  ARRAY['principal']),
    ('hod',         '',  ARRAY['hod']),
    ('tutor',       '',  ARRAY['evaluator']),
    ('student',     '',  ARRAY['student']),
    ('parent',      '',  ARRAY['parent'])
ON CONFLICT (stoody_role, tenant_id) DO NOTHING;

-- Enable RLS on both tables.
ALTER TABLE revocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE role_mappings ENABLE ROW LEVEL SECURITY;

-- RLS policies: tenant isolation via app.current_tenant session variable.
-- current_setting(..., true) returns NULL instead of raising an error
-- when the variable is unset (e.g. during migrations or super_admin ops).

-- revocations: only rows matching the current tenant are visible/writable.
CREATE POLICY tenant_isolation_revocations ON revocations
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

-- role_mappings: tenant-specific overrides plus global defaults (tenant_id='').
CREATE POLICY tenant_isolation_role_mappings ON role_mappings
    USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR tenant_id = ''
    )
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

-- Allow the service user to bypass RLS for administrative operations
-- (migrations, health checks, super_admin cross-tenant queries).
-- The service role is the owner of these tables.
ALTER TABLE revocations FORCE ROW LEVEL SECURITY;
ALTER TABLE role_mappings FORCE ROW LEVEL SECURITY;

-- Superuser/owner bypass policy: when app.current_tenant is not set
-- (NULL/empty), allow full access.  This supports migration scripts
-- and super_admin operations.
CREATE POLICY bypass_when_no_tenant_revocations ON revocations
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

CREATE POLICY bypass_when_no_tenant_role_mappings ON role_mappings
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);
