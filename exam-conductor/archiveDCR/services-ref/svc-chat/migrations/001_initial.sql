-- svc-chat: Initial schema — append-only chat messages and read receipts.
-- Owner: svc-chat (per STATE_OWNERSHIP_MAP.md)
--
-- CRITICAL: No UPDATE or DELETE is permitted on chat_messages.
-- DPDPA audit safety for minors' data requires full immutability.

-- =========================================================================
-- chat_messages — append-only message store
-- =========================================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id          UUID        PRIMARY KEY,
    sender_id   TEXT        NOT NULL,
    recipient_id TEXT       NOT NULL,
    exam_id     UUID        NOT NULL,
    content     TEXT        NOT NULL CHECK (char_length(content) <= 2000),
    attachment_uri TEXT,
    tenant_id   TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    -- NOTE: No updated_at column. Messages are immutable.
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_exam
    ON chat_messages (exam_id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
    ON chat_messages (exam_id, sender_id, recipient_id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_tenant
    ON chat_messages (tenant_id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_created
    ON chat_messages (created_at);

-- =========================================================================
-- read_receipts — append-only read tracking
-- =========================================================================

CREATE TABLE IF NOT EXISTS read_receipts (
    exam_id       UUID        NOT NULL,
    reader_id     TEXT        NOT NULL,
    other_user_id TEXT        NOT NULL,
    tenant_id     TEXT        NOT NULL,
    read_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (exam_id, reader_id, other_user_id)
);

CREATE INDEX IF NOT EXISTS idx_read_receipts_tenant
    ON read_receipts (tenant_id);

-- =========================================================================
-- APPEND-ONLY enforcement: block UPDATE and DELETE on chat_messages
-- =========================================================================

CREATE OR REPLACE FUNCTION prevent_chat_message_update()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'UPDATE on chat_messages is forbidden (append-only contract)';
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION prevent_chat_message_delete()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'DELETE on chat_messages is forbidden (append-only contract)';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_no_update_chat_messages ON chat_messages;
CREATE TRIGGER trg_no_update_chat_messages
    BEFORE UPDATE ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION prevent_chat_message_update();

DROP TRIGGER IF EXISTS trg_no_delete_chat_messages ON chat_messages;
CREATE TRIGGER trg_no_delete_chat_messages
    BEFORE DELETE ON chat_messages
    FOR EACH ROW
    EXECUTE FUNCTION prevent_chat_message_delete();

-- =========================================================================
-- Row-Level Security — tenant isolation
-- =========================================================================

ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE read_receipts ENABLE ROW LEVEL SECURITY;

-- Force RLS even for table owner
ALTER TABLE chat_messages FORCE ROW LEVEL SECURITY;
ALTER TABLE read_receipts FORCE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY tenant_isolation_chat_messages ON chat_messages
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

CREATE POLICY tenant_isolation_read_receipts ON read_receipts
    USING (tenant_id = current_setting('app.current_tenant', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant', true));

-- Bypass policies when tenant context is not set (migrations, super_admin)
CREATE POLICY bypass_when_no_tenant_chat_messages ON chat_messages
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);

CREATE POLICY bypass_when_no_tenant_read_receipts ON read_receipts
    USING (current_setting('app.current_tenant', true) IS NULL
           OR current_setting('app.current_tenant', true) = '')
    WITH CHECK (true);
