-- =============================================================
-- ExamPen — PostgreSQL Init Script
-- Creates per-service databases, user, RLS, and extensions.
-- Run automatically on first start via docker-entrypoint-initdb.d
-- =============================================================

-- Create the shared application user
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'exampen') THEN
    CREATE ROLE exampen WITH LOGIN PASSWORD 'exampen_dev';
  END IF;
END
$$;

-- Per-service databases
CREATE DATABASE exampen_auth       OWNER exampen;
CREATE DATABASE exampen_exam       OWNER exampen;
CREATE DATABASE exampen_stroke     OWNER exampen;
CREATE DATABASE exampen_score      OWNER exampen;
CREATE DATABASE exampen_review     OWNER exampen;
CREATE DATABASE exampen_analytics  OWNER exampen;
CREATE DATABASE exampen_plagiarism OWNER exampen;
CREATE DATABASE exampen_chat       OWNER exampen;
CREATE DATABASE exampen_copy       OWNER exampen;
CREATE DATABASE exampen_notify     OWNER exampen;

-- Enable TimescaleDB on stroke database
\c exampen_stroke
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Helper: enable RLS on all future tables in a schema.
-- Each service migration will CREATE TABLE ... ; ALTER TABLE ... ENABLE ROW LEVEL SECURITY;
-- This script only ensures the extension is available.

-- Enable RLS helper function in every service database
\c exampen_auth
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_exam
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_stroke
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_score
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_review
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_analytics
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_plagiarism
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_chat
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_copy
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;

\c exampen_notify
ALTER DEFAULT PRIVILEGES FOR ROLE exampen GRANT ALL ON TABLES TO exampen;
