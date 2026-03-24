# Chapter 22: Security & Compliance

## Status
- **Phase:** Cross-cutting
- **Last updated:** 2026-03-20
- **Updated by:** Claude Agent (W6 stub)
- **Build status:** DRAFT

## Overview

Security architecture and DPDPA compliance. Covers multi-tenant isolation via
PostgreSQL RLS, JWT validation via Stoody JWKS, BLE security (LESC, rotating
auth codes), data encryption at rest and in transit, data minimization, consent
management, and the six open security findings from the W6 audit.

## Architecture Context

<!-- TODO: Security boundary diagram showing trust zones: pen, hub, server,
     client. Reference Chapter 01 and Chapter 05 (Auth and RBAC). -->

## Detailed Design

### Multi-Tenant Isolation
<!-- TODO: PostgreSQL RLS policies, tenant_id injection, CI enforcement. -->

### Authentication & Authorization
<!-- TODO: Stoody JWKS validation, role mapping, exam-specific roles. -->

### BLE Security
<!-- TODO: LESC, rotating auth codes, fallback risks. Reference S3 mitigation. -->

### Encryption
<!-- TODO: TLS everywhere (transit), PostgreSQL TDE (at rest), MinIO encryption. -->

### DPDPA Compliance
<!-- TODO: Data minimization, parent consent, retention policies, auto-delete. -->

### Open Security Findings (W6 Audit)
<!-- TODO: List 6 open findings from W6 security audit with remediation plan. -->

## Interfaces
<!-- TODO: Security-relevant API headers, CORS policies, CSP headers. -->

## Configuration
<!-- TODO: TLS certificates, JWKS endpoints, RLS policy files, retention config. -->

## Failure Modes & Mitigations
<!-- TODO: Reference FAILURE_MITIGATION_REGISTER.md entries A8.1 (multi-tenant
     leak), A8.2 (DPDPA violation), S3 (BLE MITM). -->

## Testing
<!-- TODO: Reference test IDs U-AUTH-01 through U-AUTH-05,
     I-AUTH-01 through I-AUTH-03. Penetration test checklist. -->

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-20 | Stub skeleton created | Claude Agent (W6) |
