# DOCUMENTATION_PLAN.md
# ExamPen — Living Documentation Structure

Reference: R4-EXAMPEN-DEVSTACK

---

## Principle

Documentation updates are not optional. Every build phase produces documentation as a deliverable alongside code. A component is not "done" until its documentation chapter is written.

Path note: all examples and references in this file are relative to this documentation root (`new-docs/` while drafting, `docs/` after promotion).

---

## 1. Chapter Structure

```
chapters/
├── 00_INDEX.md                    # Master index with chapter status
├── 01_SYSTEM_OVERVIEW.md          # Architecture overview, component map
├── 02_HUB_HARDWARE.md             # BOM, assembly, first-boot, golden image
├── 03_HUB_SOFTWARE.md             # Hub modules, TUI, local storage, systemd
├── 04_BLE_PROTOCOL.md             # GATT services (pen + invigilator), wire format
├── 05_PEN_FIRMWARE.md             # Pen MCU, stroke capture, offline buffer, BLE stack
├── 06_EXAM_LIFECYCLE.md           # Exam orchestration, state machines, timer logic
├── 07_STROKE_PIPELINE.md          # Ingestion → processing → doc assembly
├── 08_AI_PIPELINE.md              # HWR/OCR, step detection, model registry
├── 09_SCORING_ENGINE.md           # Rubric eval, score FSM, event sourcing, overrides
├── 10_PLAGIARISM_DETECTION.md     # Detection layers, scoring, review workflow
├── 11_MISS_INDICATORS.md          # Detection pipeline, states, teacher override
├── 12_REVIEW_OBJECTIONS.md        # Objection lifecycle, assignment, resolution
├── 13_ANALYTICS_LEADERBOARD.md    # Percentile calc, leaderboard, export
├── 14_MOBILE_APP.md               # Invigilator mode, teacher mode, camera, chat
├── 15_TEACHER_DASHBOARD.md        # Web dashboard features, screens, workflows
├── 16_STUDENT_PORTAL.md           # Score view, objection, chat
├── 17_AUTH_RBAC.md                # Roles, permissions, tenant isolation, JWT
├── 18_STOODY_INTEGRATION.md       # Platform integration, SSO, API mapping
├── 19_NOTIFICATIONS.md            # Email, push, SMS triggers and templates
├── 20_INFRASTRUCTURE.md           # Docker Compose, Traefik, monitoring, backup
├── 21_TESTING.md                  # Test strategy, TUI runner, CI pipeline
├── 22_DEPLOYMENT.md               # Hub fleet deployment, server deployment, upgrades
├── 23_SECURITY_COMPLIANCE.md      # DPDPA, data residency, audit trail, encryption
├── 24_API_REFERENCE.md            # Links to OpenAPI specs per service
├── 25_TROUBLESHOOTING.md          # Common issues, diagnostic steps, log analysis
└── BUILD_STATUS.md                # Per-component build status (agent handoff)
```

## 2. Chapter Template

Every chapter follows this structure:

```markdown
# Chapter NN: {Title}

## Status
- **Phase:** P{N} — {phase name}
- **Last updated:** {date}
- **Updated by:** {agent/person}
- **Build status:** DRAFT / IN_PROGRESS / COMPLETE / NEEDS_REVIEW

## Overview
{One paragraph: what this chapter covers and why it matters.}

## Architecture Context
{Where this component sits in the overall system. Reference to Chapter 01.}

## Detailed Design
{Technical specification. Tables, schemas, state machines, protocols.}

## Interfaces
{What this component exposes and consumes. API endpoints, events, BLE chars.}

## Configuration
{Environment variables, config files, feature flags.}

## Failure Modes & Mitigations
{From FAILURE_MITIGATION_REGISTER.md, filtered to this component.}

## Testing
{From TEST_SUITE_SPEC.md, filtered to this component. Specific test IDs.}

## Changelog
| Date | Change | By |
|---|---|---|
| | | |
```

## 3. Update Triggers

| Build Phase | Chapters to Update |
|---|---|
| P0 (Auth) | 17_AUTH_RBAC, 01_SYSTEM_OVERVIEW |
| P1 (Exam orch) | 06_EXAM_LIFECYCLE, 01_SYSTEM_OVERVIEW |
| P2a–P2i (Hub) | 02_HUB_HARDWARE, 03_HUB_SOFTWARE, 04_BLE_PROTOCOL |
| P3 (Stroke proc) | 07_STROKE_PIPELINE |
| P5 (Doc assembly) | 07_STROKE_PIPELINE, 11_MISS_INDICATORS |
| P6–P7 (AI) | 08_AI_PIPELINE |
| P8 (Score engine) | 09_SCORING_ENGINE |
| P8b (Plagiarism) | 10_PLAGIARISM_DETECTION |
| P9 (Teacher dash) | 15_TEACHER_DASHBOARD, 14_MOBILE_APP |
| P10 (Student portal) | 16_STUDENT_PORTAL |
| P11 (Review) | 12_REVIEW_OBJECTIONS |
| P12 (Analytics) | 13_ANALYTICS_LEADERBOARD |
| Stoody integration | 18_STOODY_INTEGRATION |
| Any infra change | 20_INFRASTRUCTURE |
| Any test addition | 21_TESTING |

## 4. BUILD_STATUS.md Format

This file is the multi-agent coordination surface.

```markdown
# Build Status

Last updated: 2026-03-18

| Component | Phase | Status | Agent | Contract | Mock | Known Issues |
|---|---|---|---|---|---|---|
| svc-auth | P0 | COMPLETE | Agent-1 | api/auth.openapi.yaml | Yes | None |
| svc-exam-orch | P1 | IN_PROGRESS | Agent-1 | api/exam-orch.openapi.yaml | Partial | Timer edge case untested |
| hub-store | P2a | COMPLETE | Agent-2 | hub/hub-common/ipc_protocol.py | N/A | USB hot-plug not tested |
| hub-ble-mgr | P2b | NOT_STARTED | — | — | — | — |
```

## 5. Documentation Quality Gates

A chapter is **COMPLETE** only when:

- [ ] All sections of the template are filled (no TBD/TODO placeholders)
- [ ] Interfaces section matches actual OpenAPI/NATS schema files
- [ ] Failure modes section references `FAILURE_MITIGATION_REGISTER.md` entries by ID (e.g., "A1.5", "H3")
- [ ] Testing section references specific test IDs from `TEST_SUITE_SPEC.md` using the enumerated IDs (e.g., "Covered by: U-SCR-01, U-SCR-02, I-SCR-01, E2E-03"). Categories are not sufficient — explicit IDs required.
- [ ] Changelog has at least one entry
- [ ] Reviewed by at least one other agent/person

**Example — Chapter 09 (Scoring Engine) testing section:**
```
## Testing
- Unit: U-SCR-01 (FSM valid), U-SCR-02 (FSM invalid), U-SCR-03 (rubric eval), U-SCR-04 (override)
- Integration: I-SCR-01 (AI result→score), I-SCR-02 (override REST), I-SCR-03 (NATS event publish)
- E2E: E2E-03 (AI→score), E2E-04 (override→analytics)
- Not covered: Manual bulk-approve UX flow (deferred to P9 teacher dashboard)
```

---

## Changelog

| Date | Change | By |
|---|---|---|
| 2026-03-18 | Normalized example contract paths to doc-root-relative references and clarified that this file is rooted at `new-docs/`. | Codex |
