# Stoody Book Absorption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Absorb the former contextual-learning app into the backend repo as `stoody-book`, serve a backend-hosted Stoody Book surface, and replace the Mentor ChatBot option without deleting the old debugger backend code.

**Architecture:** `backend/stoody-book` becomes a normal tracked backend directory, not a nested Git repo. FastAPI serves a simple Stoody Book web entrypoint at `/stoody-book` and static assets under `/stoody-book/assets`. The frontend Mentor tab launches this backend-hosted surface while MentorAI remains visible as Coming Soon.

**Tech Stack:** FastAPI `StaticFiles`/HTML route, Python pytest, React/Vite frontend, PowerShell verification commands.

---

### Task 1: Backend Mount Contract

**Files:**
- Create: `backend/tests/test_stoody_book_static.py`
- Create: `backend/api/v1/stoody_book_static.py`
- Modify: `backend/main_async.py`

- [ ] Write tests that assert the Stoody Book mount exposes `/stoody-book`, `/stoody-book/`, and static assets without importing the full backend app.
- [ ] Implement a small router/mount helper that serves `backend/stoody-book/web/index.html` and assets from `backend/stoody-book/web/assets`.
- [ ] Include the router/mount helper from `main_async.py`.
- [ ] Verify with `backend/venv/Scripts/python -m pytest tests/test_stoody_book_static.py -q`.

### Task 2: Absorb And Rename Former Contextual App

**Files:**
- Rename: former nested app folder to `backend/stoody-book`
- Remove: `backend/stoody-book/.git`
- Modify: `backend/stoody-book/package.json`
- Modify: `backend/stoody-book/package-lock.json`
- Modify: former app source/docs references where safe

- [ ] Move the directory to `backend/stoody-book`.
- [ ] Remove the nested `.git` directory after verifying the resolved path.
- [ ] Rename package identity to `stoody-book`.
- [ ] Rename source/docs/UI references to Stoody Book where they are not generated artifacts or compatibility protocol.
- [ ] Leave generated bundles and old ZIP release artifacts out of manual semantic edits unless rebuilt.

### Task 3: Backend-Hosted Web Surface

**Files:**
- Create: `backend/stoody-book/web/index.html`
- Create: `backend/stoody-book/web/assets/stoody-book.css`
- Create: `backend/stoody-book/web/assets/stoody-book.js`

- [ ] Build a usable web surface that explains Stoody Book as the contextual learning workspace and provides a working message composer.
- [ ] Use backend-owned endpoints where available or degrade clearly without pretending extension APIs are available.
- [ ] Avoid Chrome extension APIs in this web surface.

### Task 4: Frontend Mentor Replacement

**Files:**
- Modify: `frontend/src/pages/MentorLanding.tsx`
- Modify: `frontend/src/components/student/StudentLayout.tsx`
- Modify: `frontend/src/components/ProtectedRoute.tsx`
- Modify: `frontend/vite.config.ts`

- [ ] Replace the ChatBot card with Stoody Book and launch `/stoody-book` in dev or `https://api.stoody.in/stoody-book` in production.
- [ ] Make MentorAI card disabled with Coming Soon text and no navigation.
- [ ] Remove `/debugger` as a Mentor active-route alias.
- [ ] Gate `/mentor` with `student_ai_mentor` so the new Mentor surface follows the existing entitlement.
- [ ] Add a dev proxy for `/stoody-book`.

### Task 5: Deployment And Reference Audit

**Files:**
- Modify: `backend/.github/workflows/deploy-dev-backend.yml`
- Modify: `backend/.github/workflows/deploy-prod-backend.yml`
- Modify: `backend/core/tenant_features.py`
- Check: `backend/skillbot_nginx_alb.conf`

- [ ] Add `stoody-book/**` to backend deployment triggers.
- [ ] Add `/stoody-book` to backend tenant feature path mapping if the middleware sees the request before static serving.
- [ ] Keep `/mentor-ai` proxy/deploy behavior unchanged.
- [ ] Run scoped searches for legacy app names, `/debugger`, and `stoody-book` after edits.

### Task 6: Verification

- [ ] Run backend Stoody Book pytest.
- [ ] Run frontend build.
- [ ] Run backend import/compile check for changed Python files.
- [ ] Confirm no Mentor UI link points to `/debugger`.
- [ ] Confirm `/mentor-ai` remains present but the Mentor UI labels it Coming Soon.
