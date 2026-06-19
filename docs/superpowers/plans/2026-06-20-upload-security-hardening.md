# Upload Security Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every backend file upload and hub data upload through centralized size limits, type verification, malware scanning or structured-data validation, private storage, audit verdicts, and authenticated release paths.

**Architecture:** `backend/core/upload_security/policies.py` becomes the single source of truth for upload policies, including default file size limits, allowed extensions, allowed MIME types, magic-byte families, page/image/archive/row/chunk limits, and environment override names. Binary files go through a quarantine and clean-release service before they are saved, parsed, OCR'd, or exposed. Hub JSON and stroke chunk uploads use the same policy registry for body, count, and schema limits, but skip malware scanning because they are structured data, not files.

**Tech Stack:** FastAPI, Pydantic v2, Motor/MongoDB, S3/local private storage, optional ClamAV `clamd`, nginx, pytest, mongomock-motor, pandas/openpyxl/pypdf/Pillow for bounded parsers.

---

## Design Summary

### Problem

The current checkout accepts and processes several file and data upload classes with route-local limits, client-controlled MIME trust, inconsistent magic-byte checks, no malware verdict, and local fallback under `backend/uploads`, which is mounted publicly at `/uploads`.

### Success Criteria

- Every upload route calls a shared backend policy service before saving, parsing, OCR, AI processing, or Mongo insertion.
- Developers control per-flow size limits from `backend/core/upload_security/policies.py`; deployed overrides use predictable environment variables derived from those policy IDs.
- Binary uploads have a stored verdict: `pending`, `clean`, `rejected`, or `scan_failed`, with SHA-256, policy ID, scanner version, detected type, size, and storage path.
- ClamAV/clamd scanner errors fail closed in production.
- Untrusted uploads are never directly served from `/uploads`.
- S3 failures no longer silently downgrade production uploads to a public local path.
- Structured hub JSON and stroke uploads are bounded by request body, session, frame, chunk, decoded-byte, and schema limits before durable writes.

### Out Of Scope

- Rewriting OCR/PDF extraction logic.
- Changing product workflows or permissions.
- Scanning already-uploaded historical objects during the main route refactor. That is a separate migration task in this plan.

### Critical State Ownership

| State | Writable Owner | Readers / Derivers | Notes |
|---|---|---|---|
| Upload policy defaults and limit names | `backend/core/upload_security/policies.py` | Routes, middleware, frontend policy endpoint, tests | No route-local max-size constants after migration. |
| Runtime policy overrides | `config_async.py` environment loader used by upload policy registry | Policy registry only | Keep env parsing out of route files. |
| Quarantined raw upload bytes | `backend/core/upload_security/storage.py` | Scanner, audit tooling | Stored under private prefix or private local dir only. |
| Upload verdict metadata | `upload_security_verdicts` Mongo collection | Route handlers, admin audit views | One document per accepted upload attempt. |
| Clean released objects | Upload security service | Business routes and authenticated download endpoint | Routes consume clean object references, not raw request bytes. |
| Hub raw data batches | Hub data validation service | Hub admin APIs, conversion service | Store bounded compressed object refs instead of unbounded Mongo `frames` arrays. |
| Stroke chunks | `stroke_ingest_async.py` plus validation helpers | Finalizer and canonical ingest service | Validate encoded and decoded payload limits before insert. |

### Default Policy Limits

All values are defaults in `backend/core/upload_security/policies.py`. Each has an env override named `UPLOAD_POLICY_<POLICY_ID>_<FIELD>`, for example `UPLOAD_POLICY_PDF_DOCUMENT_MAX_SIZE_MB=75`.

| Policy ID | Current routes | Max size | Count / parser limits | Allowed types |
|---|---|---:|---|---|
| `registration_document` | Admin registration docs | 20 MB | max 8 files per request | PDF, PNG, JPEG |
| `registration_reply_attachment` | Registration status replies | 20 MB each, 100 MB total | max 10 files | PDF, PNG, JPEG |
| `support_message_attachment` | Admin/superadmin messages | 20 MB each, 100 MB total | max 10 files | PDF, PNG, JPEG |
| `debugger_document` | `/api/v1/debugger/upload` and legacy `/api/debugger/upload` RAG document upload | 10 MB | PDF max 100 pages, image max 25 MP, Office ZIP/OLE guard | PDF, DOCX, DOC, PNG, JPEG, WebP |
| `stoody_book_pdf` | `/api/v1/stoody-book/sessions/{session_id}/pdfs` | 10 MB | max 100 pages, preserve existing 100 MB workspace storage cap | PDF |
| `pdf_document` | `/api/v1/pdf/upload` main file | 50 MB | max 250 pages | PDF |
| `answer_sheet_pdf` | `/api/v1/pdf/upload` answer sheet | 25 MB | max 150 pages | PDF |
| `exam_template_file` | DCR template | 25 MB | PDF max 50 pages, image max 25 MP | PDF, PNG, JPEG |
| `direct_ocr_pdf` | `/api/v1/pdf/direct-ocr` | 25 MB | max 100 pages | PDF |
| `tally_question_source_pdf` | `/api/v1/exam-tally/question-source/preview` | 25 MB | max 100 pages | PDF |
| `manual_question_image` | `/api/v1/pdf/questions` `question_image` and `option_images`, plus manual OCR image uploads | 8 MB each, 80 MB total | max 1 question image, max 10 option images, max 25 MP | PNG, JPEG, WebP |
| `generic_image_upload` | `/api/v1/images/upload` | 10 MB | max 25 MP | PNG, JPEG, GIF, BMP, WebP |
| `school_logo` | Settings logo | 3 MB | max 8 MP | PNG, JPEG, WebP |
| `bulk_students` | Student CSV/XLS/XLSX | 5 MB | max 10,000 rows, max 80 columns | CSV, XLSX, XLS |
| `bulk_tutors` | Tutor CSV/XLS/XLSX | 5 MB | max 10,000 rows, max 80 columns | CSV, XLSX, XLS |
| `bulk_timetable` | Timetable CSV/XLS/XLSX | 10 MB | max 20,000 rows, max 80 columns | CSV, XLSX, XLS |
| `teaching_material` | Tutor teaching materials | 50 MB | images max 25 MP, PDF max 250 pages | Current teaching materials allowlist |
| `desktop_diagnostics_zip` | Desktop diagnostics ZIP | 25 MB | max 500 entries, max 100 MB uncompressed, max depth 4 | ZIP |
| `desktop_bug_image` | Desktop bug report images | 5 MB each, 20 MB total | max 8 files, max 25 MP | PNG, JPEG, WebP, BMP |
| `camera_answer_image` | Mobile camera answer page | 12 MB | max 25 MP | JPEG, PNG |
| `hub_raw_data_batch` | `/api/v1/hubs/{hub_id}/data/upload` | 50 MB request body | max 20 sessions, max 50,000 frames/session, max 100,000 frames/batch, max 8 KB/frame JSON | JSON |
| `hub_stroke_chunk` | `/api/v1/ingest/strokes/{exam_id}/{pen_mac}` | 768 KB request body | max 512 KB base64 string, max 384 KB decoded payload, max `total_chunks` 5,000 | JSON/base64 |
| `hub_stroke_finalize` | `/api/v1/ingest/strokes/{exam_id}/{pen_mac}/complete` | 10 MB request body | max 5,000 chunks, max 500 pages, max 20,000 strokes/page | JSON |

### Binary Upload Flow

1. Route receives `UploadFile`.
2. `read_upload_file_limited(file, policy)` streams bytes and rejects once `max_size_bytes + 1` is reached.
3. `detect_file_type(bytes, filename, content_type)` compares extension, declared MIME, and magic bytes.
4. Raw bytes are written to private quarantine storage with a random server key.
5. Malware scanner runs against the quarantined bytes. Production scanner failure returns 503 and stores `scan_failed`.
6. Only after a clean scanner verdict, parser-specific probes run inside limits: PDF page count, image dimensions, ZIP/Office table inspection, archive limits, and spreadsheet row/column preview.
7. Clean, parser-accepted objects are copied/moved to private released storage.
8. Route-specific logic consumes released bytes or storage refs.
9. User downloads go through authenticated endpoints or presigned URLs with `X-Content-Type-Options: nosniff` and safe `Content-Disposition`.

### Structured Data Upload Flow

1. Request body middleware rejects bodies above the route policy before Pydantic allocates large objects.
2. Pydantic validators enforce counts, field lengths, base64 size, decoded size, and allowed shapes.
3. Service-level validation checks tenant/hub ownership, lifecycle, checksum, idempotency, and duplicate handling.
4. Large raw frame collections are stored as compressed private objects and Mongo stores only metadata plus object refs.

---

## Implementation Tasks

### Task 1: Central Upload Policy Registry

**Files:**
- Create: `backend/core/upload_security/__init__.py`
- Create: `backend/core/upload_security/policies.py`
- Modify: `backend/config_async.py`
- Test: `backend/tests/test_upload_policy_registry.py`

- [ ] Create `UploadPolicy`, `BinaryUploadPolicy`, and `StructuredUploadPolicy` Pydantic models with fields for `policy_id`, `max_size_bytes`, `max_files`, `max_total_size_bytes`, `allowed_extensions`, `allowed_mime_types`, `allowed_magic_types`, `max_pdf_pages`, `max_image_pixels`, `max_archive_entries`, `max_archive_uncompressed_bytes`, `max_rows`, `max_columns`, `max_sessions`, `max_frames_per_session`, `max_frames_per_batch`, `max_frame_json_bytes`, `max_total_chunks`, `max_pages`, and `max_strokes_per_page`.
- [ ] Add `DEFAULT_UPLOAD_POLICIES` in `policies.py` using exactly the policy IDs and defaults from the table above.
- [ ] Add `get_upload_policy(policy_id: str) -> UploadPolicy` and `all_public_upload_policies() -> dict[str, dict]`.
- [ ] Add env override support in `policies.py`, not in route files. Use names like `UPLOAD_POLICY_CAMERA_ANSWER_IMAGE_MAX_SIZE_MB`, `UPLOAD_POLICY_HUB_RAW_DATA_BATCH_MAX_FRAMES_PER_BATCH`, and `UPLOAD_POLICY_SUPPORT_MESSAGE_ATTACHMENT_MAX_FILES`.
- [ ] Keep `config_async.py` changes limited to generic upload security toggles: `UPLOAD_SECURITY_ENABLED`, `UPLOAD_AV_ENABLED`, `UPLOAD_AV_FAIL_CLOSED`, `CLAMD_HOST`, `CLAMD_PORT`, `UPLOAD_PRIVATE_LOCAL_DIR`, `UPLOAD_QUARANTINE_PREFIX`, `UPLOAD_RELEASED_PREFIX`, `UPLOAD_MAX_REQUEST_BODY_MB`, and `UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK`.
- [ ] Write tests that prove default values load, env overrides change only the named policy field, invalid policy IDs raise a 500-safe configuration error, and public policy serialization does not expose internal storage paths.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_policy_registry.py -q`

### Task 2: Request Body Limit Middleware

**Files:**
- Create: `backend/middleware/request_size_limit.py`
- Create: `backend/core/upload_security/routes.py`
- Modify: `backend/main_async.py`
- Test: `backend/tests/test_upload_request_size_middleware.py`

- [ ] Implement ASGI middleware that checks `Content-Length` before body consumption and returns 413 when it exceeds the configured route policy or `UPLOAD_MAX_REQUEST_BODY_MB`.
- [ ] Wrap `receive()` to count streamed/chunked request bytes and return 413 if the stream exceeds the limit even without `Content-Length`.
- [ ] Create `UPLOAD_ROUTE_POLICY_MAP` in `backend/core/upload_security/routes.py`. It is the single source used by both request-size middleware and the static coverage test.
- [ ] Add a method-aware route-to-policy resolver with most-specific matching before prefix matching. Required mappings include every current upload route: `POST /api/v1/auth/admin/register` and `POST /auth/admin/register` registration attachments -> `registration_document`; `POST /api/v1/auth/admin/registration-status-message` and `POST /auth/admin/registration-status-message` -> `registration_reply_attachment`; `POST /api/v1/admin/superadmin-messages` and `POST /api/v1/superadmin/tenants/{tenant_id}/messages` -> `support_message_attachment`; `POST /api/v1/pdf/upload -> pdf_document` with field overrides for `exam_template_file` and `answer_sheet_pdf`; `POST /api/v1/pdf/direct-ocr -> direct_ocr_pdf`; any route that stores templates through `_store_exam_template_file` -> `exam_template_file`; `POST /api/v1/exam-tally/question-source/preview -> tally_question_source_pdf`; `POST /api/v1/images/upload -> generic_image_upload`; `POST /api/v1/admin/settings/logo -> school_logo`; `POST /api/v1/teaching-materials/upload -> teaching_material`; `POST /api/v1/desktop-diagnostics/upload -> desktop_diagnostics_zip`; `POST /api/v1/desktop-bug-reports/submit -> desktop_bug_image`; `POST /api/v1/admin/students/bulk/preview` and `POST /api/v1/admin/students/bulk/import -> bulk_students`; `POST /api/v1/tutor/tutors/bulk/preview` and `POST /api/v1/tutor/tutors/bulk/import -> bulk_tutors`; `POST /api/v1/admin/timetable/bulk-upload/preview` and `POST /api/v1/admin/timetable/bulk-upload/import -> bulk_timetable`; `POST /api/v1/ingest/camera/{exam_id}/{student_id}/{page_num} -> camera_answer_image`; `POST /api/v1/debugger/upload -> debugger_document`; `POST /api/debugger/upload -> debugger_document`; `POST /api/v1/stoody-book/sessions/{session_id}/pdfs -> stoody_book_pdf`; `POST /api/v1/pdf/questions -> manual_question_image`; `POST /api/v1/ingest/strokes/{exam_id}/{pen_mac}/complete -> hub_stroke_finalize`; `POST /api/v1/ingest/strokes/{exam_id}/{pen_mac} -> hub_stroke_chunk`; and `POST /api/v1/hubs/{hub_id}/data/upload -> hub_raw_data_batch`.
- [ ] Do not apply `hub_raw_data_batch` to broad `/api/v1/hubs/` traffic. The resolver must match only the `/data/upload` suffix for hub data uploads so provisioning, command polling, audit, and hub metadata APIs keep their normal body limits.
- [ ] Register the middleware before route handlers and after CORS/TrustedHost setup in `main_async.py`.
- [ ] Add tests for `Content-Length` rejection, chunked rejection, allowed body pass-through, all entries in `UPLOAD_ROUTE_POLICY_MAP`, `/api/v1/debugger/upload` and `/api/debugger/upload` both resolving to `debugger_document`, `/ingest/strokes/{exam_id}/{pen_mac}/complete` using `hub_stroke_finalize`, `/ingest/strokes/{exam_id}/{pen_mac}` using `hub_stroke_chunk`, and unrelated `/api/v1/hubs/{hub_id}/commands/pending` not receiving the `hub_raw_data_batch` cap.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_request_size_middleware.py -q`

### Task 3: Type Detection And Parser Guards

**Files:**
- Create: `backend/core/upload_security/detection.py`
- Create: `backend/core/upload_security/validation.py`
- Test: `backend/tests/test_upload_validation.py`

- [ ] Implement `read_upload_file_limited(file, policy)` that reads in chunks and raises HTTP 413 once the policy size is exceeded.
- [ ] Implement cheap pre-scan magic detection for PDF (`%PDF-`), PNG, JPEG, GIF, BMP, WebP, broad ZIP, legacy Office OLE Compound File Binary (`D0 CF 11 E0 A1 B1 1A E1`), CSV text, and unknown bytes. Do not inspect ZIP central directories, parse PDFs, open images with Pillow, or invoke pandas/openpyxl before the malware scanner returns a clean verdict.
- [ ] Require extension, declared MIME, and magic family to agree with the selected policy. Allow `application/octet-stream` only for routes where legacy clients already send it and magic bytes match.
- [ ] Add post-scan PDF page-count guard using `pypdf.PdfReader` against a `BytesIO` and reject encrypted or unreadable PDFs unless the policy explicitly allows them. Do not OCR before this guard.
- [ ] Add post-scan image pixel guard using Pillow `Image.open()` plus `Image.verify()` and reject decompression-bomb warnings as 400.
- [ ] Add post-scan ZIP/Office guard using `zipfile.ZipFile` that rejects path traversal, absolute paths, nested archive depth above policy, too many entries, and total uncompressed bytes above policy.
- [ ] Add post-scan spreadsheet guard that reads only the minimum needed metadata first, then enforces row and column caps after pandas/openpyxl parsing.
- [ ] Write tests with small byte fixtures for valid PDF/image/CSV/ZIP, mismatched extension/MIME/magic, oversized body, ZIP traversal, ZIP uncompressed overflow, and image pixel overflow.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_validation.py -q`

### Task 4: Malware Scanner And Verdict Persistence

**Files:**
- Create: `backend/core/upload_security/scanner.py`
- Create: `backend/core/upload_security/verdicts.py`
- Modify: `backend/requirements.txt`
- Test: `backend/tests/test_upload_scanner_and_verdicts.py`

- [ ] Add a scanner abstraction with `scan_bytes(bytes, filename, policy_id) -> ScanResult`.
- [ ] Implement ClamAV `clamd` TCP support using a small client dependency or a direct socket protocol. Use `CLAMD_HOST` and `CLAMD_PORT`.
- [ ] In development, allow `UPLOAD_AV_ENABLED=false` and mark verdicts as `clean` with `scanner_name="disabled-dev"` only when the loaded backend settings report `settings.DEBUG_MODE is True`. Do not read `NODE_ENV` directly in scanner code; use the existing `config_async.settings` object.
- [ ] In production, if `UPLOAD_AV_ENABLED=true` and the scanner is unavailable, return 503 and store `scan_failed`. If malware is detected, return 400 and store `rejected`.
- [ ] Store verdicts in `upload_security_verdicts` with `upload_id`, `policy_id`, `status`, `sha256`, `size_bytes`, `original_filename`, `declared_content_type`, `detected_magic_type`, `scanner_name`, `scanner_version`, `scan_started_at`, `scan_finished_at`, `tenant_db`, `user_id`, `purpose_metadata`, `authorization_subject`, `quarantine_storage_path`, `released_storage_path`, and `rejection_reason`.
- [ ] Add indexes on `upload_id`, `sha256`, `tenant_db`, `status`, and `created_at`.
- [ ] Test clean, malware, scanner unavailable fail-closed, scanner disabled dev, and verdict persistence shape.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_scanner_and_verdicts.py -q`

### Task 5: Private Storage And No Public Fallback

**Files:**
- Create: `backend/core/upload_security/storage.py`
- Modify: `backend/utils/s3_storage.py`
- Modify: `backend/docs/S3_STORAGE_MIGRATION.md`
- Test: `backend/tests/test_upload_private_storage.py`

- [ ] Implement private upload storage with two namespaces: `quarantine/{tenant}/{upload_id}/{random_name}` and `released/{tenant}/{policy_id}/{upload_id}/{safe_name}`.
- [ ] For S3, set content type, server-side encryption, and metadata including `upload_id`, `policy_id`, `sha256`, and `verdict=clean` only after release.
- [ ] For local development, write to `backend/data/private_uploads`, not `backend/uploads`.
- [ ] Change `utils/s3_storage.py` so production S3 failure does not silently fallback to public local storage unless `UPLOAD_ALLOW_PUBLIC_LOCAL_FALLBACK=true`. Default this setting to false in production.
- [ ] Keep existing `upload_file()` behavior for generated derivatives where needed, but route all user-controlled raw uploads through the new private storage service.
- [ ] Update S3 migration docs to state that local fallback is development-only unless explicitly enabled.
- [ ] Test local private paths, S3 metadata construction with mocks, production fail-closed behavior, and no returned path under `uploads/` for quarantined files.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_private_storage.py -q`

### Task 6: Authenticated Download And Public Mount Guard

**Files:**
- Create: `backend/api/v1/upload_downloads_async.py`
- Modify: `backend/main_async.py`
- Test: `backend/tests/test_upload_downloads.py`

- [ ] Add `GET /api/v1/uploads/{upload_id}/download` for authenticated users with tenant ownership checks against `upload_security_verdicts`, then purpose-specific authorization using the verdict's `purpose_metadata` and `authorization_subject`.
- [ ] Prefer business endpoints for user-facing downloads. The generic upload download route is allowed only for purposes that register an explicit authorization callback such as `can_download_support_attachment`, `can_download_teaching_material`, `can_download_stoody_book_pdf`, or `can_download_desktop_diagnostic`.
- [ ] Store enough `purpose_metadata` at upload time to authorize inside a tenant: `purpose`, `collection`, `document_id` or object ID, `session_id` when relevant, `admin_id`, `tutor_id`, `student_id`, and `created_by`. Do not treat shared `tenant_db` as sufficient authorization.
- [ ] Serve only `status=clean` released objects.
- [ ] Add `Content-Disposition: attachment; filename="<safe name>"`, `X-Content-Type-Options: nosniff`, and a conservative content type from the verdict.
- [ ] Replace or guard the current `app.mount("/uploads", StaticFiles(directory=str(_BACKEND_DIR / "uploads")), name="uploads")`. Production default must not expose raw `backend/uploads`. Keep a dev-only mount behind `UPLOAD_ENABLE_PUBLIC_STATIC_MOUNT=true` if needed for legacy generated images.
- [ ] Review `/images` separately. Generated images can remain public only if they are not raw user uploads and have no tenant-private data.
- [ ] Test unauthenticated denial, cross-tenant denial, same-tenant unauthorized user denial, rejected verdict denial, missing purpose authorization callback denial, clean authorized download success, and headers.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_downloads.py -q`

### Task 7: File Upload Service

**Files:**
- Create: `backend/core/upload_security/service.py`
- Test: `backend/tests/test_upload_security_service.py`

- [ ] Implement `secure_upload(file, policy_id, actor, db, purpose_metadata, authorization_subject) -> CleanUpload`. `actor` may be an authenticated `current_user` dict or `None` for intentionally unauthenticated flows such as pre-account registration, but `authorization_subject` is always required.
- [ ] Validate `purpose_metadata` and `authorization_subject` before quarantine, scanning, or release. They must include a stable `purpose`, tenant/workspace owner fields when known, and the business object identifiers needed for later authorization.
- [ ] `secure_upload` must perform size-limited read, cheap detection, quarantine write, scanner verdict, post-scan parser guards, clean release, and verdict persistence in that order.
- [ ] Return `CleanUpload(upload_id, original_filename, content_type, size_bytes, sha256, detected_magic_type, released_storage_path, purpose_metadata, bytes)` where `bytes` may be omitted for routes that only need storage refs.
- [ ] Add `secure_upload_many(files, policy_id, actor, db, purpose_metadata_factory, authorization_subject_factory)` that enforces file count and total bytes from policy while assigning per-file `purpose_metadata` and `authorization_subject`.
- [ ] Make error responses consistent: 400 for type/parser policy rejection, 413 for size, 503 for scanner/storage unavailable, 500 for unexpected internal errors.
- [ ] Test successful flow, each rejection class, post-scan parser invocation order, scanner rejection preventing parser calls, required authorization subject, and multi-file total limit with per-file metadata.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_security_service.py -q`

### Task 8: Wire PDF, OCR, Tally, And Template Uploads

**Files:**
- Modify: `backend/api/v1/pdf_async.py`
- Modify: `backend/api/v1/exam_tally_async.py`
- Test: `backend/tests/test_pdf_upload_security.py`
- Test: `backend/tests/test_tally_upload_security.py`

- [ ] In `/api/v1/pdf/upload`, replace `.filename.endswith(".pdf")` and direct `await file.read()` with `secure_upload(file=file, policy_id="pdf_document", actor=current_user, db=db, purpose_metadata={"purpose": "pdf_document", "document_id": document_id, "document_type": document_type}, authorization_subject=f"document:{document_id}")`.
- [ ] Apply `answer_sheet_pdf` to `answer_sheet` and `exam_template_file` to `exam_template`.
- [ ] Ensure page counting and OCR use `CleanUpload.bytes` from the clean released object, not the raw request body.
- [ ] In `/api/v1/pdf/direct-ocr`, validate with `direct_ocr_pdf` before `call_sarvam_ocr`.
- [ ] In `/api/v1/exam-tally/question-source/preview`, validate with `tally_question_source_pdf` before layout analysis, Sarvam OCR, PyMuPDF augmentation, or GPT extraction.
- [ ] Keep generated derivatives under a separate `derived/` prefix and attach `source_upload_id` to derivative metadata.
- [ ] Test malicious type mismatch, oversize PDF, scanner rejection, encrypted/unreadable PDF, valid PDF upload, valid direct OCR, and valid tally preview.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_pdf_upload_security.py tests/test_tally_upload_security.py -q`

### Task 9: Wire Image, Logo, Support, Debugger, Stoody Book, Desktop, Teaching, And Camera Uploads

**Files:**
- Modify: `backend/utils/message_attachments.py`
- Modify: `backend/api/v1/images_async.py`
- Modify: `backend/api/v1/settings_async.py`
- Modify: `backend/api/v1/debugger_async.py`
- Modify: `backend/api/v1/stoody_book_async.py`
- Modify: `backend/api/v1/pdf_async.py`
- Modify: `backend/api/v1/desktop_diagnostics_async.py`
- Modify: `backend/api/v1/desktop_bug_reports_async.py`
- Modify: `backend/api/v1/teaching_materials_async.py`
- Modify: `backend/api/v1/camera_upload_async.py`
- Modify: `backend/api/v1/auth_async.py`
- Test: `backend/tests/test_image_and_attachment_upload_security.py`
- Test: `backend/tests/test_debugger_and_stoody_book_upload_security.py`
- Test: `backend/tests/test_camera_upload_security.py`

- [ ] Replace message attachment constants in `message_attachments.py` with `support_message_attachment` policy.
- [ ] Tighten `/api/v1/images/upload` to image-only using `generic_image_upload`. If document upload compatibility is needed, create a separate product endpoint and policy rather than leaving documents on an image route.
- [ ] Validate school logo with `school_logo` before base64 storage in settings.
- [ ] Add `current_user: Dict[str, Any] = Depends(get_current_user)` to `backend/api/v1/debugger_async.py` upload before calling `secure_upload`. The same authenticated handler covers both mounted prefixes: `/api/v1/debugger/upload` and legacy `/api/debugger/upload`.
- [ ] Validate debugger upload with `debugger_document` before RAG extraction or chunking. Preserve supported PDF, Word, and image formats only if cheap magic/type checks pass, the file is quarantined, the scanner returns clean, and post-scan parser guards pass.
- [ ] If product requires anonymous debugger uploads later, do not overload this route. Add a separate explicitly anonymous route with `authorization_subject="anonymous_debugger_session:{sessionId_hash}"`, strict rate limiting, no generic download permission, and an explicit policy-map entry.
- [ ] Validate `/api/v1/stoody-book/sessions/{session_id}/pdfs` with `stoody_book_pdf` before PDF text extraction, Mongo Binary storage, or workspace storage accounting. Keep the existing 100 MB total workspace cap as a second business limit after the per-file policy passes.
- [ ] Validate `/api/v1/pdf/questions` `question_image` and `option_images` with `manual_question_image` before base64 conversion or `save_image_to_disk`.
- [ ] Validate desktop diagnostics ZIP with `desktop_diagnostics_zip` before storage.
- [ ] Validate desktop bug report images with `desktop_bug_image` and enforce per-file plus total limits.
- [ ] Validate teaching materials with `teaching_material`, preserving its current S3-required posture unless product explicitly allows private local storage.
- [ ] Validate mobile camera answer pages with `camera_answer_image` before hashing or canonical artifact creation.
- [ ] For registration docs and reply attachments in `auth_async.py`, validate with `registration_document` and `registration_reply_attachment` before Mongo base64 storage. Prefer storing released private object refs instead of new base64 blobs.
- [ ] Test valid uploads for each policy, oversize rejection, MIME/magic mismatch, total-size rejection, same-tenant unauthorized download denial where applicable, and scanner failure.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_image_and_attachment_upload_security.py tests/test_debugger_and_stoody_book_upload_security.py tests/test_camera_upload_security.py -q`

### Task 10: Wire Spreadsheet Bulk Imports

**Files:**
- Modify: `backend/api/v1/student_bulk_upload.py`
- Modify: `backend/api/v1/tutor_bulk_upload.py`
- Modify: `backend/api/v1/timetable_bulk_upload.py`
- Test: `backend/tests/test_bulk_upload_security.py`

- [ ] Replace direct `await file.read()` limits with `read_upload_file_limited()` using `bulk_students`, `bulk_tutors`, or `bulk_timetable`.
- [ ] Validate extension/MIME/broad magic before quarantine, then run pandas/openpyxl parsing only after the scanner returns clean.
- [ ] Apply the same policy to preview and import endpoints. Do not allow timetable import to skip the preview size cap.
- [ ] Reject spreadsheets exceeding max rows or columns before business validation begins.
- [ ] Reject ZIP-based spreadsheet containers with path traversal, nested archives, too many entries, or uncompressed size overflow.
- [ ] Test student, tutor, and timetable preview/import with valid CSV/XLSX, oversize file, wrong magic, too many rows, and malicious ZIP container.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_bulk_upload_security.py -q`

### Task 11: Wire Hub Raw Data And Stroke Upload Limits

**Files:**
- Modify: `backend/api/v1/hub_ops_async.py`
- Modify: `backend/api/v1/stroke_ingest_async.py`
- Modify: `stoody-multi-pen/edge_hub/hub_core/uplink_client.py`
- Modify: `stoody-multi-pen/edge_hub/modes/exampen/upload_worker.py`
- Test: `backend/tests/exam_conductor/test_hub_upload_security.py`
- Test: `backend/tests/exam_conductor/test_stroke_upload_limits.py`
- Test: `stoody-multi-pen/edge_hub/tests/test_upload_limits.py`

- [ ] Add Pydantic validators to `HubRawSessionUpload` and `HubDataUploadRequest` for session count, frame count, file size, frame JSON byte size, raw session key length, and string field lengths.
- [ ] Store hub raw `frames` as compressed private NDJSON objects through the upload storage layer. Keep Mongo fields for `frame_count`, `file_size`, `frames_storage_path`, `sha256`, `status`, and `validation_status`. Do not store unbounded `frames` arrays in Mongo.
- [ ] Cap admin `fetch_data` command output by adding a payload field such as `max_sessions` and `max_frames_per_session` from `hub_raw_data_batch` policy.
- [ ] Update `uplink_client.py` to respect backend command caps and stop reading more than the requested limit from `raw_frame_spool`.
- [ ] Add Pydantic validators to `StrokeChunkUpload` for `payload_base64` length, valid base64, decoded byte length, `total_chunks <= 5000`, and allowed `exam_type`.
- [ ] Add validators to `FinalizeRequest` for `total_chunks`, page count, raw stroke count per page, page number range, and expected checksum length.
- [ ] Update `upload_worker.py` to use chunk sizes compatible with `hub_stroke_chunk` and to handle 413/400 responses as permanent policy failures in the ledger.
- [ ] Test hub data over session/frame/body caps, valid bounded upload, compressed object ref storage, stroke base64 over limit, decoded over limit, too many chunks, too many pages, and valid finalization.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/exam_conductor/test_hub_upload_security.py tests/exam_conductor/test_stroke_upload_limits.py -q`
- [ ] Run from `stoody-multi-pen/edge_hub`: `python -m pytest tests/test_upload_limits.py -q`

### Task 12: Frontend And Mobile UX Limits

**Files:**
- Modify: `backend/api/v1/upload_downloads_async.py`
- Modify: `frontend/src/services/api.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/admin/BulkUploadDialog.tsx`
- Modify: `frontend/src/services/timetableService.ts`
- Modify: `frontend/src/services/teachingMaterialsService.ts`
- Modify: `stoody-multi-pen/mobile-app/src/services/cameraUploadService.ts`
- Modify: `stoody-multi-pen/mobile-app/src/services/exampenHubLocalService.ts`
- Test: frontend/mobile existing test locations as available

- [ ] Add `GET /api/v1/upload-policies/public` that returns non-sensitive policy IDs, max sizes, allowed extensions, and max file counts for UX only.
- [ ] Update web upload forms to display and pre-check the policy limits, but keep backend enforcement authoritative.
- [ ] Update mobile camera upload to compress or reject images above `camera_answer_image` limit before upload.
- [ ] Update hub local upload trigger UI to surface permanent policy failures differently from retryable network failures.
- [ ] Run: `cd frontend; npm run build`
- [ ] Run from `stoody-multi-pen/mobile-app`: `npx tsc --noEmit`

### Task 13: Ops, Deployment, And Health

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/skillbot_nginx_alb.conf`
- Modify: `backend/deploy.sh`
- Modify: `backend/ops/remote_deploy_python_service.sh`
- Modify: `backend/docs/BACKEND_SETUP.md`
- Create: `backend/docs/UPLOAD_SECURITY.md`
- Test: deployment config checks

- [ ] Add ClamAV/clamd installation and service enablement to deployment docs/scripts where production backend is provisioned.
- [ ] Add environment docs for `UPLOAD_SECURITY_ENABLED`, `UPLOAD_AV_ENABLED`, `UPLOAD_AV_FAIL_CLOSED`, `CLAMD_HOST`, `CLAMD_PORT`, `UPLOAD_PRIVATE_LOCAL_DIR`, `UPLOAD_MAX_REQUEST_BODY_MB`, and selected `UPLOAD_POLICY_*` overrides. Document that scanner-disabled clean verdicts are allowed only when `config_async.settings.DEBUG_MODE` is true.
- [ ] Lower nginx `client_max_body_size` from 120 MB to a value aligned with the largest supported request plus margin, recommended 64 MB unless product explicitly needs larger teaching material uploads through the backend.
- [ ] Add health reporting for scanner availability, but do not make `/health` fail if scanner is disabled in development.
- [ ] Add logs/metrics counters for accepted, rejected, scan_failed, oversized, type_mismatch, and scanner_unavailable by `policy_id`.
- [ ] Run: `cd backend; nginx -t` on the deployment host as part of CI/CD or manual release validation.

### Task 14: Static Upload Coverage Guard

**Files:**
- Create: `backend/tests/test_upload_policy_coverage.py`
- Create: `backend/core/upload_security/coverage.py`
- Modify: `backend/docs/UPLOAD_SECURITY.md`

- [ ] Build an AST-based coverage check that scans `backend/api/v1/**/*.py` for route handlers with parameters annotated as `UploadFile` or `List[UploadFile]` and FastAPI `File` defaults.
- [ ] Require each discovered upload route to appear in the central `UPLOAD_ROUTE_POLICY_MAP` from `backend/core/upload_security/routes.py` with method, path template or regex, policy ID, and owner note.
- [ ] Allow explicit exemptions only in `backend/core/upload_security/coverage.py` with `reason`, `owner`, and `expires_on`. The initial exemption list should be empty unless a route is proven not to accept user-controlled bytes.
- [ ] Add assertions that the current known routes are covered: `auth_async.py` registration attachments, `admin_async.py` attachments, `superadmin_async.py` attachments, `debugger_async.py`, `stoody_book_async.py`, `pdf_async.py` upload/template/direct OCR/question images, `exam_tally_async.py`, `images_async.py`, `settings_async.py`, `teaching_materials_async.py`, desktop diagnostics, desktop bug reports, camera upload, student/tutor/timetable bulk upload.
- [ ] The test must fail when a developer adds a new `UploadFile` route parameter with a FastAPI `File` default without adding a policy map entry or a time-limited exemption.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_policy_coverage.py -q`

### Task 15: Legacy Upload Audit And Migration

**Files:**
- Create: `backend/scripts/migrations/audit_legacy_uploads.py`
- Create: `backend/scripts/migrations/migrate_public_uploads_to_private.py`
- Test: `backend/tests/test_legacy_upload_audit.py`

- [ ] Build a dry-run audit that scans Mongo references to existing local/S3 uploads and classifies them by likely policy.
- [ ] For local `backend/uploads`, compute SHA-256, detect magic type, and produce a CSV/JSON report of clean-looking, missing, mismatched, and unknown files.
- [ ] Do not mark legacy files clean without a scanner verdict. Use status `legacy_unverified` until scanned.
- [ ] Build a migration that copies verified clean files to released private storage and updates DB references.
- [ ] Keep rollback output containing original storage paths and updated document IDs.
- [ ] Run: `cd backend; .\venv\Scripts\python -m pytest tests/test_legacy_upload_audit.py -q`

---

## Validation Matrix

- Policy registry: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_policy_registry.py -q`
- Upload validators: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_validation.py -q`
- Scanner/storage service: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_scanner_and_verdicts.py tests/test_upload_private_storage.py tests/test_upload_security_service.py -q`
- PDF/OCR/tally: `cd backend; .\venv\Scripts\python -m pytest tests/test_pdf_upload_security.py tests/test_tally_upload_security.py -q`
- Attachments/images/camera/debugger/Stoody Book/manual question images: `cd backend; .\venv\Scripts\python -m pytest tests/test_image_and_attachment_upload_security.py tests/test_debugger_and_stoody_book_upload_security.py tests/test_camera_upload_security.py -q`
- Bulk imports: `cd backend; .\venv\Scripts\python -m pytest tests/test_bulk_upload_security.py -q`
- Hub and stroke ingest: `cd backend; .\venv\Scripts\python -m pytest tests/exam_conductor/test_hub_upload_security.py tests/exam_conductor/test_stroke_upload_limits.py -q`
- Static upload coverage: `cd backend; .\venv\Scripts\python -m pytest tests/test_upload_policy_coverage.py -q`
- Edge hub limits: from `stoody-multi-pen/edge_hub`, run `python -m pytest tests/test_upload_limits.py -q`
- Existing focused regressions: `cd backend; .\venv\Scripts\python -m pytest tests/test_student_section_required.py tests/test_main_404_handler.py tests/exam_conductor/test_stroke_ingest_exam_owner.py -q`
- Frontend build: `cd frontend; npm run build`
- Mobile typecheck: from `stoody-multi-pen/mobile-app`, run `npx tsc --noEmit`

## Rollout Order

1. Add policy registry, validators, request-size middleware, scanner abstraction, and private storage behind `UPLOAD_SECURITY_ENABLED=false`.
2. Enable for low-risk routes first: desktop bug images, school logo, message attachments.
3. Enable for high-risk parser routes: PDF upload, direct OCR, tally preview, spreadsheets.
4. Enable for mobile camera and teaching materials.
5. Enable structured hub raw data and stroke limits.
6. Disable public `/uploads` in production after DB references and download endpoints are verified.
7. Run legacy upload audit and migrate historical clean objects.

## Failure Modes And Mitigations

| Failure mode | Mitigation | Residual risk |
|---|---|---|
| Scanner unavailable in production | Fail closed with 503 and `scan_failed` verdict | Users cannot upload until scanner recovers. |
| S3 outage | Fail closed for production raw uploads; optional private local development fallback | Upload availability depends on storage health. |
| Large chunked request without `Content-Length` | ASGI receive wrapper counts bytes and aborts | Some clients may see connection abort instead of JSON body. |
| Existing frontend expects `/uploads` direct URLs | Authenticated download endpoint and migration mapping | Legacy links need backfill or compatibility redirects. |
| Hub raw session payload too large | Backend caps plus edge command caps | Old hubs need update before strict enforcement. |
| Route bypasses service during future development | AST coverage test fails when a new `UploadFile` plus FastAPI `File` route lacks a policy map entry or explicit expiring exemption | Dynamic route construction can still need review. |

## Ready-To-Implement Gate

- [x] Success criteria are explicit.
- [x] Writable owners are identified.
- [x] Interfaces are described.
- [x] Transactional boundaries are identified.
- [x] Failure modes are named.
- [x] Validation is specific.
- [x] Assumptions are recorded.

## Assumptions

- ClamAV/clamd is acceptable as the first malware engine.
- Production should fail closed when scanner or private storage is unavailable.
- Existing product flows should remain, but generic image upload should stop accepting PDFs/docs unless a separate document route is intentionally created.
- Local development may use private local storage under `backend/data/private_uploads`; production should prefer S3 private storage.
- Scanner-disabled clean verdicts use `config_async.settings.DEBUG_MODE`, not direct `NODE_ENV` checks.
- Not runtime verified. This is a source-grounded implementation plan only.
