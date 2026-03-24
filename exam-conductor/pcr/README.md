# PCR — Physical Copy Review

Physical Copy Review mode for ExamPen. Traditional paper-based exam workflow where physical answer sheets are photographed or scanned, then processed through the AI scoring pipeline.

## Status

**Not yet implemented.** This directory is a placeholder for future development.

## Planned Approach

PCR will reuse significant parts of the DCR infrastructure:

### Shared with DCR (no new implementation needed)
- `svc-auth` — Same Stoody JWT authentication
- `svc-score-engine` — Same event-sourced scoring, rubric eval, overrides
- `svc-review` — Same objection lifecycle
- `svc-analytics` — Same percentile/leaderboard computation
- `svc-plagiarism` — Same detection algorithms
- `svc-chat` — Same messaging
- `svc-notify` — Same notification triggers
- `svc-teacher-bff` / `svc-student-bff` — Same aggregation layers
- Teacher dashboard / Student portal — Same web frontends

### PCR-specific (new implementation)
- **Copy acquisition**: Mobile camera capture OR bulk scanner upload (replacing pen stroke capture)
- **Image preprocessing**: Deskew, crop, enhance, page segmentation (replacing stroke processing)
- **OCR pipeline**: Image-to-text OCR (replacing HWR from pen strokes)
- **Question detection**: Identify answer regions from scanned pages (replacing pen coordinate-based assignment)
- **Copy storage**: High-resolution image storage with per-page indexing

### Not needed for PCR
- Hub hardware (RPi, BLE dongles, pen firmware)
- Stroke pipeline (svc-stroke-ingest, svc-stroke-proc)
- Hub modules (hub-supervisor, hub-ble-mgr, hub-pen-sync, etc.)
- BLE mobile hub-control mode

## Architecture Notes

PCR can be implemented as:
1. A new `svc-copy-processor` service that replaces the stroke pipeline
2. Enhanced `svc-copy-upload` (already exists in DCR) for bulk scan upload
3. Modified `svc-doc-assembly` to accept scanned images instead of stroke-rendered pages
4. The same `svc-ai-pipeline` with OCR models instead of (or in addition to) HWR models

The scoring, review, analytics, and frontend layers remain identical.
