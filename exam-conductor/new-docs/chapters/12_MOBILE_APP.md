# Chapter 12: Mobile App

## Status
- **Build status:** ACTIVE — invigilator surfaces implemented in `stoody-multi-pen/mobile-app/src/`
- **Authority source:** `integration/STOODY_INTEGRATION_SPEC.md`

## Overview

The mobile app participates in ExamPen through two kinds of flows:

- invigilation and hub-adjacent operations during conducted exams
- Stoody-facing teacher or student views outside the exam hall

Where practice evaluation is exposed, the mobile app reaches it through existing Stoody backend integration. The mobile app does not create a second practice persistence model.

## Implementation Status

The following invigilator surfaces are implemented:

| Surface | Component | Status | Notes |
|---|---|---|---|
| Hub list | `exampenHubService.ts`, `ExamHub` type | **Built** | `GET /api/v1/hubs` |
| Exam selection + hub assignment | `ExamPenHubSelectScreen.tsx` | **Built** | `exampenSessionService.ts` |
| Session dashboard | `ExamPenSessionDashboardScreen.tsx`, `HubStatusCard.tsx` | **Built** | Polls invig console API, per-hub status cards |
| Camera fallback upload | `CameraFallbackScreen.tsx`, `cameraUploadService.ts` | **Built** | Multipart upload with offline retry queue, pending count banner |
| Offline upload queue | `CameraFallbackScreen.tsx` | **Built** | `PendingUpload[]`, retry-all (max 3 retries), discard queue |

**Known gaps:**
- Student ID in camera fallback is manual entry; roster-backed student selector is future work
- No runtime/browser QA done yet on mobile app

## Alignment Rules

1. Conducted-exam collection still routes through the shared ingest substrate.
2. Mobile practice calls must respect the stateless PCR practice endpoint boundary.
3. Mobile flows must not bypass Stoody identity or tutor visibility rules.

## Related Docs

- `integration/STOODY_INTEGRATION_SPEC.md`
- `api/student-bff.openapi.yaml`
- `api/teacher-bff.openapi.yaml`
- `api/invig-console.openapi.yaml`
