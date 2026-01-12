# SmartBoard Full Migration Plan

Migrate complete SmartBoard system from `stoody-pen-multi` to main `skill-bot` codebase.

---

## 1. Backend API Migration

### Source: `stoody-pen-multi/backend/app/api/`

| File | Purpose | Action |
|------|---------|--------|
| `dashboard.py` | Pens status, WebSocket, color updates | Copy to `backend/api/v1/` |
| `hub.py` | BLE hub registration and heartbeat | Copy to `backend/api/v1/` |
| `notes.py` | Notes CRUD for teacher canvas | Copy to `backend/api/v1/` |
| `ocr.py` | OCR analysis endpoint | Copy to `backend/api/v1/` |
| `question_attempts.py` | Question locking, evaluation | Copy to `backend/api/v1/` |
| `pages.py` | Page management | Copy to `backend/api/v1/` |
| `pen_frames.py` | Pen frame data | Copy to `backend/api/v1/` |

### Source: `stoody-pen-multi/backend/app/core/`

| File | Purpose | Action |
|------|---------|--------|
| `dashboard_registry.py` | In-memory pen registry | Copy to `backend/core/` |
| `pen_workers.py` | Background pen processing | Copy to `backend/core/` |
| `pen_router.py` | Pen routing logic | Copy to `backend/core/` |
| `ocr_service.py` | OCR service (Mistral/OpenAI) | Copy to `backend/core/` |
| `storage.py` | Storage utilities | Copy to `backend/core/` |
| `stroke_engine_adapter.py` | Stroke processing | Copy to `backend/core/` |

### Router Registration

Add to `backend/main_async.py`:
- `/dashboard/*` - Dashboard API
- `/hub/*` - Hub registration
- `/notes/*` - Notes CRUD
- `/ocr/*` - OCR analysis
- `/question-attempts/*` - Question attempts

---

## 2. Frontend (Already Copied)

Already copied in previous step:
- 8 SmartBoard components
- SmartBoardContext.tsx
- smartBoardService.ts
- hooks, types, utils

---

## 3. Edge Folder Migration

### Action
```
Copy: stoody-pen-multi/edge/
To: skill-bot/MAIN_ble_hub_deploy/
```

### Configuration Updates
Update config files in `MAIN_ble_hub_deploy/hub_config/` to point to main backend:
- Change API URL from stoody-pen-multi backend to skill-bot backend
- Update WebSocket endpoint

---

## 4. Verification

1. Start backend - check Swagger for new endpoints
2. Start frontend - verify SmartBoard tab works
3. Test BLE hub connection (if hardware available)

---

## Files to Modify

| File | Change |
|------|--------|
| `backend/main_async.py` | Register new routers |
| `MAIN_ble_hub_deploy/hub_config/*.json` | Update URLs |
