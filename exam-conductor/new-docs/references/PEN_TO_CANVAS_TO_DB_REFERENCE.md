# Stoody Pen: Pen-to-Canvas-to-Database Reference

This document is the consolidated technical reference for the current Stoody pen stack across:

- `stoody-ble-agent/agent`
- `frontend`
- `backend`

It is intended to describe the system as it exists in code today, not as a generic target architecture. The goal is that another engineer could reproduce the same behavior, message shapes, storage semantics, and failure handling from this document plus the referenced code.

Existing helper references used while preparing this document:

- `stoody-ble-agent/docs/ARCHITECTURE.md`
- `frontend/docs/canvas-stroke-flow-reference.md`
- `stoody-ble-agent/agent/docs/canvas-stroke-flow-reference.md`

Those remain useful, but this file is the single end-to-end reference for current pen-stack behavior only.

This file is **not** the authority for the ExamPen architecture split. For active architecture decisions, use the active root specs under `architecture/`.

---

## 1. High-Level System Map

```text
PHYSICAL PEN
  -> BLE frames / GATT notifications
  -> agent BLE manager
  -> coordinate processing + scope detection
  -> stroke processor
  -> canonical stroke batches

From there the flow splits:

LIVE WRITING
  -> local agent websocket: wss://localhost:8001/ws/live
  -> frontend queueing, preview rendering, committed batch merge
  -> ActivePageModel
  -> localStorage page snapshot
  -> dirty queue
  -> backend canvas_pages sync

OFFLINE REPLAY
  -> pen offline packet stream
  -> same coordinate + canonical stroke pipeline
  -> agent-owned offline page buffer
  -> direct backend /strokes/pages/batch upload
  -> frontend receives completion event
  -> frontend refreshes/reconciles from backend

CLOUD RELAY
  -> agent also sends live canonical stroke batches over its cloud websocket
  -> pen backend routes/broadcasts and supports token/provisioning flows
```

Three important design facts in the current implementation:

1. The pen is an input device, not the identity key for persisted pages.
2. Canonical stroke construction happens in the agent for both live and offline data.
3. The frontend owns the active live page experience, but offline replay is agent-owned until upload completes.

---

## 2. Main Code Entry Points

### Agent

- Local HTTP/WebSocket server: `stoody-ble-agent/agent/src/stoody_agent/api.py`
- Runtime orchestration: `stoody-ble-agent/agent/src/stoody_agent/service.py`
- BLE transport and device status: `stoody-ble-agent/agent/src/stoody_agent/ble_manager.py`
- Pen protocol and CRC: `stoody-ble-agent/agent/src/stoody_agent/pen_protocol.py`
- Coordinate parsing / page-book detection / button detection: `stoody-ble-agent/agent/src/stoody_agent/coordinates.py`
- Stroke processing / smoothing / coordinate transform: `stoody-ble-agent/agent/src/stoody_agent/stroke_processor.py`
- Offline sync state machine: `stoody-ble-agent/agent/src/stoody_agent/offline_sync.py`
- Cloud websocket client: `stoody-ble-agent/agent/src/stoody_agent/websocket_client.py`

### Frontend

- Local websocket transport: `frontend/src/services/stoody/strokeWebSocket.ts`
- WebSocket hook: `frontend/src/hooks/stoody/useStrokeWebSocket.ts`
- Pen/network state and queues: `frontend/src/hooks/stoody/usePenStatus.ts`
- Main canvas container: `frontend/src/components/stoody/StoodyPenCanvas.tsx`
- Active page load/save/hydration: `frontend/src/hooks/stoody/useCanvasPages.ts`
- Local storage keys and page/session persistence: `frontend/src/services/stoody/canvasStorage.ts`
- Server sync layer: `frontend/src/services/stoody/canvasSync.ts`
- Stroke serialization: `frontend/src/services/stoody/strokePersistence.ts`
- Coordinate scale helpers: `frontend/src/utils/stoody/coordinateMapper.ts`
- Render smoothing and gap handling: `frontend/src/utils/stoody/enhanced-canvas/strokeUtils.ts`

### Backend

- Canvas page persistence API: `backend/api/v1/strokes_async.py`
- Pen-token helpers: `backend/core/pen_tokens.py`
- Pen frame ingestion router: `backend/api/v1/pen_frames.py`
- Pen registry / routing: `backend/core/pen_router.py`

### Pen Backend / Relay

- Agent and frontend websocket/token endpoints: `stoody-ble-agent/server/api/agent_routes.py`
- Frontend websocket broadcast manager: `stoody-ble-agent/server/services/websocket_manager.py`

---

## 3. Connection and Auth Flows

## 3.1 Frontend -> Local Agent

The browser connects to the agent over:

- WebSocket: `wss://localhost:8001/ws/live`
- Health/API base: `https://localhost:8001/api/...`

The local live websocket is defined in `agent/src/stoody_agent/api.py`:

- route: `@app.websocket("/ws/live")`
- on connect:
  - accepts the socket
  - subscribes to the runtime's local live queue via `subscribe_local_live()`
  - immediately sends a `pen_status` payload
  - then forwards queued local events one by one

Frontend transport behavior in `frontend/src/services/stoody/strokeWebSocket.ts`:

- primary URL: `VITE_AGENT_LOCAL_WSS_URL` or default `wss://localhost:8001/ws/live`
- fallback URL: legacy remote relay URL when configured
- message types handled:
  - `stroke_batch`
  - `stroke_preview`
  - `stroke_preview_clear`
  - `pen_status`
  - `button_action`
  - offline sync events
  - calibration events
- reconnect behavior:
  - attempts alternate URL if the initial candidate never opens
  - otherwise reconnects with backoff

Important: the browser does not authenticate the local websocket with a bearer token. The browser token is currently ignored by the local websocket constructor and is only meaningful for backend HTTP calls.

## 3.2 Agent -> Cloud WebSocket

The agent uses `CloudWebSocket` in `agent/src/stoody_agent/websocket_client.py`.

- It loads stored device credentials from disk.
- It connects to the configured backend websocket using `device_token` as a query parameter.
- It keeps a separate send lock and reconnect logic.
- On send failure, live stroke batches are not dropped immediately; they can be queued locally for later drain.

Key security and session behaviors:

- device token is required for cloud websocket auth
- logout sets a stopped flag before disconnect to avoid reconnect races
- reset-for-new-user closes the existing socket so stale credentials are not reused
- incoming websocket commands are schema-validated

## 3.3 Agent Provisioning and Device Registration

The local agent exposes login/provisioning endpoints in `agent/src/stoody_agent/api.py`.

Current flow:

1. User authenticates through frontend/backend flow.
2. Agent receives callback with user token.
3. Agent requests a provisioning token.
4. Agent registers the device and stores:
   - `device_id`
   - `device_token`
   - `user_id`
5. Agent also stores the user access token separately for later backend API usage.

This matters because:

- cloud websocket needs a device token
- offline sync uploads need a user token

## 3.4 Frontend -> Main Backend

Backend sync in `frontend/src/services/stoody/canvasSync.ts` uses browser auth:

- token source: `localStorage.getItem("skillbot_token")`
- every request sends:
  - `Authorization: Bearer <token>`
  - `Content-Type: application/json`

Used APIs include:

- `PUT /strokes/pages`
- `POST /strokes/pages/batch`
- `GET /strokes/pages/{bookType}/{pageNumber}`
- `POST /strokes/pages/bulk-load`

## 3.5 Offline Sync Upload Auth

The agent owns offline upload in `service.py`:

- it resolves a user token
- then posts page batches to backend `/strokes/pages/batch`
- payloads use `version: 0` to force merge semantics on the backend

The pen backend also exposes `/agent/user-token` in `stoody-ble-agent/server/api/agent_routes.py` for short-lived user token minting from a device-authenticated context.

---

## 4. Pen Protocol, Frames, and CRC

Protocol implementation: `stoody-ble-agent/agent/src/stoody_agent/pen_protocol.py`

Frame shape:

```text
Head(2) + SerialNum(4) + ID(4) + Cmd(1) + DataFormat(1) + DataLen(2) + Data(N) + CRC16(2)
```

Constants:

- frame header: `0x5A 0x5B`
- minimum frame length: 16 bytes
- data format:
  - `0x00` byte stream
  - `0x01` JSON

CRC behavior:

- primary CRC: `CRC-16/XMODEM`
- parser also accepts:
  - `CRC-16/CCITT-FALSE`
  - `CRC-16/MODBUS`

Current commands in use:

- `0x00` coordinate data
- `0x03` device info
- `0x04` device status
- `0x05-0x0B` offline sync family
- `0x0B` offline transfer complete
- `0xE8` heartbeat

Important replication rule:

- CRC is computed over everything from `SerialNum` through `Data`
- the fixed head bytes are not included in the CRC input

---

## 5. BLE, Battery, and Connection Stability

BLE handling lives in `agent/src/stoody_agent/ble_manager.py`.

Key behaviors:

- coordinate packets are delivered as 14-byte units
- device info updates can include battery
- battery polling runs in a background task
- battery polling is gated by `set_battery_poll_gate(...)`

The runtime uses that gate to avoid battery traffic interfering with active stroke flow.

Battery propagation path:

1. BLE manager receives battery level.
2. Runtime updates `_status["battery"]`.
3. Runtime emits `pen_status` to local live subscribers.
4. Frontend `usePenStatus` updates `deviceInfo.battery`.

Frontend stability logic in `frontend/src/hooks/stoody/usePenStatus.ts`:

- `IDLE_GRACE_MS = 120000`
- `DISCONNECT_GRACE_MS = 5000`
- reconnect churn window: `30000`
- reconnect churn threshold: `2`

Derived connection quality states:

- `connected`
- `degraded`
- `reconnecting`
- `offline`

These states directly affect hydration behavior. `useCanvasPages` will skip server refresh/reconcile when connection quality is not `connected`.

---

## 6. Coordinate Processing, Page Detection, and Button Logic

Main file: `stoody-ble-agent/agent/src/stoody_agent/coordinates.py`

This layer turns raw pen coordinate records into:

- preview stroke
- completed strokes
- current page / current book type
- detected paper button actions

### 6.1 Page and book type handling

The coordinate service:

- validates page number and book type from raw coordinate packets
- keeps current page/book as session state
- uses confirmation logic before accepting page/book changes
- uses different page-switch confirmation counts for:
  - live writing
  - offline replay

Important constants currently in play:

- `PAGE_SWITCH_CONFIRM_FRAMES`
- `OFFLINE_PAGE_SWITCH_CONFIRM_FRAMES`

This means page changes are intentionally debounced and not accepted from a single noisy sample.

### 6.2 Pen-up/down handling

Current pen-up/down logic is not based only on firmware pen state. It uses:

- pressure thresholds
- hysteresis
- confirm frame counts

Key constants:

- `PRESSURE_UP_THRESHOLD`
- `PRESSURE_DOWN_THRESHOLD`
- `PRESSURE_UP_CONFIRM_FRAMES`
- live-mode override `LIVE_PRESSURE_UP_CONFIRM_FRAMES`

### 6.3 Jump detection and stroke-fragment stability

The coordinate service applies anti-teleport and anti-fragment heuristics:

- max jump distance
- confirmation distance
- return distance
- max time delta
- straight-line continuation rules
- pressure ratio checks
- speed ratio checks

The important architectural split is:

- **live mode** uses relaxed thresholds for fast cursive tolerance
- **offline replay** keeps stricter thresholds

This is implemented through `JumpThresholds.for_mode(source_mode)`.

### 6.4 Button actions on paper

Button definitions are coded in `coordinates.py` and must match frontend button mappings.

Current behavior:

- A4/A5 paper buttons are defined by physical positions in mm
- buttons can be circle or rectangle hit areas
- some buttons are single-tap, some rely on double-tap logic
- actions include:
  - `MENU`
  - `PREV_PAGE`
  - `NEXT_PAGE`
  - `SUBMIT`
  - color changes

Detected button actions are emitted immediately and also forwarded to cloud where possible.

---

## 7. Stroke Processing and Canonical Stroke Shape

Main file: `stoody-ble-agent/agent/src/stoody_agent/stroke_processor.py`

This layer converts raw stroke point lists into canonical render-ready data.

### 7.1 Physical dimensions and scale

Book sizes are encoded by book type, for example:

- `MS/MN/MM` -> A5 portrait
- `LS/LN/LM` -> A4 portrait
- `LL/LW` -> A3 landscape

Two core scaling constants:

- pen units: `10 units / mm`
- canvas resolution: `4 px / mm`

Therefore:

- A5 portrait `148 x 210 mm` -> `1480 x 2100` pen units -> `592 x 840 px`
- A4 portrait `210 x 297 mm` -> `2100 x 2970` pen units -> `840 x 1188 px`

### 7.2 Coordinate transform

Transform rules:

- X scales directly from pen units to canvas width
- Y is inverted because:
  - pen origin is bottom-left
  - canvas origin is top-left
- values are clamped to valid bounds

Agent and frontend both use the same `4 px / mm` coordinate space.

### 7.3 Edge compensation and calibration

Current transform pipeline includes:

- manual edge offsets via environment variables
- optional auto edge calibration
- edge inset to avoid clipping at boundaries

These exist to compensate for pens whose physical page edges do not report exact raw `0,0`.

### 7.4 Smoothing and cleanup

The agent currently applies:

- spike suppression
- short excursion suppression
- centerline smoothing
- optional pressure calibration

The canonical stroke output still preserves point-level information; it is not reduced to a purely visual path.

### 7.5 Canonical stroke fields

Canonical strokes are enriched with:

- `version`
- `strokeId`
- `deviceId`
- `penMac`
- `sessionId`
- `sourceMode`
- `pageNumber`
- `bookType`
- `tool`
- `color`
- `baseWidthMm`
- `points`
- `bbox`
- `svgPath`
- `renderMode`
- `startedAt`
- `endedAt`
- `pointCount`
- `originalPointCount`

Important replication rule:

- `strokeId` is the cross-layer identity key used for dedup and merge
- downstream layers must not invent incompatible IDs for the same stroke

---

## 8. Live Path: From Pen to Frontend Canvas

This is the current live committed-stroke path.

### 8.1 Agent live path

In `service.py`:

1. BLE coordinates arrive.
2. Runtime calls `_handle_coordinates(..., source_mode="live")`.
3. `coordinates.process(...)` returns:
   - preview stroke info
   - completed strokes
   - current page/book
   - button actions
4. Runtime updates `_status["current_page"]` and `_status["book_type"]`.
5. Preview is broadcast as:
   - `stroke_preview`
   - `stroke_preview_clear`
6. Completed strokes are grouped by `(pageNumber, bookType)`.
7. Each scoped group is converted into a local `stroke_batch` payload.
8. That payload is:
   - broadcast immediately to local websocket subscribers
   - also added to the cloud stroke buffer for later flush

### 8.2 Agent cloud flush

Buffered live strokes are flushed roughly every `150 ms` by `_flush_stroke_buffer()`.

Behavior:

- enrich canonical strokes
- regroup by scope
- send batch over cloud websocket
- if send fails:
  - enqueue locally for retry
  - do not silently discard the batch

### 8.3 Frontend queueing

Frontend path:

1. `StrokeWebSocket` receives `stroke_batch`.
2. `usePenStatus.handleBackendStrokeReceived(...)`:
   - dedups by batch ID
   - updates connection fallback state
   - pushes batch into `pendingBackendStrokesRef`
   - increments `pendingStrokeTick`

Preview follows a separate queue:

- `pendingPreviewEventsRef`
- `pendingPreviewTick`

### 8.4 Frontend committed batch merge

`StoodyPenCanvas.tsx` is the main integration point.

Committed batch processing does all of the following:

- infer actual scope from explicit batch fields or stroke vote
- convert canonical strokes to `StrokeElement`
- clamp and normalize points where needed
- guard against noisy foreign-page jumps
- merge queued foreign batches when status later confirms the new scope
- perform transactional page switch when needed
- merge committed strokes into `ActivePageModel`
- render the updated stroke set via `EnhancedCanvasAdapter`
- project the result into React state via `updateStrokes(...)`

Current important live invariants:

1. Preview strokes are visual-only and do not own persistence.
2. Committed batches are the persistence source.
3. `switchActivePage()` owns transactional page boundary behavior.
4. Current page/book refs must not change before the transaction establishes the new page.

---

## 9. Preview Rendering

Preview events are generated only for live mode.

Agent side:

- preview payloads come from `_build_local_preview_stroke_payload(...)`
- preview events are rate-limited by `LOCAL_PREVIEW_MIN_INTERVAL_MS`

Frontend side:

- `usePenStatus` queues preview events
- `StoodyPenCanvas` consumes them separately from committed batches
- preview is drawn on the temp/preview canvas via `EnhancedCanvasAdapter`
- preview is cleared explicitly with `stroke_preview_clear`

Important:

- preview should not be treated as durable stroke data
- if a preview was shown but no committed stroke batch followed, it is not part of saved page state

---

## 10. Frontend Page Ownership and Rendering

The current frontend design uses a single active-page model plus derived views.

### 10.1 Active in-memory owner

`StoodyPenCanvas.tsx` holds:

- `activePageModelRef`
- page/book refs that track the model's current identity

The model is the authoritative active in-memory page state for:

- page identity
- committed strokes
- dirty flag
- last modified/session metadata

### 10.2 Derived views

Derived layers:

- `EnhancedCanvasAdapter` -> imperative rendered canvas state
- `useCanvasPages` -> React state + localStorage-facing page state

The design goal is:

- committed batches merge into the model first
- React state and rendered canvas follow the model

### 10.3 Page switch transaction

Current `switchActivePage()` behavior in `StoodyPenCanvas.tsx`:

1. flush pending committed stroke queue
2. snapshot current model
3. save current page to localStorage
4. mark old page dirty for backend sync
5. optionally sync immediately for notes canvas
6. load target page from local storage
7. load target page into the model
8. render target strokes immediately
9. call `navigateToPage(..., { skipOldPageSave: true })`

This is the core protection against page-switch overwrite bugs.

---

## 11. Scaling, Smoothing, and Stroke Rendering on Frontend

Two main files:

- `frontend/src/utils/stoody/coordinateMapper.ts`
- `frontend/src/utils/stoody/enhanced-canvas/strokeUtils.ts`

### 11.1 Coordinate space

The frontend uses the same book-size mapping as the agent:

- `PIXELS_PER_MM = 4`
- pen range per book type derives from physical dimensions * `10 units/mm`

This means agent-produced canonical points and frontend canvas dimensions stay aligned.

### 11.2 Pressure normalization

When raw pen coordinates are mapped directly on frontend, pressure is normalized roughly into a `0-1` range with a minimum visible floor.

Canonical strokes often already carry normalized usable points, so the main direct mapping utility is most relevant for non-canonical raw flows and compatibility paths.

### 11.3 Stroke rendering options

Frontend render stroke generation uses `perfect-freehand` with options centralized in `getStoodyStrokeOptions(...)`.

Current behaviors:

- no simulated pressure
- moderate thinning, smoothing, and streamline values
- width derived from:
  - `baseWidthMm` when available
  - otherwise `strokeWidth`

### 11.4 Gap filling and smoothing

Frontend render helpers also do:

- dedupe of very near points
- interpolation for large gaps
- segment splitting when distance jumps are too large
- centerline smoothing

`prepareRenderableSegments(...)` currently widens gap tolerance relative to stroke width for smoother within-stroke rendering.

### 11.5 Stitching

`stitchNearbyStrokes(...)` exists but intentionally skips canonical pen strokes:

- canonical live/offline pen strokes represent real pen-down/up cycles
- render-time stitching across those cycles caused text corruption
- fragmentation is handled earlier by:
  - agent threshold tuning
  - within-stroke interpolation

---

## 12. Local Storage and Persistence

Main file: `frontend/src/services/stoody/canvasStorage.ts`

There are two storage key systems.

### 12.1 Legacy pen-scoped keys

Format:

`stoody_pen_canvas_{user}_{bluetoothAddress}_{BOOKTYPE}_page_{pageNumber}`

Used historically for page storage tied to a pen MAC.

### 12.2 Current pen-agnostic server keys

Format:

`stoody_canvas_{user}_{copyId}_{BOOKTYPE}_page_{pageNumber}`

This is the current main page cache key for server-synced pages.

Important:

- page identity is user + copy + book type + page number
- pen MAC is stored as metadata only
- changing pen hardware must not change the page key

### 12.3 Pen session key

Format:

`stoody_pen_session_{user}_{bluetoothAddress}`

Stores:

- last book type
- last page number
- last accessed

### 12.4 Page data shape in localStorage

Stored page payload includes:

- `strokes`
- `lastModified`
- `bookType`
- `pageNumber`
- `bluetoothAddress`
- `userId`
- `source`
- `pageStyle`
- `canvasBackground`
- `version`
- `sessionId`
- `firstActivity`
- `lastActivity`

### 12.5 Stroke serialization

`strokePersistence.ts` converts between frontend `StrokeElement` and stored records.

It supports:

- canonical point arrays
- legacy point arrays
- preservation of:
  - IDs
  - color/tool
  - timestamps
  - `svgPath`
  - `baseWidthMm`
  - `sourceMode`
  - `pageNumber`
  - `bookType`
  - `penMac`
  - `deviceId`

---

## 13. useCanvasPages: Save, Load, Hydrate, and Reconcile

Main file: `frontend/src/hooks/stoody/useCanvasPages.ts`

### 13.1 Initial load behavior

On page change:

1. load page from localStorage by server key
2. set React `strokes`
3. record `loadedPageNumber` and `loadedBookType`
4. optionally refresh from server if:
   - page is stale
   - or missing locally but known on server
   - and connection quality is `connected`

### 13.2 Save behavior

Save triggers include:

- autosave debounce
- page switch
- unmount
- beforeunload

`saveCurrentPage()`:

- reads freshest strokes from `latestStrokesRef` if available
- persists empty pages when they overwrite an older snapshot
- writes pen session when `effectiveAddress` exists
- calls `markPageDirty(...)`

### 13.3 Autosave behavior

Autosave compares current `strokes` with `lastSavedStrokesRef`.

If changed:

- debounce timer is reset
- `saveCurrentPage()` runs after delay

This includes erase-to-empty behavior; empty page state is treated as a meaningful change.

### 13.4 Hydration behavior

Server hydration is merge-based.

When `activePageModelRef` is provided:

- hydration merges into the model via `hydrateModel(...)`
- React state is then set from the model

The hydration guard requires full scope:

- current page number
- current book type

This prevents cross-book contamination.

### 13.5 Reconcile behavior

After offline sync or explicit reconcile:

- frontend reloads already-merged local page data
- if model is active for that scope, it hydrates the model
- otherwise it merges with latest local strokes before setting React state

---

## 14. Dirty Queue and Backend Sync

Main file: `frontend/src/services/stoody/canvasSync.ts`

### 14.1 Dirty queue model

The dirty queue is a per-user set of server page keys.

Characteristics:

- stored in localStorage
- flush timer default: `30000 ms`
- flush paused during offline sync
- flush can also happen:
  - on explicit trigger
  - on unmount/beforeunload path

### 14.2 Flush behavior

`flushDirtyPages(userId)`:

1. loads dirty page snapshots from localStorage
2. serializes each page
3. sends:
   - `PUT /strokes/pages` for single-page path
   - or `POST /strokes/pages/batch` for batch path
4. updates local version on success
5. dispatches `stoody-canvas-pages-synced`

Important current rule:

- local cache is written first
- backend sync is eventual
- if local cache is wrong, the wrong snapshot can later be synced

### 14.3 Server refresh and merge

`loadPageFromServer(...)`:

- fetches page from backend
- converts response to local page format
- merges with existing local page using:
  - local strokes as base
  - server strokes as incoming
- dedups by stroke ID
- uses metadata precedence:
  - `lastModified = max(local, server)`
  - `firstActivity = min`
  - `lastActivity = max`
  - local `version` can win when local is ahead

### 14.4 Offline reconcile

`reconcileOfflineSyncedPages(...)`:

- refreshes server truth for affected pages
- if local dirty state contains extra strokes, it merges and re-syncs
- emits `stoody-canvas-page-reconciled`

---

## 15. Backend Canvas Storage and Merge Rules

Main file: `backend/api/v1/strokes_async.py`

The modern persisted page storage lives in Mongo `canvas_pages`.

### 15.1 Page identity

One document per:

- `user_id`
- `book_type`
- `page_number`
- `copy_id`

Important:

- `pen_mac` is not part of the uniqueness key
- pen MAC is metadata only

### 15.2 Accepted page payload

Backend page upsert accepts:

- page identity
- array of strokes
- style/background metadata
- client last modified timestamp
- version
- session/activity metadata
- optional `device_id`

### 15.3 Merge behavior

Key backend helpers:

- `_merge_stroke_docs(...)`
- `_is_stale_canvas_page_update(...)`
- `_build_merged_page_doc(...)`
- `upsert_canvas_page(...)`
- `batch_upsert_canvas_pages(...)`

Current merge model:

- merge by stroke ID
- do not replace the whole page blindly
- use optimistic versioning and stale-update guards
- prefer additive merge when conflict is possible

### 15.4 Historical raw stroke collection

The backend still exposes historical stroke ingestion/listing against `strokes`.

That is separate from the current page-level persistence model in `canvas_pages`.

Use this distinction:

- `strokes` -> historical batch-oriented stroke documents
- `canvas_pages` -> page snapshot / merged persistence layer used by current Stoody canvas

---

## 16. Offline Sync: Detailed Current Behavior

Main files:

- `agent/src/stoody_agent/offline_sync.py`
- `agent/src/stoody_agent/service.py`

### 16.1 Protocol flow

Current offline command flow:

1. app sends `OFFLINE_SIZE_REQ (0x08)`
2. pen replies with total buffered size
3. app sends `OFFLINE_START (0x09)`
4. pen streams `OFFLINE_PACKET (0x07)` data
5. app ACKs each packet
6. pen sends `OFFLINE_COMPLETE (0x0B)`

### 16.2 Packet handling

For each offline packet:

- serial number is checked
- first packet can realign expected serial
- each 14-byte coordinate record is decoded
- decoded coordinates are fed into the same `_handle_coordinates(...)` path with `source_mode="offlineReplay"`

### 16.3 Canonical construction

Offline replay does not have a separate stroke format.

It uses the same:

- coordinate service
- stroke processor
- scope grouping
- local stroke batch payload builder

Difference:

- live mode broadcasts immediately to frontend and buffers for cloud relay
- offline replay buffers canonical batches page-by-page for agent-owned upload

### 16.4 Offline upload

On sync completion:

1. flush any active partial stroke from coordinate service
2. convert buffered canonical strokes to backend upload format
3. upload page batches to backend `/strokes/pages/batch`
4. send local completion event with uploaded page keys

Current upload details:

- chunks pages in groups of 5 for backend requests
- each page payload uses `version: 0`
- each stroke uses `id = strokeId`
- source is marked as `offline_sync`

### 16.5 Frontend role in offline sync

Frontend is an observer until upload is complete.

It may show:

- started/progress/completed/failed events
- page refresh after completion

But it is not the primary writer of offline replay page state.

---

## 17. Status Events and Message Shapes

Important local websocket event types currently emitted by the agent:

- `pen_status`
- `stroke_batch`
- `stroke_preview`
- `stroke_preview_clear`
- `button_action`
- `offline_sync_started`
- `offline_sync_progress`
- `offline_sync_completed`
- `offline_sync_failed`
- `offline_check_result`
- `offline_awaiting_copy_selection`
- calibration events

### 17.1 pen_status

Fields:

- `connected`
- `pen_mac`
- `battery`
- `queue_size`
- `current_page`
- `book_type`

### 17.2 stroke_batch

Important fields:

- `type`
- `version`
- `batchId`
- `sessionId`
- `deviceId`
- `penMac`
- `copyId`
- `strokes`
- `pageNumber`
- `bookType`
- `sentAt`

### 17.3 Canonical stroke inside a batch

Important fields:

- `strokeId`
- `deviceId`
- `penMac`
- `pageNumber`
- `bookType`
- `startedAt`
- `endedAt`
- `baseWidthMm`
- `sourceMode`
- `points`
- optional `svgPath`

Replication requirement:

- `strokeId` must remain stable across merges and uploads
- downstream consumers should normalize aliases like:
  - `batchId` / `batch_id`
  - `sessionId` / `session_id`
  - `penMac` / `pen_mac`
  - `pageNumber` / `page_number`
  - `bookType` / `book_type`

---

## 18. Error Handling and Bad-Data Handling

Current protections span all three layers.

### 18.1 Agent / protocol

- parser rejects malformed frames
- CRC is checked before trusting payload
- offline packet serial mismatches are detected
- undecodable coordinate units in offline packets are skipped individually, not fatal to the whole packet
- cloud websocket send failure falls back to local offline queue
- local live subscriber queue overflow:
  - `pen_status` and offline events can replace older same-type queue items
  - other event types may be dropped when the queue is full

### 18.2 Agent / coordinate layer

- invalid book types fall back to supported defaults in processing paths
- page/book changes require confirmation
- jump filters quarantine suspicious teleports
- button detection resets when taps are outside button hit areas

### 18.3 Frontend

- batch dedup by batch ID
- preview and committed batches are separated
- hydration is gated by connection quality
- page switches flush pending committed batches first
- full page scope guards prevent cross-book hydration corruption
- stale local load is skipped when preloaded transaction data should win
- empty pages are persisted so erased content does not resurrect

### 18.4 Backend

- page writes use merge semantics instead of naive full replacement
- stale update checks exist
- duplicate-key write errors are sanitized
- user identity resolution handles legacy ID variants

---

## 19. Current Source-of-Truth Rules

These are the practical ownership rules required to reproduce current behavior correctly.

### Live active page

- authoritative in-memory owner: `ActivePageModel` in `StoodyPenCanvas`
- authoritative rendered view: `EnhancedCanvasAdapter` derived from the model
- authoritative durable browser cache: localStorage server-key page snapshot
- authoritative long-term persisted store: backend `canvas_pages`

### Offline replay

- authoritative owner during replay and upload: agent
- frontend should not treat previewed offline data as its own persisted truth
- backend `canvas_pages` becomes durable truth after upload

### Identity

- page identity: `user + copyId + bookType + pageNumber`
- stroke identity: `strokeId`
- pen MAC: metadata, not page identity

---

## 20. How to Replicate This Stack Exactly

To replicate current behavior faithfully, preserve all of the following:

1. **Protocol**
   - frame structure, command bytes, CRC behavior, 14-byte coordinate units

2. **Geometry**
   - `10 pen units/mm`
   - `4 canvas px/mm`
   - Y inversion
   - same book dimension table in agent and frontend

3. **Canonical stroke construction**
   - performed in the agent
   - same canonical fields
   - same `strokeId` semantics

4. **Live path**
   - local websocket immediate broadcast
   - separate preview and committed paths
   - committed batches merge into active page model before persistence

5. **Offline path**
   - same coordinate/stroke pipeline as live
   - agent-owned page buffering
   - direct backend batch upload on completion

6. **Frontend persistence**
   - page switch transaction
   - server-key localStorage page snapshots
   - dirty queue with delayed sync
   - merge-based hydration, not destructive replacement

7. **Backend persistence**
   - page-level merge by stroke ID
   - page identity independent of pen MAC
   - version/stale-update safeguards

8. **Stability logic**
   - page/book confirmation
   - jump filtering
   - disconnect grace
   - reconnect churn gating
   - foreign-page auto-switch protections

If any of the above is simplified independently, the resulting system will not behave the same as the current Stoody stack.

---

## 21. Environment-Dependent Variables

Some behavior depends on deployment configuration and environment variables rather than hardcoded constants only.

Most important categories:

- pressure thresholds
- jump thresholds
- live-mode relaxed thresholds
- edge compensation
- auto-calibration toggles
- JWT secrets and token TTLs
- websocket endpoints

So "100% replication" requires both:

- code-level replication
- equivalent runtime configuration

Without matching environment configuration, behavior can still differ even if the code structure is the same.
