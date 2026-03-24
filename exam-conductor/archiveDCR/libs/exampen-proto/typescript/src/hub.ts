/** Hub types: dongle state, pen sync, session, WebSocket envelope. */

import type {
  DongleStatus,
  PenSyncStatus,
  UploadStatus,
  WebSocketEventType,
} from "./enums";

/** Backend view of an exam session for the invigilator console. */
export interface SessionSummary {
  exam_id: string;
  state: string;
  timer_remaining_sec: number;
  upload_status: UploadStatus;
  backend_seen_at?: string;
}

/** Per-pen BLE sync progress row. */
export interface PenSyncRow {
  pen_mac: string;
  student_id?: string;
  sync_status: PenSyncStatus;
  bytes_received?: number;
  total_chunks?: number;
}

/** BLE dongle health and capacity state. */
export interface DongleRow {
  dongle_mac: string;
  status: DongleStatus;
  connected_pens: number;
  capacity?: number;
}

/** WebSocket message envelope for invigilator console updates. */
export interface WebSocketEnvelope {
  event_type: WebSocketEventType;
  payload: Record<string, unknown>;
}
