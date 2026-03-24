/** Matches OpenAPI SessionSummary schema */
export interface SessionSummary {
  exam_id: string;
  state: string;
  timer_remaining_sec: number;
  upload_status: UploadStatus;
  backend_seen_at?: string;
}

export type UploadStatus = "pending" | "in_progress" | "complete" | "partial";

/** Matches OpenAPI PenSyncRow schema */
export interface PenSyncRow {
  pen_mac: string;
  student_id?: string;
  sync_status: PenSyncStatus;
  bytes_received?: number;
  total_chunks?: number;
}

export type PenSyncStatus =
  | "pending"
  | "connecting"
  | "syncing"
  | "complete"
  | "failed"
  | "timeout";

/** Matches OpenAPI DongleRow schema */
export interface DongleRow {
  dongle_mac: string;
  status: DongleStatus;
  connected_pens: number;
  capacity?: number;
}

export type DongleStatus = "healthy" | "degraded" | "failed";

/** Matches OpenAPI WebSocketEnvelope schema */
export interface WebSocketEnvelope<T = unknown> {
  event_type: WsEventType;
  payload: T;
}

export type WsEventType =
  | "session.snapshot"
  | "sync.progress"
  | "dongle.health"
  | "upload.progress";

/** Hub connectivity info (derived from session snapshot) */
export interface HubConnectivity {
  wifi_connected: boolean;
  backend_reachable: boolean;
  signal_strength?: number;
}

/** Upload progress payload */
export interface UploadProgressPayload {
  exam_id: string;
  uploaded_chunks: number;
  total_chunks: number;
  status: UploadStatus;
}

/** Active alert for the alert banner */
export interface Alert {
  id: string;
  severity: "error" | "warning" | "info";
  message: string;
  timestamp: number;
}
