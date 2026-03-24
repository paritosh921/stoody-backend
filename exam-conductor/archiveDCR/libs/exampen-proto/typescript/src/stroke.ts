/** Stroke domain types: raw chunks, processed strokes, upload status. */

import type { ChunkBindingStatus, UploadPath } from "./enums";

/** A single chunk of raw stroke data uploaded from the hub. */
export interface StrokeChunkUploadRequest {
  exam_id: string;
  pen_mac: string;
  chunk_index: number;
  total_chunks: number;
  payload_base64: string;
  checksum_crc32: string;
  upload_path: UploadPath;
  idempotency_key: string;
  binding_status?: ChunkBindingStatus;
}

/** Server acknowledgement for a received stroke chunk. */
export interface IngestAck {
  exam_id: string;
  pen_mac: string;
  chunk_index: number;
  accepted: boolean;
  deduplicated: boolean;
  next_expected_chunk: number;
  pen_upload_complete?: boolean;
}

/** Per-pen upload reconciliation state. */
export interface PenUploadStatus {
  pen_mac: string;
  acked_chunks: number[];
  total_chunks: number;
  complete: boolean;
}

/** Upload progress for all pens in an exam. */
export interface ExamUploadStatus {
  exam_id: string;
  pens: PenUploadStatus[];
}

/** Mapping of a stroke segment to a question on a page. */
export interface PageAssignment {
  page_number: number;
  question_id: string;
  point_count: number;
}

/** NATS event: raw stroke chunk ingested from hub. */
export interface StrokeRawEvent {
  event_id: string;
  event_type: "stroke.raw";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  pen_mac: string;
  chunk_index: number;
  total_chunks: number;
  payload_base64: string;
  checksum_crc32: string;
  upload_path: UploadPath;
  binding_status?: ChunkBindingStatus;
}

/** NATS event: stroke data normalized, deduplicated, committed. */
export interface StrokeProcessedEvent {
  event_id: string;
  event_type: "stroke.processed";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  pen_mac: string;
  student_id?: string;
  normalized_stroke_uri: string;
  page_assignments: PageAssignment[];
}
