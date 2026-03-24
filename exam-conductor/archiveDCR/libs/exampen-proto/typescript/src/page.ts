/** Page and copy-upload types: page images, miss indicators, copy pages. */

import type {
  AuthoritativeSource,
  CopyAuthoritativeCandidate,
  MissIndicatorState,
} from "./enums";

/** NATS event: rendered page image ready for AI pipeline. */
export interface PageReadyEvent {
  event_id: string;
  event_type: "page.ready";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  student_id: string;
  page_id: string;
  page_number: number;
  image_uri: string;
  vector_uri?: string;
  authoritative_source: AuthoritativeSource;
  question_ids?: string[];
}

/** NATS event: photographed copy page ingested. */
export interface CopyReadyEvent {
  event_id: string;
  event_type: "copy.ready";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  student_id: string;
  page_number: number;
  copy_image_uri: string;
  authoritative_candidate?: CopyAuthoritativeCandidate;
}

/** Result of uploading a photographed answer page. */
export interface CopyUploadResult {
  exam_id: string;
  student_id: string;
  page_number: number;
  copy_image_uri: string;
  data_source: "copy_image";
}

/** A single copy page record. */
export interface CopyPage {
  page_number: number;
  copy_image_uri: string;
  authoritative_source?: AuthoritativeSource;
}

/** Single cell in the miss indicator matrix. */
export interface MissIndicatorCell {
  student_id: string;
  question_id: string;
  state: MissIndicatorState;
}

/** Student-by-question miss indicator matrix for a teacher view. */
export interface MissIndicatorMatrix {
  exam_id: string;
  students: string[];
  questions: string[];
  cells: MissIndicatorCell[];
}
