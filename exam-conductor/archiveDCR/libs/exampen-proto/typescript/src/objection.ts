/** Objection types: filing, resolution, escalation, events. */

import type { ObjectionResolution, ObjectionStatus } from "./enums";

/** Lightweight objection record for list views. */
export interface ObjectionSummary {
  objection_id: string;
  exam_id: string;
  student_id: string;
  question_id: string;
  status: ObjectionStatus;
  filed_at: string;
}

/** Full objection with text and resolution detail. */
export interface ObjectionDetail extends ObjectionSummary {
  objection_text: string;
  assigned_to?: string;
  resolution?: string;
  resolution_reason?: string;
  score_delta?: number;
}

/** Request to file a new objection. */
export interface CreateObjectionRequest {
  exam_id: string;
  student_id: string;
  question_id: string;
  objection_text: string;
}

/** Request to resolve an objection. */
export interface ResolveObjectionRequest {
  actor_id: string;
  resolution: ObjectionResolution;
  reason: string;
  new_score?: number;
}

/** Request to escalate an objection to a senior reviewer. */
export interface EscalateObjectionRequest {
  actor_id: string;
  escalated_to: string;
  reason: string;
}

/** NATS event: objection lifecycle transition. */
export interface ObjectionEvent {
  event_id: string;
  event_type: "objection";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  objection_id: string;
  student_id: string;
  question_id: string;
  action: ObjectionStatus;
  state: ObjectionStatus;
  actor_id?: string;
}
