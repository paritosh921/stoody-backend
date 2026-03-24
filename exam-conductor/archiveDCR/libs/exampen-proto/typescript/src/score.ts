/** Score domain types: projections, overrides, audit history, workflow. */

import type { ScoreEventType, ScoreLifecycleState } from "./enums";

/** Score for a single step within a question. */
export interface StepScore {
  label: string;
  awarded: number;
  max: number;
}

/** Score projection for one question. */
export interface QuestionScore {
  question_id: string;
  ai_score: number;
  current_score: number;
  max_score: number;
  confidence: number;
  override_reason?: string;
  step_scores?: StepScore[];
}

/** Full score projection for one student in an exam. */
export interface StudentScoreDetail {
  exam_id: string;
  student_id: string;
  total_score: number;
  max_score?: number;
  lifecycle_state: ScoreLifecycleState;
  published_at?: string;
  objection_window_closes_at?: string;
  questions: QuestionScore[];
}

/** Teacher override for a single question score. */
export interface ScoreOverrideRequest {
  teacher_id: string;
  new_score: number;
  reason: string;
}

/** Single entry in the score audit event stream. */
export interface ScoreHistoryItem {
  event_id: string;
  event_type: ScoreEventType;
  old_value: number;
  new_value: number;
  actor_id: string;
  reason?: string;
  created_at: string;
}

/** Request to finalize reviewed scores for an exam. */
export interface FinalizeRequest {
  actor_id: string;
}

/** Request to publish finalized scores and open objection window. */
export interface PublishRequest {
  actor_id: string;
  objection_window_days: number;
}

/** Result of a score workflow state transition. */
export interface WorkflowStateResponse {
  exam_id: string;
  lifecycle_state: ScoreLifecycleState;
  changed_at: string;
}

/** NATS event: score projection changed. */
export interface ScoreUpdatedEvent {
  event_id: string;
  event_type: "score.updated";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  student_id: string;
  question_id?: string;
  lifecycle_state: ScoreLifecycleState;
  total_score: number;
  previous_total_score?: number;
  reason: ScoreEventType;
}
