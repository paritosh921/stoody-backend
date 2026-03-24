/** NATS event envelope types — re-exports from domain modules. */

export type { AIResultEvent } from "./ai";
export type { ObjectionEvent } from "./objection";
export type { CopyReadyEvent, PageReadyEvent } from "./page";
export type {
  PlagiarismCheckEvent,
  PlagiarismResultEvent,
} from "./plagiarism";
export type { ScoreUpdatedEvent } from "./score";
export type { StrokeProcessedEvent, StrokeRawEvent } from "./stroke";

/** NATS event: exam lifecycle state transition. */
export interface ExamLifecycleEvent {
  event_id: string;
  event_type: "exam.lifecycle";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  from_state: string;
  to_state: string;
  actor_id: string;
  reason?: string;
}

/** Union of all NATS event types for type-safe routing. */
export type ExamPenEvent =
  | import("./stroke").StrokeRawEvent
  | import("./stroke").StrokeProcessedEvent
  | import("./page").PageReadyEvent
  | import("./page").CopyReadyEvent
  | import("./ai").AIResultEvent
  | import("./score").ScoreUpdatedEvent
  | ExamLifecycleEvent
  | import("./objection").ObjectionEvent
  | import("./plagiarism").PlagiarismCheckEvent
  | import("./plagiarism").PlagiarismResultEvent;
