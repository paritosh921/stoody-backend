/** AI pipeline types: recognition results, confidence, step breakdowns. */

import type { AISourceType } from "./enums";

/** AI recognition result for a single question. */
export interface QuestionResult {
  question_id: string;
  recognized_text: string;
  confidence: number;
  step_breakdown?: string[];
}

/** NATS event: AI recognition complete for a student's exam. */
export interface AIResultEvent {
  event_id: string;
  event_type: "ai.result";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  student_id: string;
  model_version: string;
  source_type?: AISourceType;
  question_results: QuestionResult[];
}

/** Student-facing answer detail with AI analysis. */
export interface AnswerInsight {
  question_id: string;
  answer_image_uri: string;
  recognized_text: string;
  confidence: number;
  step_breakdown?: string[];
  feedback?: string;
}
