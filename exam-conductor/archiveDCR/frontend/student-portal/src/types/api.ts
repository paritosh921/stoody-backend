// ─── Types matching student-bff.openapi.yaml schemas ───

export type ExamStatus =
  | "upcoming"
  | "scores_pending"
  | "published"
  | "objection_window_open"
  | "locked";

export interface StudentExamCard {
  exam_id: string;
  title: string;
  subject_name?: string;
  scheduled_at: string;
  status: ExamStatus;
}

export interface QuestionScore {
  question_id: string;
  marks_obtained: number;
  max_marks: number;
  ai_confidence?: number;
  miss_indicator?: string;
}

export interface StudentScoreView {
  exam_id: string;
  total_score: number;
  percentage: number;
  percentile: number;
  pass_fail?: "pass" | "fail";
  teacher_id?: string;
  questions: QuestionScore[];
}

export interface AnswerInsight {
  question_id: string;
  answer_image_uri: string;
  recognized_text: string;
  confidence: number;
  step_breakdown?: string[];
  feedback?: string;
}

export type ObjectionStatus =
  | "filed"
  | "assigned"
  | "reviewing"
  | "resolved"
  | "escalated";

export interface StudentObjection {
  objection_id: string;
  exam_id: string;
  question_id: string;
  status: ObjectionStatus;
  objection_text?: string;
  resolution_reason?: string;
  new_score?: number;
}

export interface CreateStudentObjectionRequest {
  exam_id: string;
  question_id: string;
  objection_text: string;
}

export interface Message {
  message_id: string;
  sender_id: string;
  content: string;
  attachment_uri?: string;
  sent_at: string;
  read_at?: string;
}

export interface SendMessageRequest {
  content: string;
  attachment_uri?: string;
}

export interface PerformanceHistoryEntry {
  exam_id: string;
  score: number;
  percentile: number;
}

export interface PerformanceView {
  history: PerformanceHistoryEntry[];
  strengths: string[];
  weaknesses: string[];
}

// ─── Generic list wrapper ───

export interface ListResponse<T> {
  items: T[];
}
