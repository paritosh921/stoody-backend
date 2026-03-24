// ---------------------------------------------------------------------------
// Shared types for Teacher BFF API responses and payloads.
// ---------------------------------------------------------------------------

export interface TeacherExamCard {
  exam_id: string;
  title: string;
  subject_id: string;
  scheduled_at: string;
  state: string;
  class_label?: string;
}

export interface ClassScoreRow {
  student_id: string;
  student_name: string;
  total_score: number;
  percentile?: number;
  ai_confidence: number;
  miss_indicator_count?: number;
  plagiarism_flag_count?: number;
}

export interface QuestionDetail {
  question_id: string;
  current_score: number;
  confidence: number;
  recognized_text?: string;
  miss_indicator?: string;
  copy_image_uri?: string;
}

export interface TeacherStudentDetail {
  student_id: string;
  student_name: string;
  total_score: number;
  answer_pages?: string[];
  questions: QuestionDetail[];
}

export interface ScoreOverridePayload {
  question_id: string;
  new_score: number;
  reason: string;
}

export type MissState =
  | 'answered'
  | 'miss_no_strokes'
  | 'miss_sync_failure'
  | 'miss_pen_inactive'
  | 'not_attempted_confirmed';

export interface MissIndicatorCell {
  student_id: string;
  question_id: string;
  state: MissState;
}

export interface MissIndicatorMatrix {
  exam_id: string;
  students: string[];
  questions: string[];
  cells: MissIndicatorCell[];
}

export interface PlagiarismPreview {
  flag_id: string;
  student_a_id: string;
  student_b_id: string;
  question_id: string;
  composite_score: number;
  severity: string;
  teacher_verdict?: string;
}

export interface ObjectionInboxItem {
  objection_id: string;
  exam_id: string;
  student_id: string;
  question_id: string;
  status: string;
  filed_at: string;
}

export interface ChatMessage {
  message_id: string;
  sender_id: string;
  content: string;
  attachment_uri?: string;
  sent_at: string;
  read_at?: string;
}

export interface LeaderboardEntry {
  rank: number;
  student_id: string;
  student_name: string;
  total_score: number;
  percentile: number;
}

export interface ClassStats {
  mean: number;
  median: number;
  std_dev: number;
  pass_rate: number;
  question_difficulty?: { question_id: string; avg_score: number }[];
}

export interface RubricStep {
  name: string;
  marks: number;
}

export interface RubricQuestion {
  question_number: number;
  max_marks: number;
  answer_type: string;
  steps: RubricStep[];
}

export interface RubricPayload {
  questions: RubricQuestion[];
  confidence_threshold: number;
}

export interface QuestionRegion {
  question_number: number;
  x_pct: number;
  y_pct: number;
  width_pct: number;
  height_pct: number;
}
