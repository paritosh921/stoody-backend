/** BFF view types: teacher and student aggregation surfaces. */

import type { ObjectionStatus, PassFail, StudentExamStatus } from "./enums";

/** Exam card for the teacher dashboard list. */
export interface TeacherExamCard {
  exam_id: string;
  title: string;
  subject_id: string;
  scheduled_at: string;
  state: string;
  class_label?: string;
}

/** One row in the teacher class score grid. */
export interface ClassScoreRow {
  student_id: string;
  student_name: string;
  total_score: number;
  percentile?: number;
  ai_confidence: number;
  miss_indicator_count?: number;
  plagiarism_flag_count?: number;
}

/** Per-question detail for teacher drill-down. */
export interface QuestionDetail {
  question_id: string;
  current_score: number;
  confidence: number;
  recognized_text?: string;
  miss_indicator?: string;
  copy_image_uri?: string;
}

/** Teacher drill-down view for one student's exam. */
export interface TeacherStudentDetail {
  student_id: string;
  student_name: string;
  total_score: number;
  answer_pages?: string[];
  questions: QuestionDetail[];
}

/** Teacher-initiated score override forwarded through BFF. */
export interface TeacherScoreOverrideRequest {
  question_id: string;
  new_score: number;
  reason: string;
}

/** Plagiarism flag preview for teacher review. */
export interface PlagiarismPreview {
  flag_id: string;
  student_a_id: string;
  student_b_id: string;
  question_id: string;
  composite_score: number;
  severity: string;
  teacher_verdict?: string;
}

/** Objection summary for the teacher inbox. */
export interface ObjectionInboxItem {
  objection_id: string;
  student_id: string;
  question_id: string;
  status: string;
  filed_at: string;
}

/** Exam card for the student portal list. */
export interface StudentExamCard {
  exam_id: string;
  title: string;
  subject_name?: string;
  scheduled_at: string;
  status: StudentExamStatus;
}

/** Per-question score in the student score view. */
export interface StudentQuestionScore {
  question_id: string;
  marks_obtained: number;
  max_marks: number;
  ai_confidence?: number;
  miss_indicator?: string;
}

/** Student-facing score summary for an exam. */
export interface StudentScoreView {
  exam_id: string;
  total_score: number;
  percentage: number;
  percentile: number;
  pass_fail?: PassFail;
  questions: StudentQuestionScore[];
}

/** Student-facing objection record. */
export interface StudentObjection {
  objection_id: string;
  exam_id: string;
  question_id: string;
  status: ObjectionStatus;
  objection_text?: string;
  resolution_reason?: string;
  new_score?: number;
}

/** Student-initiated objection request. */
export interface CreateStudentObjectionRequest {
  exam_id: string;
  question_id: string;
  objection_text: string;
}
