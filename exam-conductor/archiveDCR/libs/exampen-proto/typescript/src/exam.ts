/** Exam domain types: exam definitions, lifecycle, variants, bindings. */

import type { BindingSource, BindingStatus, ExamState } from "./enums";

/** Lightweight exam record for list views. */
export interface ExamSummary {
  exam_id: string;
  subject_id: string;
  class_id: string;
  title?: string;
  scheduled_at: string;
  duration_min?: number;
  state: ExamState;
}

/** Full exam configuration including rubric metadata. */
export interface ExamDetail extends ExamSummary {
  section_id: string;
  total_marks: number;
  question_count: number;
  late_entry_cutoff_min?: number;
  objection_window_days?: number;
  variants?: string[];
  created_by: string;
}

/** Request body for creating a new exam. */
export interface CreateExamRequest {
  title: string;
  subject_id: string;
  class_id: string;
  section_id: string;
  scheduled_at: string;
  duration_min: number;
  question_count: number;
  total_marks: number;
  negative_marking?: boolean;
  variants?: string[];
}

/** Partial update for mutable exam fields (before lock). */
export interface PatchExamRequest {
  scheduled_at?: string;
  duration_min?: number;
  objection_window_days?: number;
  late_entry_cutoff_min?: number;
}

/** Request to transition an exam to a new lifecycle state. */
export interface LifecycleTransitionRequest {
  to_state: ExamState;
  actor_id: string;
  reason?: string;
}

/** Result of applying a lifecycle state transition. */
export interface LifecycleTransitionResult {
  exam_id: string;
  from_state: string;
  to_state: string;
  changed_at: string;
}

/** Assign invigilators and evaluators to an exam. */
export interface AssignmentRequest {
  invigilator_ids: string[];
  evaluator_ids: string[];
  double_blind?: boolean;
}

/** Reference to a student from the Stoody roster. */
export interface StudentRef {
  student_id: string;
  name: string;
  roll?: string;
  section_id?: string;
}

/** Request to bind a pen to a student for an exam. */
export interface CreateBindingRequest {
  pen_mac: string;
  student_id: string;
  source: BindingSource;
  request_id?: string;
}

/** Confirm or reject a provisional pen-student binding. */
export interface ConfirmBindingRequest {
  status: BindingStatus;
  rejection_reason?: string;
}

/** Server-side pen-student binding record. */
export interface BindingRecord {
  exam_id: string;
  pen_mac: string;
  student_id?: string;
  student_name?: string;
  student_roll?: string;
  status: BindingStatus;
  source: BindingSource;
  bound_at: string;
  server_confirmed_at?: string;
  rejection_reason?: string;
}
