export type ExamStatus = 'created' | 'armed' | 'timer_running' | 'sync_pending' | 'scoring' | 'finalized' | 'published' | 'locked' | 'cancelled';
export interface Exam {
    exam_id: string;
    title: string;
    subject_id: string;
    class_id: string;
    section_id: string;
    scheduled_at: string;
    duration_min: number;
    total_marks: number;
    question_count: number;
    state: ExamStatus;
    late_entry_cutoff_min?: number;
    objection_window_days?: number;
    variants?: string[];
    created_by: string;
}
export interface ExamVariant {
    variant_label: string;
    exam_id: string;
    question_paper_uri?: string;
}
export interface Rubric {
    exam_id: string;
    question_id: string;
    max_marks: number;
    step_breakdown: RubricStep[];
    expected_answer_type: 'text' | 'formula' | 'diagram';
    auto_score_confidence_threshold: number;
}
export interface RubricStep {
    label: string;
    max_marks: number;
}
export interface QuestionRegion {
    question_id: string;
    exam_id: string;
    page_number: number;
    x: number;
    y: number;
    width: number;
    height: number;
}
export type ScoreStatus = 'ai_draft' | 'teacher_reviewed' | 'finalized' | 'published' | 'objection_window' | 'locked';
export interface Score {
    exam_id: string;
    student_id: string;
    total_score: number;
    max_score: number;
    lifecycle_state: ScoreStatus;
    published_at?: string;
    objection_window_closes_at?: string;
    questions: QuestionScore[];
}
export interface QuestionScore {
    question_id: string;
    ai_score: number;
    current_score: number;
    max_score: number;
    confidence: number;
    override_reason?: string;
    step_scores?: StepScore[];
}
export interface StepScore {
    label: string;
    awarded: number;
    max: number;
}
export interface ScoreOverride {
    teacher_id: string;
    question_id: string;
    new_score: number;
    reason: string;
}
export type ScoreEventType = 'ai_draft_created' | 'override_applied' | 'finalized' | 'published' | 'objection_rescored';
export interface ScoreHistoryItem {
    event_id: string;
    event_type: ScoreEventType;
    old_value: number;
    new_value: number;
    actor_id: string;
    reason?: string;
    created_at: string;
}
export type ObjectionStatus = 'filed' | 'assigned' | 'reviewing' | 'resolved' | 'escalated';
export type ObjectionResolution = 'approved' | 'rejected';
export interface Objection {
    objection_id: string;
    exam_id: string;
    student_id: string;
    question_id: string;
    status: ObjectionStatus;
    objection_text: string;
    filed_at: string;
    assigned_to?: string;
    resolution?: ObjectionResolution;
    resolution_reason?: string;
    score_delta?: number;
}
export type StoodyRole = 'super_admin' | 'principal' | 'hod' | 'tutor' | 'student' | 'parent';
export type ExamPenRole = 'super_admin' | 'principal' | 'hod' | 'tutor' | 'invigilator' | 'evaluator' | 'reviewer' | 'student' | 'parent';
export interface User {
    user_id: string;
    tenant_id: string;
    stoody_role: StoodyRole;
    exampen_roles: ExamPenRole[];
    display_name: string;
    email?: string;
    phone?: string;
    institute_name?: string;
}
export interface StudentBinding {
    exam_id: string;
    pen_mac: string;
    student_id: string;
    student_name?: string;
    student_roll?: string;
    status: 'provisional' | 'confirmed' | 'rejected';
    source: 'registration_scan' | 'manual_register' | 'server_sync';
    bound_at: string;
    server_confirmed_at?: string;
    rejection_reason?: string;
}
//# sourceMappingURL=types.d.ts.map