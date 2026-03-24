/** Plagiarism types: flags, evidence, verdicts, check/result events. */

import type {
  PlagiarismCheckTrigger,
  PlagiarismSeverity,
  TeacherVerdict,
} from "./enums";

/** Pair of matching text segments between two students. */
export interface MatchingSegment {
  student_a_text: string;
  student_b_text: string;
}

/** Plagiarism detection evidence for a flag. */
export interface Evidence {
  matching_segments: MatchingSegment[];
  temporal_correlation_score?: number;
  seating_proximity_score?: number;
}

/** Lightweight plagiarism flag for list views. */
export interface FlagSummary {
  flag_id: string;
  exam_id: string;
  student_a_id: string;
  student_b_id: string;
  question_id: string;
  composite_score: number;
  severity: PlagiarismSeverity;
  teacher_verdict?: TeacherVerdict;
}

/** Full plagiarism flag with evidence and verdict detail. */
export interface FlagDetail extends FlagSummary {
  evidence: Evidence;
  verdict_reason?: string;
  verdict_by?: string;
  verdict_at?: string;
}

/** Teacher verdict on a plagiarism flag. */
export interface VerdictRequest {
  teacher_id: string;
  verdict: TeacherVerdict;
  reason: string;
}

/** Single flag entry within a plagiarism.result event. */
export interface PlagiarismFlagEvent {
  flag_id: string;
  student_a_id: string;
  student_b_id: string;
  question_id: string;
  composite_score: number;
  severity: PlagiarismSeverity;
}

/** NATS event: plagiarism check requested. */
export interface PlagiarismCheckEvent {
  event_id: string;
  event_type: "plagiarism.check";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  student_count: number;
  question_count: number;
  trigger: PlagiarismCheckTrigger;
}

/** NATS event: plagiarism check completed with flags. */
export interface PlagiarismResultEvent {
  event_id: string;
  event_type: "plagiarism.result";
  event_version: "1.0.0";
  occurred_at: string;
  exam_id: string;
  flags: PlagiarismFlagEvent[];
}
