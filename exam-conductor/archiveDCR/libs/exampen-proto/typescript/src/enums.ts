/** Shared enum definitions for ExamPen domain types. */

/** Exam lifecycle FSM states (svc-exam-orch authoritative). */
export type ExamState =
  | "created"
  | "armed"
  | "timer_running"
  | "sync_pending"
  | "scoring"
  | "finalized"
  | "published"
  | "locked"
  | "cancelled";

/** Score lifecycle FSM states (svc-score-engine authoritative). */
export type ScoreLifecycleState =
  | "ai_draft"
  | "teacher_reviewed"
  | "finalized"
  | "published"
  | "objection_window"
  | "locked";

/** Objection lifecycle FSM states (svc-review authoritative). */
export type ObjectionStatus =
  | "filed"
  | "assigned"
  | "reviewing"
  | "resolved"
  | "escalated";

/** Roles originating from Stoody identity system. */
export type StoodyRole =
  | "super_admin"
  | "principal"
  | "hod"
  | "tutor"
  | "student"
  | "parent";

/** ExamPen-specific roles mapped from Stoody roles. */
export type ExamPenRole =
  | "super_admin"
  | "principal"
  | "hod"
  | "tutor"
  | "invigilator"
  | "evaluator"
  | "reviewer"
  | "student"
  | "parent";

/** JWT introspection result status. */
export type TokenStatus = "valid" | "revoked";

/** Pen-student binding status. */
export type BindingStatus = "provisional" | "confirmed" | "rejected";

/** How a pen binding was created. */
export type BindingSource =
  | "registration_scan"
  | "manual_register"
  | "server_sync";

/** How stroke data reached the server. */
export type UploadPath = "wifi" | "mobile";

/** Hub-to-server upload progress status. */
export type UploadStatus = "pending" | "in_progress" | "complete" | "partial";

/** Per-pen BLE sync status. */
export type PenSyncStatus =
  | "pending"
  | "connecting"
  | "syncing"
  | "complete"
  | "failed"
  | "timeout";

/** BLE dongle health status. */
export type DongleStatus = "healthy" | "degraded" | "failed";

/** Plagiarism detection severity level. */
export type PlagiarismSeverity = "review_recommended" | "strong_match";

/** Teacher verdict on a plagiarism flag. */
export type TeacherVerdict =
  | "pending"
  | "confirmed_plagiarism"
  | "dismissed";

/** How an objection was resolved. */
export type ObjectionResolution = "approved" | "rejected";

/** Score audit history event types. */
export type ScoreEventType =
  | "ai_draft_created"
  | "override_applied"
  | "finalized"
  | "published"
  | "objection_rescored";

/** Authoritative data source for a page. */
export type AuthoritativeSource = "strokes" | "copy_image" | "both";

/** Source type for AI recognition input. */
export type AISourceType = "strokes" | "copy_image";

/** Analytics export file format. */
export type ExportFormat = "csv" | "pdf";

/** Question miss indicator state in the teacher matrix. */
export type MissIndicatorState =
  | "answered"
  | "miss_no_strokes"
  | "miss_sync_failure"
  | "miss_pen_inactive"
  | "not_attempted_confirmed";

/** Student-facing exam status. */
export type StudentExamStatus =
  | "upcoming"
  | "scores_pending"
  | "published"
  | "objection_window_open"
  | "locked";

/** Pass/fail status for a student exam result. */
export type PassFail = "pass" | "fail";

/** What triggered a plagiarism check. */
export type PlagiarismCheckTrigger =
  | "all_ai_results_ready"
  | "manual_recheck";

/** Whether a copy image is an authoritative candidate. */
export type CopyAuthoritativeCandidate = "copy_image" | "comparison_only";

/** WebSocket event types for invigilator console. */
export type WebSocketEventType =
  | "session.snapshot"
  | "sync.progress"
  | "dongle.health"
  | "upload.progress";

/** Binding status as reported with a stroke chunk upload. */
export type ChunkBindingStatus =
  | "unknown"
  | "provisional"
  | "confirmed"
  | "rejected";
