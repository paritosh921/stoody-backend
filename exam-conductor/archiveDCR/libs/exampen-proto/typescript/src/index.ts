/**
 * exampen-proto: Shared TypeScript type definitions for the ExamPen system.
 *
 * Usage:
 *   import type { ExamDetail, ScoreLifecycleState, StrokeRawEvent } from "@exampen/proto";
 */

// --- Enums ---
export type {
  AISourceType,
  AuthoritativeSource,
  BindingSource,
  BindingStatus,
  ChunkBindingStatus,
  CopyAuthoritativeCandidate,
  DongleStatus,
  ExamPenRole,
  ExamState,
  ExportFormat,
  MissIndicatorState,
  ObjectionResolution,
  ObjectionStatus,
  PassFail,
  PenSyncStatus,
  PlagiarismCheckTrigger,
  PlagiarismSeverity,
  ScoreEventType,
  ScoreLifecycleState,
  StudentExamStatus,
  StoodyRole,
  TeacherVerdict,
  TokenStatus,
  UploadPath,
  UploadStatus,
  WebSocketEventType,
} from "./enums";

// --- Exam ---
export type {
  AssignmentRequest,
  BindingRecord,
  ConfirmBindingRequest,
  CreateBindingRequest,
  CreateExamRequest,
  ExamDetail,
  ExamSummary,
  LifecycleTransitionRequest,
  LifecycleTransitionResult,
  PatchExamRequest,
  StudentRef,
} from "./exam";

// --- Stroke ---
export type {
  ExamUploadStatus,
  IngestAck,
  PageAssignment,
  PenUploadStatus,
  StrokeChunkUploadRequest,
} from "./stroke";

// --- Score ---
export type {
  FinalizeRequest,
  PublishRequest,
  QuestionScore,
  ScoreHistoryItem,
  ScoreOverrideRequest,
  StepScore,
  StudentScoreDetail,
  WorkflowStateResponse,
} from "./score";

// --- User ---
export type {
  ErrorResponse,
  IntrospectRequest,
  NormalizedClaims,
  Profile,
  RevocationRequest,
  RevocationStatus,
} from "./user";

// --- Page ---
export type {
  CopyPage,
  CopyUploadResult,
  MissIndicatorCell,
  MissIndicatorMatrix,
} from "./page";

// --- AI ---
export type { AnswerInsight, QuestionResult } from "./ai";

// --- Plagiarism ---
export type {
  Evidence,
  FlagDetail,
  FlagSummary,
  MatchingSegment,
  PlagiarismFlagEvent,
  VerdictRequest,
} from "./plagiarism";

// --- Objection ---
export type {
  CreateObjectionRequest,
  EscalateObjectionRequest,
  ObjectionDetail,
  ObjectionSummary,
  ResolveObjectionRequest,
} from "./objection";

// --- Analytics ---
export type {
  ClassStats,
  ExamPerformanceEntry,
  ExportResult,
  LeaderboardRow,
  PerformanceView,
  QuestionDifficulty,
  StudentPerformance,
} from "./analytics";

// --- Chat ---
export type {
  ChatMessage,
  Message,
  ReadReceipt,
  SendChatMessageRequest,
  SendMessageRequest,
} from "./chat";

// --- Hub ---
export type {
  DongleRow,
  PenSyncRow,
  SessionSummary,
  WebSocketEnvelope,
} from "./hub";

// --- BFF ---
export type {
  ClassScoreRow,
  CreateStudentObjectionRequest,
  ObjectionInboxItem,
  PlagiarismPreview,
  QuestionDetail,
  StudentExamCard,
  StudentObjection,
  StudentQuestionScore,
  StudentScoreView,
  TeacherExamCard,
  TeacherScoreOverrideRequest,
  TeacherStudentDetail,
} from "./bff";

// --- Events ---
export type {
  AIResultEvent,
  CopyReadyEvent,
  ExamLifecycleEvent,
  ExamPenEvent,
  ObjectionEvent,
  PageReadyEvent,
  PlagiarismCheckEvent,
  PlagiarismResultEvent,
  ScoreUpdatedEvent,
  StrokeProcessedEvent,
  StrokeRawEvent,
} from "./events";
