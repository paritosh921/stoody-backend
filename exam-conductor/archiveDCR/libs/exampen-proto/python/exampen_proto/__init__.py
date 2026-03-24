"""exampen-proto: Shared Pydantic v2 models for the ExamPen system.

Usage:
    from exampen_proto import ExamDetail, ScoreLifecycleState, StrokeRawEvent
    from exampen_proto.enums import ExamState
    from exampen_proto.exam import CreateExamRequest
"""

# --- Enums ---
from .enums import (
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
)

# --- Exam ---
from .exam import (
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
)

# --- Stroke ---
from .stroke import (
    ExamUploadStatus,
    IngestAck,
    PageAssignment,
    PenUploadStatus,
    StrokeChunkUploadRequest,
)

# --- Score ---
from .score import (
    FinalizeRequest,
    PublishRequest,
    QuestionScore,
    ScoreHistoryItem,
    ScoreOverrideRequest,
    StepScore,
    StudentScoreDetail,
    WorkflowStateResponse,
)

# --- User ---
from .user import (
    ErrorResponse,
    IntrospectRequest,
    NormalizedClaims,
    Profile,
    RevocationRequest,
    RevocationStatus,
)

# --- Page ---
from .page import (
    CopyPage,
    CopyUploadResult,
    MissIndicatorCell,
    MissIndicatorMatrix,
)

# --- AI ---
from .ai import AnswerInsight, QuestionResult

# --- Plagiarism ---
from .plagiarism import (
    Evidence,
    FlagDetail,
    FlagSummary,
    MatchingSegment,
    PlagiarismFlagEvent,
    VerdictRequest,
)

# --- Objection ---
from .objection import (
    CreateObjectionRequest,
    EscalateObjectionRequest,
    ObjectionDetail,
    ObjectionSummary,
    ResolveObjectionRequest,
)

# --- Analytics ---
from .analytics import (
    ClassStats,
    ExamPerformanceEntry,
    ExportResult,
    LeaderboardRow,
    PerformanceView,
    QuestionDifficulty,
    StudentPerformance,
)

# --- Chat ---
from .chat import (
    ChatMessage,
    Message,
    ReadReceipt,
    SendChatMessageRequest,
    SendMessageRequest,
)

# --- Hub ---
from .hub import (
    DongleRow,
    PenSyncRow,
    SessionSummary,
    WebSocketEnvelope,
)

# --- BFF ---
from .bff import (
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
)

# --- Events ---
from .events import (
    AIResultEvent,
    CopyReadyEvent,
    ExamLifecycleEvent,
    ObjectionEvent,
    PageReadyEvent,
    PlagiarismCheckEvent,
    PlagiarismResultEvent,
    ScoreUpdatedEvent,
    StrokeProcessedEvent,
    StrokeRawEvent,
)

__all__ = [
    # Enums
    "AISourceType",
    "AuthoritativeSource",
    "BindingSource",
    "BindingStatus",
    "ChunkBindingStatus",
    "CopyAuthoritativeCandidate",
    "DongleStatus",
    "ExamPenRole",
    "ExamState",
    "ExportFormat",
    "MissIndicatorState",
    "ObjectionResolution",
    "ObjectionStatus",
    "PassFail",
    "PenSyncStatus",
    "PlagiarismCheckTrigger",
    "PlagiarismSeverity",
    "ScoreEventType",
    "ScoreLifecycleState",
    "StudentExamStatus",
    "StoodyRole",
    "TeacherVerdict",
    "TokenStatus",
    "UploadPath",
    "UploadStatus",
    "WebSocketEventType",
    # Exam
    "AssignmentRequest",
    "BindingRecord",
    "ConfirmBindingRequest",
    "CreateBindingRequest",
    "CreateExamRequest",
    "ExamDetail",
    "ExamSummary",
    "LifecycleTransitionRequest",
    "LifecycleTransitionResult",
    "PatchExamRequest",
    "StudentRef",
    # Stroke
    "ExamUploadStatus",
    "IngestAck",
    "PageAssignment",
    "PenUploadStatus",
    "StrokeChunkUploadRequest",
    # Score
    "FinalizeRequest",
    "PublishRequest",
    "QuestionScore",
    "ScoreHistoryItem",
    "ScoreOverrideRequest",
    "StepScore",
    "StudentScoreDetail",
    "WorkflowStateResponse",
    # User
    "ErrorResponse",
    "IntrospectRequest",
    "NormalizedClaims",
    "Profile",
    "RevocationRequest",
    "RevocationStatus",
    # Page
    "CopyPage",
    "CopyUploadResult",
    "MissIndicatorCell",
    "MissIndicatorMatrix",
    # AI
    "AnswerInsight",
    "QuestionResult",
    # Plagiarism
    "Evidence",
    "FlagDetail",
    "FlagSummary",
    "MatchingSegment",
    "PlagiarismFlagEvent",
    "VerdictRequest",
    # Objection
    "CreateObjectionRequest",
    "EscalateObjectionRequest",
    "ObjectionDetail",
    "ObjectionSummary",
    "ResolveObjectionRequest",
    # Analytics
    "ClassStats",
    "ExamPerformanceEntry",
    "ExportResult",
    "LeaderboardRow",
    "PerformanceView",
    "QuestionDifficulty",
    "StudentPerformance",
    # Chat
    "ChatMessage",
    "Message",
    "ReadReceipt",
    "SendChatMessageRequest",
    "SendMessageRequest",
    # Hub
    "DongleRow",
    "PenSyncRow",
    "SessionSummary",
    "WebSocketEnvelope",
    # BFF
    "ClassScoreRow",
    "CreateStudentObjectionRequest",
    "ObjectionInboxItem",
    "PlagiarismPreview",
    "QuestionDetail",
    "StudentExamCard",
    "StudentObjection",
    "StudentQuestionScore",
    "StudentScoreView",
    "TeacherExamCard",
    "TeacherScoreOverrideRequest",
    "TeacherStudentDetail",
    # Events
    "AIResultEvent",
    "CopyReadyEvent",
    "ExamLifecycleEvent",
    "ObjectionEvent",
    "PageReadyEvent",
    "PlagiarismCheckEvent",
    "PlagiarismResultEvent",
    "ScoreUpdatedEvent",
    "StrokeProcessedEvent",
    "StrokeRawEvent",
]
