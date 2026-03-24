"""Shared enum definitions for ExamPen domain models."""

from enum import Enum


class ExamState(str, Enum):
    """Exam lifecycle FSM states (svc-exam-orch authoritative)."""

    CREATED = "created"
    ARMED = "armed"
    TIMER_RUNNING = "timer_running"
    SYNC_PENDING = "sync_pending"
    SCORING = "scoring"
    FINALIZED = "finalized"
    PUBLISHED = "published"
    LOCKED = "locked"
    CANCELLED = "cancelled"


class ScoreLifecycleState(str, Enum):
    """Score lifecycle FSM states (svc-score-engine authoritative)."""

    AI_DRAFT = "ai_draft"
    TEACHER_REVIEWED = "teacher_reviewed"
    FINALIZED = "finalized"
    PUBLISHED = "published"
    OBJECTION_WINDOW = "objection_window"
    LOCKED = "locked"


class ObjectionStatus(str, Enum):
    """Objection lifecycle FSM states (svc-review authoritative)."""

    FILED = "filed"
    ASSIGNED = "assigned"
    REVIEWING = "reviewing"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class StoodyRole(str, Enum):
    """Roles originating from Stoody identity system."""

    SUPER_ADMIN = "super_admin"
    PRINCIPAL = "principal"
    HOD = "hod"
    TUTOR = "tutor"
    STUDENT = "student"
    PARENT = "parent"


class ExamPenRole(str, Enum):
    """ExamPen-specific roles mapped from Stoody roles."""

    SUPER_ADMIN = "super_admin"
    PRINCIPAL = "principal"
    HOD = "hod"
    TUTOR = "tutor"
    INVIGILATOR = "invigilator"
    EVALUATOR = "evaluator"
    REVIEWER = "reviewer"
    STUDENT = "student"
    PARENT = "parent"


class TokenStatus(str, Enum):
    """JWT introspection result status."""

    VALID = "valid"
    REVOKED = "revoked"


class BindingStatus(str, Enum):
    """Pen-student binding status."""

    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class BindingSource(str, Enum):
    """How a pen binding was created."""

    REGISTRATION_SCAN = "registration_scan"
    MANUAL_REGISTER = "manual_register"
    SERVER_SYNC = "server_sync"


class UploadPath(str, Enum):
    """How stroke data reached the server."""

    WIFI = "wifi"
    MOBILE = "mobile"


class UploadStatus(str, Enum):
    """Hub-to-server upload progress status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    PARTIAL = "partial"


class PenSyncStatus(str, Enum):
    """Per-pen BLE sync status."""

    PENDING = "pending"
    CONNECTING = "connecting"
    SYNCING = "syncing"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"


class DongleStatus(str, Enum):
    """BLE dongle health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class PlagiarismSeverity(str, Enum):
    """Plagiarism detection severity level."""

    REVIEW_RECOMMENDED = "review_recommended"
    STRONG_MATCH = "strong_match"


class TeacherVerdict(str, Enum):
    """Teacher verdict on a plagiarism flag."""

    PENDING = "pending"
    CONFIRMED_PLAGIARISM = "confirmed_plagiarism"
    DISMISSED = "dismissed"


class ObjectionResolution(str, Enum):
    """How an objection was resolved."""

    APPROVED = "approved"
    REJECTED = "rejected"


class ScoreEventType(str, Enum):
    """Score audit history event types."""

    AI_DRAFT_CREATED = "ai_draft_created"
    OVERRIDE_APPLIED = "override_applied"
    FINALIZED = "finalized"
    PUBLISHED = "published"
    OBJECTION_RESCORED = "objection_rescored"


class AuthoritativeSource(str, Enum):
    """Authoritative data source for a page."""

    STROKES = "strokes"
    COPY_IMAGE = "copy_image"
    BOTH = "both"


class AISourceType(str, Enum):
    """Source type for AI recognition input."""

    STROKES = "strokes"
    COPY_IMAGE = "copy_image"


class ExportFormat(str, Enum):
    """Analytics export file format."""

    CSV = "csv"
    PDF = "pdf"


class MissIndicatorState(str, Enum):
    """Question miss indicator state in the teacher matrix."""

    ANSWERED = "answered"
    MISS_NO_STROKES = "miss_no_strokes"
    MISS_SYNC_FAILURE = "miss_sync_failure"
    MISS_PEN_INACTIVE = "miss_pen_inactive"
    NOT_ATTEMPTED_CONFIRMED = "not_attempted_confirmed"


class StudentExamStatus(str, Enum):
    """Student-facing exam status."""

    UPCOMING = "upcoming"
    SCORES_PENDING = "scores_pending"
    PUBLISHED = "published"
    OBJECTION_WINDOW_OPEN = "objection_window_open"
    LOCKED = "locked"


class PassFail(str, Enum):
    """Pass/fail status for a student exam result."""

    PASS = "pass"
    FAIL = "fail"


class PlagiarismCheckTrigger(str, Enum):
    """What triggered a plagiarism check."""

    ALL_AI_RESULTS_READY = "all_ai_results_ready"
    MANUAL_RECHECK = "manual_recheck"


class CopyAuthoritativeCandidate(str, Enum):
    """Whether a copy image is an authoritative candidate."""

    COPY_IMAGE = "copy_image"
    COMPARISON_ONLY = "comparison_only"


class WebSocketEventType(str, Enum):
    """WebSocket event types for invigilator console."""

    SESSION_SNAPSHOT = "session.snapshot"
    SYNC_PROGRESS = "sync.progress"
    DONGLE_HEALTH = "dongle.health"
    UPLOAD_PROGRESS = "upload.progress"


class ChunkBindingStatus(str, Enum):
    """Binding status as reported with a stroke chunk upload."""

    UNKNOWN = "unknown"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
