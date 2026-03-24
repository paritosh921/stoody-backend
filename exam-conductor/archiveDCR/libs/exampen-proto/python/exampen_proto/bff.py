"""BFF view models: teacher and student aggregation surfaces."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import ObjectionStatus, PassFail, StudentExamStatus


class TeacherExamCard(BaseModel):
    """Exam card for the teacher dashboard list."""

    exam_id: UUID
    title: str
    subject_id: str
    scheduled_at: datetime
    state: str
    class_label: Optional[str] = None


class ClassScoreRow(BaseModel):
    """One row in the teacher class score grid."""

    student_id: str
    student_name: str
    total_score: float
    percentile: Optional[float] = None
    ai_confidence: float
    miss_indicator_count: Optional[int] = None
    plagiarism_flag_count: Optional[int] = None


class QuestionDetail(BaseModel):
    """Per-question detail for teacher drill-down."""

    question_id: str
    current_score: float
    confidence: float
    recognized_text: Optional[str] = None
    miss_indicator: Optional[str] = None
    copy_image_uri: Optional[str] = None


class TeacherStudentDetail(BaseModel):
    """Teacher drill-down view for one student's exam."""

    student_id: str
    student_name: str
    total_score: float
    answer_pages: Optional[list[str]] = None
    questions: list[QuestionDetail]


class TeacherScoreOverrideRequest(BaseModel):
    """Teacher-initiated score override forwarded through BFF."""

    question_id: str
    new_score: float
    reason: str


class PlagiarismPreview(BaseModel):
    """Plagiarism flag preview for teacher review."""

    flag_id: UUID
    student_a_id: str
    student_b_id: str
    question_id: str
    composite_score: float
    severity: str
    teacher_verdict: Optional[str] = None


class ObjectionInboxItem(BaseModel):
    """Objection summary for the teacher inbox."""

    objection_id: UUID
    student_id: str
    question_id: str
    status: str
    filed_at: datetime


class StudentExamCard(BaseModel):
    """Exam card for the student portal list."""

    exam_id: UUID
    title: str
    subject_name: Optional[str] = None
    scheduled_at: datetime
    status: StudentExamStatus


class StudentQuestionScore(BaseModel):
    """Per-question score in the student score view."""

    question_id: str
    marks_obtained: float
    max_marks: float
    ai_confidence: Optional[float] = None
    miss_indicator: Optional[str] = None


class StudentScoreView(BaseModel):
    """Student-facing score summary for an exam."""

    exam_id: UUID
    total_score: float
    percentage: float
    percentile: float
    pass_fail: Optional[PassFail] = None
    questions: list[StudentQuestionScore]


class StudentObjection(BaseModel):
    """Student-facing objection record."""

    objection_id: UUID
    exam_id: UUID
    question_id: str
    status: ObjectionStatus
    objection_text: Optional[str] = None
    resolution_reason: Optional[str] = None
    new_score: Optional[float] = None


class CreateStudentObjectionRequest(BaseModel):
    """Student-initiated objection request."""

    exam_id: UUID
    question_id: str
    objection_text: str
