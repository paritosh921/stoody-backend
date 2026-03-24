"""Analytics models: leaderboard, class stats, performance trends, exports."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from .enums import ExportFormat


class LeaderboardRow(BaseModel):
    """Single row in the exam leaderboard."""

    rank: int
    student_id: str
    student_name: Optional[str] = None
    score: float
    percentile: float


class QuestionDifficulty(BaseModel):
    """Per-question difficulty metric within class stats."""

    question_id: str
    avg_score: float


class ClassStats(BaseModel):
    """Class-level statistical summary for an exam."""

    mean: float
    median: float
    std_dev: float
    pass_rate: float
    question_difficulty: Optional[list[QuestionDifficulty]] = None


class ExamPerformanceEntry(BaseModel):
    """Single exam entry in a student's performance history."""

    exam_id: UUID
    score: float
    percentile: float


class StudentPerformance(BaseModel):
    """Longitudinal performance data for a student."""

    student_id: str
    history: list[ExamPerformanceEntry]
    strengths: Optional[list[str]] = None
    weaknesses: Optional[list[str]] = None


class ExportResult(BaseModel):
    """Metadata for a generated analytics export file."""

    exam_id: UUID
    format: ExportFormat
    download_uri: str


class PerformanceView(BaseModel):
    """Student-facing historical performance and trend data."""

    history: list[ExamPerformanceEntry]
    strengths: list[str]
    weaknesses: list[str]
