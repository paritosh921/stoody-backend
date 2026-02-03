"""
Paper models for assembled question papers.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .question import GeneratedQuestion
from .config import PaperConfig, QuestionGenerationConfig


class PaperStatus(str, Enum):
    """Status of a generated paper."""
    GENERATING = "generating"
    DRAFT = "draft"
    COMPLETED = "completed"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class PaperSection:
    """A section within the paper."""
    
    name: str  # "Section A - Multiple Choice Questions"
    instructions: str  # Section-specific instructions
    questions: List[GeneratedQuestion] = field(default_factory=list)
    
    @property
    def total_marks(self) -> int:
        return sum(q.marks for q in self.questions)
    
    @property
    def question_count(self) -> int:
        return len(self.questions)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "question_count": self.question_count,
            "total_marks": self.total_marks,
            "questions": [q.to_dict() for q in self.questions],
        }


@dataclass
class GeneratedPaper:
    """A complete generated question paper."""
    
    # Identification
    paper_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    teacher_id: str = ""
    
    # Paper configuration
    paper_config: Optional[PaperConfig] = None
    generation_config: Optional[QuestionGenerationConfig] = None
    
    # Content
    sections: List[PaperSection] = field(default_factory=list)
    
    # Source information
    source_type: str = "notes"  # "notes" or "topic"
    source_upload_ids: List[str] = field(default_factory=list)  # For Mode 1
    source_topic: Optional[str] = None  # For Mode 2
    source_subject: Optional[str] = None
    
    # Generated files
    question_paper_url: Optional[str] = None
    answer_key_url: Optional[str] = None
    marking_scheme_url: Optional[str] = None
    
    # Status and metadata
    status: PaperStatus = PaperStatus.GENERATING
    error_message: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Statistics
    generation_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_questions(self) -> int:
        return sum(section.question_count for section in self.sections)
    
    @property
    def total_marks(self) -> int:
        return sum(section.total_marks for section in self.sections)
    
    @property
    def title(self) -> str:
        return self.paper_config.title if self.paper_config else "Untitled Paper"
    
    @property
    def subject(self) -> str:
        return self.paper_config.subject if self.paper_config else self.source_subject or ""
    
    @property
    def grade(self) -> str:
        return self.paper_config.grade if self.paper_config else ""
    
    @property
    def duration_minutes(self) -> int:
        return self.paper_config.duration_minutes if self.paper_config else 90
    
    def get_all_questions(self) -> List[GeneratedQuestion]:
        """Get all questions from all sections."""
        questions = []
        for section in self.sections:
            questions.extend(section.questions)
        return questions
    
    def get_questions_by_type(self, question_type: str) -> List[GeneratedQuestion]:
        """Get all questions of a specific type."""
        return [q for q in self.get_all_questions() if q.question_type == question_type]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[GeneratedQuestion]:
        """Get all questions of a specific difficulty."""
        return [q for q in self.get_all_questions() if q.difficulty == difficulty]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "paper_id": self.paper_id,
            "tenant_id": self.tenant_id,
            "teacher_id": self.teacher_id,
            "title": self.title,
            "subject": self.subject,
            "grade": self.grade,
            "duration_minutes": self.duration_minutes,
            "total_questions": self.total_questions,
            "total_marks": self.total_marks,
            "status": self.status.value,
            "source_type": self.source_type,
            "created_at": self.created_at.isoformat(),
        }
        
        if self.paper_config:
            result["paper_config"] = {
                "title": self.paper_config.title,
                "subject": self.paper_config.subject,
                "grade": self.paper_config.grade,
                "duration_minutes": self.paper_config.duration_minutes,
                "school_name": self.paper_config.school_name,
                "exam_name": self.paper_config.exam_name,
                "chapter": self.paper_config.chapter,
                "general_instructions": self.paper_config.general_instructions,
            }
        
        if self.sections:
            result["sections"] = [s.to_dict() for s in self.sections]
        
        if self.question_paper_url:
            result["question_paper_url"] = self.question_paper_url
        
        if self.answer_key_url:
            result["answer_key_url"] = self.answer_key_url
        
        if self.marking_scheme_url:
            result["marking_scheme_url"] = self.marking_scheme_url
        
        if self.source_upload_ids:
            result["source_upload_ids"] = self.source_upload_ids
        
        if self.source_topic:
            result["source_topic"] = self.source_topic
        
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
        
        if self.error_message:
            result["error_message"] = self.error_message
        
        if self.generation_stats:
            result["generation_stats"] = self.generation_stats
        
        return result
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary (without full questions)."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "subject": self.subject,
            "grade": self.grade,
            "total_questions": self.total_questions,
            "total_marks": self.total_marks,
            "duration_minutes": self.duration_minutes,
            "status": self.status.value,
            "source_type": self.source_type,
            "created_at": self.created_at.isoformat(),
            "question_paper_url": self.question_paper_url,
            "answer_key_url": self.answer_key_url,
        }
    
    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document format."""
        result = {
            "_id": self.paper_id,
            "tenant_id": self.tenant_id,
            "teacher_id": self.teacher_id,
            "status": self.status.value,
            "source_type": self.source_type,
            "source_upload_ids": self.source_upload_ids,
            "source_topic": self.source_topic,
            "source_subject": self.source_subject,
            "question_paper_url": self.question_paper_url,
            "answer_key_url": self.answer_key_url,
            "marking_scheme_url": self.marking_scheme_url,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "generation_stats": self.generation_stats,
        }
        
        if self.paper_config:
            result["paper_config"] = {
                "title": self.paper_config.title,
                "subject": self.paper_config.subject,
                "grade": self.paper_config.grade,
                "duration_minutes": self.paper_config.duration_minutes,
                "school_name": self.paper_config.school_name,
                "exam_name": self.paper_config.exam_name,
                "exam_date": self.paper_config.exam_date,
                "chapter": self.paper_config.chapter,
                "topics": self.paper_config.topics,
                "general_instructions": self.paper_config.general_instructions,
            }
        
        if self.generation_config:
            result["generation_config"] = {
                "question_types": {
                    k.value: {"count": v.count, "marks_per_question": v.marks_per_question}
                    for k, v in self.generation_config.question_types.items()
                },
                "difficulty_distribution": {
                    "easy": self.generation_config.difficulty_distribution.easy,
                    "medium": self.generation_config.difficulty_distribution.medium,
                    "hard": self.generation_config.difficulty_distribution.hard,
                },
                "bloom_levels": [b.value for b in self.generation_config.bloom_levels],
                "include_diagrams": self.generation_config.include_diagrams,
                "include_solutions": self.generation_config.include_solutions,
                "include_marking_scheme": self.generation_config.include_marking_scheme,
            }
        
        if self.sections:
            result["sections"] = [s.to_dict() for s in self.sections]
        
        return result
    
    @classmethod
    def from_mongo_dict(cls, doc: Dict[str, Any]) -> "GeneratedPaper":
        """Create GeneratedPaper from MongoDB document."""
        from .config import PaperConfig, QuestionGenerationConfig, QuestionTypeConfig, DifficultyDistribution, QuestionType, BloomLevel
        from .question import GeneratedQuestion, QuestionOption, MarkingStep, DiagramSpec
        
        paper = cls(
            paper_id=doc["_id"],
            tenant_id=doc.get("tenant_id", ""),
            teacher_id=doc.get("teacher_id", ""),
            source_type=doc.get("source_type", "notes"),
            source_upload_ids=doc.get("source_upload_ids", []),
            source_topic=doc.get("source_topic"),
            source_subject=doc.get("source_subject"),
            question_paper_url=doc.get("question_paper_url"),
            answer_key_url=doc.get("answer_key_url"),
            marking_scheme_url=doc.get("marking_scheme_url"),
            status=PaperStatus(doc.get("status", "generating")),
            error_message=doc.get("error_message"),
            created_at=doc.get("created_at", datetime.utcnow()),
            completed_at=doc.get("completed_at"),
            generation_stats=doc.get("generation_stats", {}),
        )
        
        # Reconstruct paper_config
        if "paper_config" in doc and doc["paper_config"]:
            pc = doc["paper_config"]
            paper.paper_config = PaperConfig(
                title=pc.get("title", ""),
                subject=pc.get("subject", ""),
                grade=pc.get("grade", ""),
                duration_minutes=pc.get("duration_minutes", 90),
                school_name=pc.get("school_name"),
                exam_name=pc.get("exam_name"),
                exam_date=pc.get("exam_date"),
                chapter=pc.get("chapter"),
                topics=pc.get("topics"),
                general_instructions=pc.get("general_instructions"),
            )
        
        # Reconstruct generation_config
        if "generation_config" in doc and doc["generation_config"]:
            gc = doc["generation_config"]
            question_types = {}
            for qt_str, qt_config in gc.get("question_types", {}).items():
                question_types[QuestionType(qt_str)] = QuestionTypeConfig(
                    count=qt_config.get("count", 0),
                    marks_per_question=qt_config.get("marks_per_question", 1)
                )
            
            dd = gc.get("difficulty_distribution", {})
            difficulty_dist = DifficultyDistribution(
                easy=dd.get("easy", 30),
                medium=dd.get("medium", 50),
                hard=dd.get("hard", 20),
            )
            
            bloom_levels = [BloomLevel(b) for b in gc.get("bloom_levels", [])]
            
            paper.generation_config = QuestionGenerationConfig(
                question_types=question_types,
                difficulty_distribution=difficulty_dist,
                bloom_levels=bloom_levels,
                include_diagrams=gc.get("include_diagrams", True),
                include_solutions=gc.get("include_solutions", True),
                include_marking_scheme=gc.get("include_marking_scheme", True),
            )
        
        # Reconstruct sections
        if "sections" in doc and doc["sections"]:
            for section_data in doc["sections"]:
                questions = []
                for q_data in section_data.get("questions", []):
                    # Reconstruct options
                    options = []
                    for opt in q_data.get("options", []):
                        options.append(QuestionOption(
                            label=opt.get("label", ""),
                            content=opt.get("content", ""),
                            is_correct=opt.get("is_correct", False),
                        ))
                    
                    # Reconstruct marking scheme
                    marking_scheme = []
                    for step in q_data.get("marking_scheme", []):
                        marking_scheme.append(MarkingStep(
                            step_number=step.get("step_number", 0),
                            description=step.get("description", ""),
                            marks=step.get("marks", 0),
                        ))
                    
                    # Reconstruct diagram spec
                    diagram_spec = None
                    if q_data.get("diagram_spec"):
                        ds = q_data["diagram_spec"]
                        diagram_spec = DiagramSpec(
                            diagram_type=ds.get("diagram_type", ""),
                            description=ds.get("description", ""),
                            elements=ds.get("elements", []),
                            labels=ds.get("labels", {}),
                        )
                    
                    question = GeneratedQuestion(
                        question_id=q_data.get("question_id", ""),
                        question_text=q_data.get("question_text", ""),
                        question_type=q_data.get("question_type", ""),
                        options=options,
                        correct_answer=q_data.get("correct_answer"),
                        solution=q_data.get("solution", ""),
                        marking_scheme=marking_scheme,
                        marks=q_data.get("marks", 1),
                        difficulty=q_data.get("difficulty", "medium"),
                        bloom_level=q_data.get("bloom_level"),
                        topic=q_data.get("topic"),
                        has_diagram=q_data.get("has_diagram", False),
                        diagram_spec=diagram_spec,
                        diagram_url=q_data.get("diagram_url"),
                        source_chunks=q_data.get("source_chunks", []),
                    )
                    questions.append(question)
                
                section = PaperSection(
                    name=section_data.get("name", ""),
                    instructions=section_data.get("instructions", ""),
                    questions=questions,
                )
                paper.sections.append(section)
        
        return paper
