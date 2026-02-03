"""
Question models for generated questions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class GenerationSource(str, Enum):
    """Source of question generation."""
    NOTES = "notes"  # Mode 1: Generated from uploaded notes
    TOPIC = "topic"  # Mode 2: Generated from topic search


@dataclass
class QuestionOption:
    """Option for MCQ or matching questions."""
    label: str  # A, B, C, D or 1, 2, 3, 4
    content: str
    is_correct: bool = False


@dataclass
class MarkingStep:
    """A step in the marking scheme."""
    step: str
    marks: float
    criteria: Optional[str] = None


@dataclass
class DiagramSpec:
    """Specification for generating a diagram."""
    subject: str  # physics, chemistry, math, biology
    diagram_type: str  # circuit, graph, molecule, etc.
    title: Optional[str] = None
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for renderer. Flattens parameters to top level."""
        result = {
            "subject": self.subject,
            "diagram_type": self.diagram_type,
            "title": self.title,
            "description": self.description,
        }
        # Flatten parameters to top level for renderer compatibility
        # Renderers expect fields like 'points', 'lines', 'functions' at top level
        if self.parameters:
            result.update(self.parameters)
        return result


@dataclass
class GeneratedQuestion:
    """A fully generated question with all components."""
    
    # Identification
    question_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Question content
    question_text: str = ""
    question_type: str = "mcq"  # QuestionType value
    
    # For MCQ/True-False
    options: List[QuestionOption] = field(default_factory=list)
    correct_answer: Optional[str] = None  # For MCQ: "A", "B", etc.
    
    # For matching
    left_column: List[str] = field(default_factory=list)
    right_column: List[str] = field(default_factory=list)
    matching_answers: Dict[str, str] = field(default_factory=dict)  # {"1": "C", "2": "A"}
    
    # For fill in blanks
    blanks_answers: List[str] = field(default_factory=list)
    
    # Solution and marking
    solution: str = ""
    solution_steps: List[str] = field(default_factory=list)
    marking_scheme: List[MarkingStep] = field(default_factory=list)
    
    # Marks and difficulty
    marks: int = 1
    difficulty: str = "medium"  # DifficultyLevel value
    bloom_level: str = "understand"  # BloomLevel value
    
    # Classification
    subject: str = ""
    chapter: Optional[str] = None
    topic: Optional[str] = None
    
    # Diagram information
    has_diagram: bool = False
    diagram_spec: Optional[DiagramSpec] = None
    diagram_url: Optional[str] = None
    solution_diagram_spec: Optional[DiagramSpec] = None
    solution_diagram_url: Optional[str] = None
    
    # Source tracking
    source_type: GenerationSource = GenerationSource.NOTES
    source_chunk_ids: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "marks": self.marks,
            "difficulty": self.difficulty,
            "bloom_level": self.bloom_level,
            "subject": self.subject,
            "chapter": self.chapter,
            "topic": self.topic,
            "has_diagram": self.has_diagram,
            "created_at": self.created_at.isoformat(),
        }
        
        # Add type-specific fields
        if self.options:
            result["options"] = [
                {"label": o.label, "content": o.content, "is_correct": o.is_correct}
                for o in self.options
            ]
            result["correct_answer"] = self.correct_answer
        
        if self.left_column:
            result["left_column"] = self.left_column
            result["right_column"] = self.right_column
            result["matching_answers"] = self.matching_answers
        
        if self.blanks_answers:
            result["blanks_answers"] = self.blanks_answers
        
        if self.solution:
            result["solution"] = self.solution
            result["solution_steps"] = self.solution_steps
        
        if self.marking_scheme:
            result["marking_scheme"] = [
                {"step": m.step, "marks": m.marks, "criteria": m.criteria}
                for m in self.marking_scheme
            ]
        
        if self.diagram_url:
            result["diagram_url"] = self.diagram_url
        
        if self.diagram_spec:
            result["diagram_spec"] = self.diagram_spec.to_dict()
        
        if self.solution_diagram_url:
            result["solution_diagram_url"] = self.solution_diagram_url
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedQuestion":
        """Create from dictionary (e.g., from LLM response)."""
        question = cls(
            question_id=data.get("question_id", str(uuid.uuid4())),
            question_text=data.get("question_text", ""),
            question_type=data.get("question_type", "mcq"),
            marks=data.get("marks", 1),
            difficulty=data.get("difficulty", "medium"),
            bloom_level=data.get("bloom_level", "understand"),
            subject=data.get("subject", ""),
            chapter=data.get("chapter"),
            topic=data.get("topic"),
            solution=data.get("solution", ""),
            solution_steps=data.get("solution_steps", []),
            has_diagram=data.get("has_diagram", False),
        )
        
        # Parse options
        if "options" in data:
            question.options = [
                QuestionOption(
                    label=o.get("label", ""),
                    content=o.get("content", ""),
                    is_correct=o.get("is_correct", False),
                )
                for o in data["options"]
            ]
            question.correct_answer = data.get("correct_answer")
        
        # Parse matching columns
        if "left_column" in data:
            question.left_column = data["left_column"]
            question.right_column = data.get("right_column", [])
            question.matching_answers = data.get("matching_answers", {})
        
        # Parse blanks
        if "blanks_answers" in data:
            question.blanks_answers = data["blanks_answers"]
        
        # Parse marking scheme
        if "marking_scheme" in data:
            question.marking_scheme = [
                MarkingStep(
                    step=m.get("step", ""),
                    marks=m.get("marks", 0),
                    criteria=m.get("criteria"),
                )
                for m in data["marking_scheme"]
            ]
        
        # Parse diagram spec
        if "diagram_spec" in data and data["diagram_spec"]:
            ds = data["diagram_spec"]
            # Extract known fields
            known_fields = {"subject", "diagram_type", "title", "description", "parameters"}
            # Get explicit parameters or empty dict
            params = ds.get("parameters", {})
            # Add all OTHER fields from diagram_spec to parameters
            # This captures fields like points, lines, functions, x_range, etc.
            for key, value in ds.items():
                if key not in known_fields and value is not None:
                    params[key] = value
            
            question.diagram_spec = DiagramSpec(
                subject=ds.get("subject", ""),
                diagram_type=ds.get("diagram_type", ""),
                title=ds.get("title"),
                description=ds.get("description"),
                parameters=params,
            )
            question.has_diagram = True
        
        return question
