"""
Question Planner Service

Creates a detailed plan for question generation based on extracted concepts.
Ensures variety, coverage, and appropriate difficulty distribution.
"""

import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .content_analyzer import ContentAnalysis, ExtractedConcept, ConceptType
from .models.config import QuestionGenerationConfig, QuestionType, QuestionTypeConfig

logger = logging.getLogger(__name__)


@dataclass
class PlannedQuestion:
    """A planned question to be generated."""
    plan_id: str
    question_type: QuestionType
    target_concept: ExtractedConcept
    difficulty: str
    marks: int
    requires_diagram: bool
    diagram_type: Optional[str] = None
    section_name: str = ""
    additional_context_query: str = ""  # Query for RAG retrieval
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "question_type": self.question_type.value,
            "target_concept": self.target_concept.to_dict(),
            "difficulty": self.difficulty,
            "marks": self.marks,
            "requires_diagram": self.requires_diagram,
            "diagram_type": self.diagram_type,
            "section_name": self.section_name,
            "additional_context_query": self.additional_context_query,
        }


@dataclass
class QuestionPlan:
    """Complete plan for paper generation."""
    subject: str
    grade: str
    total_questions: int
    total_marks: int
    sections: Dict[str, List[PlannedQuestion]]  # section_name -> questions
    planned_questions: List[PlannedQuestion]
    concepts_covered: List[str]
    estimated_time_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "grade": self.grade,
            "total_questions": self.total_questions,
            "total_marks": self.total_marks,
            "sections": {
                name: [q.to_dict() for q in qs]
                for name, qs in self.sections.items()
            },
            "planned_questions": [q.to_dict() for q in self.planned_questions],
            "concepts_covered": self.concepts_covered,
            "estimated_time_minutes": self.estimated_time_minutes,
        }


class QuestionPlannerService:
    """
    Plans question generation based on content analysis and configuration.
    
    Ensures:
    - Each question targets a specific concept
    - Concepts are not repeated unnecessarily
    - Difficulty distribution is maintained
    - Diagram opportunities are utilized appropriately
    """
    
    # Question type to section mapping
    SECTION_MAPPING = {
        QuestionType.MCQ: "Section A - Multiple Choice Questions",
        QuestionType.TRUE_FALSE: "Section A - True/False",
        QuestionType.FILL_IN_BLANKS: "Section B - Fill in the Blanks",
        QuestionType.SHORT_ANSWER: "Section B - Short Answer Questions",
        QuestionType.LONG_ANSWER: "Section C - Long Answer Questions",
        QuestionType.NUMERICAL: "Section D - Numerical Problems",
        QuestionType.MATCH_THE_FOLLOWING: "Section B - Match the Following",
    }
    
    # Concept types suitable for each question type
    CONCEPT_TYPE_SUITABILITY = {
        QuestionType.MCQ: [
            ConceptType.DEFINITION, ConceptType.FORMULA, ConceptType.EXAMPLE,
            ConceptType.COMPARISON, ConceptType.PROCESS
        ],
        QuestionType.TRUE_FALSE: [
            ConceptType.DEFINITION, ConceptType.THEOREM, ConceptType.COMPARISON
        ],
        QuestionType.SHORT_ANSWER: [
            ConceptType.DEFINITION, ConceptType.PROCESS, ConceptType.FORMULA,
            ConceptType.APPLICATION, ConceptType.DIAGRAM_CONCEPT
        ],
        QuestionType.LONG_ANSWER: [
            ConceptType.PROCESS, ConceptType.APPLICATION, ConceptType.THEOREM,
            ConceptType.COMPARISON, ConceptType.DIAGRAM_CONCEPT
        ],
        QuestionType.NUMERICAL: [
            ConceptType.FORMULA, ConceptType.APPLICATION, ConceptType.EXAMPLE
        ],
        QuestionType.FILL_IN_BLANKS: [
            ConceptType.DEFINITION, ConceptType.FORMULA
        ],
    }
    
    def create_plan(
        self,
        content_analysis: ContentAnalysis,
        generation_config: QuestionGenerationConfig,
    ) -> QuestionPlan:
        """
        Create a detailed question generation plan.
        
        Args:
            content_analysis: Analysis of the content with extracted concepts
            generation_config: Configuration for question generation
            
        Returns:
            QuestionPlan with all planned questions
        """
        logger.info(f"Creating question plan for {content_analysis.total_concepts} concepts")
        
        planned_questions: List[PlannedQuestion] = []
        sections: Dict[str, List[PlannedQuestion]] = {}
        used_concepts: List[str] = []
        
        # Get available concepts
        available_concepts = list(content_analysis.concepts)
        
        # Process each question type
        for q_type, type_config in generation_config.question_types.items():
            if type_config.count == 0:
                continue
            
            section_name = self.SECTION_MAPPING.get(q_type, f"Section - {q_type.value}")
            sections[section_name] = []
            
            # Plan questions for this type
            type_questions = self._plan_questions_for_type(
                q_type=q_type,
                type_config=type_config,
                available_concepts=available_concepts,
                used_concepts=used_concepts,
                generation_config=generation_config,
                section_name=section_name,
            )
            
            planned_questions.extend(type_questions)
            sections[section_name].extend(type_questions)
        
        # ENSURE MINIMUM 30% DIAGRAMS
        if generation_config.include_diagrams:
            planned_questions = self._ensure_minimum_diagrams(
                planned_questions, 
                min_diagram_percentage=30
            )
            # Update sections with modified questions
            for section_name in sections:
                sections[section_name] = [
                    q for q in planned_questions if q.section_name == section_name
                ]
        
        # Calculate totals
        total_marks = sum(q.marks for q in planned_questions)
        estimated_time = self._estimate_time(planned_questions)
        
        # Log diagram stats
        diagram_count = sum(1 for q in planned_questions if q.requires_diagram)
        diagram_pct = (diagram_count / len(planned_questions) * 100) if planned_questions else 0
        logger.info(f"Diagram allocation: {diagram_count}/{len(planned_questions)} ({diagram_pct:.1f}%)")
        
        return QuestionPlan(
            subject=content_analysis.subject,
            grade=content_analysis.grade,
            total_questions=len(planned_questions),
            total_marks=total_marks,
            sections=sections,
            planned_questions=planned_questions,
            concepts_covered=used_concepts,
            estimated_time_minutes=estimated_time,
        )
    
    def _plan_questions_for_type(
        self,
        q_type: QuestionType,
        type_config: QuestionTypeConfig,
        available_concepts: List[ExtractedConcept],
        used_concepts: List[str],
        generation_config: QuestionGenerationConfig,
        section_name: str,
    ) -> List[PlannedQuestion]:
        """Plan questions for a specific question type."""
        planned = []
        
        # Get suitable concept types for this question type
        suitable_types = self.CONCEPT_TYPE_SUITABILITY.get(q_type, list(ConceptType))
        
        # Filter concepts suitable for this question type
        suitable_concepts = [
            c for c in available_concepts
            if c.type in suitable_types
        ]
        
        # If no suitable concepts, use all available
        if not suitable_concepts:
            suitable_concepts = available_concepts.copy()
        
        # Determine difficulty distribution for this type
        total_count = type_config.count
        diff_dist = generation_config.difficulty_distribution
        
        easy_count = int(total_count * diff_dist.easy / 100)
        medium_count = int(total_count * diff_dist.medium / 100)
        hard_count = total_count - easy_count - medium_count
        
        difficulty_targets = (
            ["easy"] * easy_count +
            ["medium"] * medium_count +
            ["hard"] * hard_count
        )
        
        # Shuffle to mix difficulties
        random.shuffle(difficulty_targets)
        
        # Plan each question
        for i, target_difficulty in enumerate(difficulty_targets):
            # Select a concept (prefer unused, matching difficulty)
            concept = self._select_concept(
                suitable_concepts=suitable_concepts,
                used_concepts=used_concepts,
                target_difficulty=target_difficulty,
                prefer_diagram=(
                    generation_config.include_diagrams and
                    q_type in [QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER, QuestionType.NUMERICAL]
                ),
            )
            
            if concept is None:
                # Reuse a concept if necessary
                if suitable_concepts:
                    concept = random.choice(suitable_concepts)
                else:
                    continue
            
            # Mark concept as used
            if concept.concept_id not in used_concepts:
                used_concepts.append(concept.concept_id)
            
            # Determine if this question should have a diagram
            should_have_diagram = (
                generation_config.include_diagrams and
                concept.requires_diagram and
                q_type in [QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER, QuestionType.NUMERICAL, QuestionType.MCQ]
            )
            
            # Create planned question
            plan_id = f"plan_{q_type.value}_{i+1}"
            
            planned.append(PlannedQuestion(
                plan_id=plan_id,
                question_type=q_type,
                target_concept=concept,
                difficulty=target_difficulty,
                marks=type_config.marks_per_question,
                requires_diagram=should_have_diagram,
                diagram_type=concept.diagram_type_hint if should_have_diagram else None,
                section_name=section_name,
                additional_context_query=self._build_context_query(concept),
            ))
        
        return planned
    
    def _select_concept(
        self,
        suitable_concepts: List[ExtractedConcept],
        used_concepts: List[str],
        target_difficulty: str,
        prefer_diagram: bool = False,
    ) -> Optional[ExtractedConcept]:
        """Select the best concept for a question."""
        # Prefer unused concepts
        unused = [c for c in suitable_concepts if c.concept_id not in used_concepts]
        
        candidates = unused if unused else suitable_concepts
        
        if not candidates:
            return None
        
        # Filter by difficulty if possible
        difficulty_matches = [
            c for c in candidates
            if c.difficulty_estimate == target_difficulty
        ]
        
        if difficulty_matches:
            candidates = difficulty_matches
        
        # Prefer concepts with diagrams if requested
        if prefer_diagram:
            diagram_concepts = [c for c in candidates if c.requires_diagram]
            if diagram_concepts:
                candidates = diagram_concepts
        
        return random.choice(candidates)
    
    def _build_context_query(self, concept: ExtractedConcept) -> str:
        """Build a query for RAG retrieval."""
        # Combine concept name, description, and keywords
        parts = [concept.name]
        if concept.keywords:
            parts.extend(concept.keywords[:3])
        if concept.description:
            parts.append(concept.description[:100])
        
        return " ".join(parts)
    

    def _ensure_minimum_diagrams(
        self,
        planned_questions: List[PlannedQuestion],
        min_diagram_percentage: int = 30
    ) -> List[PlannedQuestion]:
        """
        Ensure at least the minimum percentage of questions have diagrams.
        
        Args:
            planned_questions: List of planned questions
            min_diagram_percentage: Minimum percentage of questions that should have diagrams
            
        Returns:
            Updated list with diagram requirements adjusted
        """
        if not planned_questions:
            return planned_questions
        
        total_questions = len(planned_questions)
        current_diagram_count = sum(1 for q in planned_questions if q.requires_diagram)
        min_diagram_count = max(1, int(total_questions * min_diagram_percentage / 100))
        
        if current_diagram_count >= min_diagram_count:
            logger.info(f"Diagram requirement already met: {current_diagram_count}/{total_questions}")
            return planned_questions
        
        # Need to add diagrams to more questions
        needed = min_diagram_count - current_diagram_count
        logger.info(f"Adding diagrams to {needed} more questions to meet {min_diagram_percentage}% minimum")
        
        # Question types that work well with diagrams (in priority order)
        diagram_suitable_types = [
            QuestionType.NUMERICAL,
            QuestionType.SHORT_ANSWER,
            QuestionType.LONG_ANSWER,
            QuestionType.MCQ,
        ]
        
        # Find questions without diagrams that could have one
        candidates = [
            q for q in planned_questions
            if not q.requires_diagram and q.question_type in diagram_suitable_types
        ]
        
        # Prioritize: NUMERICAL > SHORT_ANSWER > LONG_ANSWER > MCQ
        def priority_sort(q):
            try:
                return diagram_suitable_types.index(q.question_type)
            except ValueError:
                return 999
        
        candidates.sort(key=priority_sort)
        
        # Add diagram requirement to candidates
        added = 0
        for candidate in candidates:
            if added >= needed:
                break
            
            # Find the question in the list and update it
            for i, q in enumerate(planned_questions):
                if q.plan_id == candidate.plan_id:
                    # Create a new PlannedQuestion with requires_diagram=True
                    planned_questions[i] = PlannedQuestion(
                        plan_id=q.plan_id,
                        question_type=q.question_type,
                        target_concept=q.target_concept,
                        difficulty=q.difficulty,
                        marks=q.marks,
                        requires_diagram=True,
                        diagram_type=q.target_concept.diagram_type_hint,
                        section_name=q.section_name,
                        additional_context_query=q.additional_context_query,
                    )
                    added += 1
                    break
        
        final_count = sum(1 for q in planned_questions if q.requires_diagram)
        logger.info(f"Diagram allocation after adjustment: {final_count}/{total_questions}")
        
        return planned_questions

    def _estimate_time(self, questions: List[PlannedQuestion]) -> int:
        """Estimate time needed to complete the paper."""
        time_per_type = {
            QuestionType.MCQ: 1,
            QuestionType.TRUE_FALSE: 0.5,
            QuestionType.FILL_IN_BLANKS: 1,
            QuestionType.SHORT_ANSWER: 3,
            QuestionType.LONG_ANSWER: 8,
            QuestionType.NUMERICAL: 5,
            QuestionType.MATCH_THE_FOLLOWING: 2,
        }
        
        total_time = sum(
            time_per_type.get(q.question_type, 2)
            for q in questions
        )
        
        return int(total_time)
