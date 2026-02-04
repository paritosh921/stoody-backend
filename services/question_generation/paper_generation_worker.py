"""
Paper Generation Worker for async question paper generation.

Implements the question-by-question generation pipeline with:
- Async job processing
- Progress tracking
- Two-LLM loop integration
- Memory/state management
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models.job import (
    Job,
    JobType,
    JobStatus,
    PaperGenerationMetadata,
    PaperGenerationResult,
    PaperGenerationState,
    GeneratedQuestionRecord,
)
from .models.paper import GeneratedPaper, PaperSection, PaperStatus
from .models.config import (
    QuestionType,
    DifficultyLevel,
    QuestionGenerationConfig,
    PaperConfig,
)
from .models.question import GeneratedQuestion, QuestionOption
from .jobs_repository import get_jobs_repository, JobsRepository
from .papers_repository import get_papers_repository, PapersRepository
from .llm_orchestrator import get_orchestrator, LLMOrchestrator
from .qdrant_service import get_qdrant_service

logger = logging.getLogger(__name__)


class PaperGenerationWorker:
    """
    Worker for async paper generation.
    
    Generates questions one at a time using the two-LLM loop,
    updating job progress throughout.
    """
    
    def __init__(
        self,
        jobs_repo: JobsRepository,
        papers_repo: PapersRepository,
        orchestrator: LLMOrchestrator,
    ):
        """
        Initialize the worker.
        
        Args:
            jobs_repo: Jobs repository
            papers_repo: Papers repository
            orchestrator: LLM orchestrator
        """
        self.jobs_repo = jobs_repo
        self.papers_repo = papers_repo
        self.orchestrator = orchestrator

    async def _verify_source_documents(
        self,
        tenant_id: str,
        pdf_ids: List[str],
    ) -> tuple[bool, List[str], List[str]]:
        """
        Verify that source documents exist in Qdrant before starting generation.
        
        This pre-check prevents wasted LLM calls when documents haven't been
        uploaded to the knowledge base.
        
        Args:
            tenant_id: Tenant identifier
            pdf_ids: List of source file IDs to verify
            
        Returns:
            Tuple of (all_exist, missing_ids, existing_ids)
        """
        if not pdf_ids:
            logger.warning("No pdf_ids provided for paper generation")
            return True, [], []
        
        try:
            qdrant_service = get_qdrant_service()
            await qdrant_service.initialize()
            
            # Check collection stats first
            stats = await qdrant_service.get_collection_stats(tenant_id)
            
            if not stats.get("exists"):
                logger.error(
                    f"Qdrant collection does not exist for tenant {tenant_id}. "
                    f"No documents have been uploaded to the knowledge base."
                )
                return False, pdf_ids, []
            
            if stats.get("points_count", 0) == 0:
                logger.error(
                    f"Qdrant collection for tenant {tenant_id} is empty. "
                    f"No documents have been uploaded to the knowledge base."
                )
                return False, pdf_ids, []
            
            # Verify which source_file_ids exist
            existence_check = await qdrant_service.verify_source_files_exist(
                tenant_id=tenant_id,
                source_file_ids=pdf_ids,
            )
            
            missing_ids = [sfid for sfid, exists in existence_check.items() if not exists]
            existing_ids = [sfid for sfid, exists in existence_check.items() if exists]
            
            if missing_ids:
                logger.error(
                    f"SOURCE DOCUMENTS NOT FOUND in Qdrant: {missing_ids}. "
                    f"These documents must be uploaded via /api/v1/knowledge-base/upload first."
                )
                
                # Log available source_file_ids for debugging
                available_ids = await qdrant_service.get_source_file_ids(
                    tenant_id=tenant_id,
                    limit=20,
                )
                if available_ids:
                    logger.info(
                        f"Available source_file_ids in collection: {available_ids}"
                    )
            
            if existing_ids:
                logger.info(f"Verified {len(existing_ids)} source documents exist in Qdrant")
            
            all_exist = len(missing_ids) == 0
            return all_exist, missing_ids, existing_ids
            
        except Exception as e:
            logger.error(f"Failed to verify source documents: {e}")
            # Don't block generation on verification failure - just warn
            return True, [], pdf_ids
    
    async def process_job(self, job: Job) -> None:
        """
        Process a paper generation job.
        
        Args:
            job: Job to process
        """
        start_time = time.time()
        paper_id = job.metadata.get("paper_id")
        
        logger.info(f"Starting paper generation job {job.job_id} for paper {paper_id}")
        
        try:
            # Mark job as running
            job.start()
            await self.jobs_repo.update_job(job)
            
            # Get paper from database
            # Note: get_by_id takes (tenant_id, paper_id) - tenant_id first
            paper = await self.papers_repo.get_by_id(job.institution_id, paper_id)
            if not paper:
                raise ValueError(f"Paper not found: {paper_id}")
            
            # Parse metadata
            metadata = PaperGenerationMetadata(**job.metadata)
            
            # Pre-check: Verify source documents exist in Qdrant
            if metadata.pdf_ids:
                all_exist, missing_ids, existing_ids = await self._verify_source_documents(
                    tenant_id=job.institution_id,
                    pdf_ids=metadata.pdf_ids,
                )
                
                if not all_exist:
                    # Log detailed warning but continue with generation
                    # Questions will fall back to LLM knowledge
                    logger.warning(
                        f"DOCUMENT VERIFICATION FAILED: {len(missing_ids)}/{len(metadata.pdf_ids)} "
                        f"documents not found in Qdrant. Missing IDs: {missing_ids}. "
                        f"Questions will be generated using LLM knowledge without RAG context."
                    )
                    
                    # Update job with warning
                    await self.jobs_repo.update_job_progress(
                        job_id=job.job_id,
                        current_step=0,
                        step_name="Document verification warning",
                        details=f"Warning: {len(missing_ids)} source documents not found in knowledge base. "
                                f"Generation will proceed with LLM fallback.",
                    )
            
            # Build question slots from blueprint
            question_slots = self._build_question_slots(metadata.blueprint)
            total_questions = len(question_slots)
            
            if total_questions == 0:
                raise ValueError("No questions to generate based on blueprint")
            
            # Initialize generation state
            state = PaperGenerationState(
                paper_id=paper_id,
                blueprint_summary=self._build_blueprint_summary(metadata),
                constraints=self._extract_constraints(metadata),
                questions_pending=total_questions,
            )
            await self.jobs_repo.create_generation_state(state)
            
            # Update job progress
            await self.jobs_repo.update_job_progress(
                job_id=job.job_id,
                current_step=0,
                total_steps=total_questions,
                step_name="Starting generation",
                details=f"Generating {total_questions} questions",
            )
            
            # Track results
            generated_questions: List[GeneratedQuestionRecord] = []
            questions_approved = 0
            questions_needing_review = 0
            total_llm_calls = 0
            
            # Generate questions one at a time
            for idx, slot in enumerate(question_slots):
                question_num = idx + 1
                
                logger.info(
                    f"Generating question {question_num}/{total_questions} "
                    f"({slot['type']}, {slot['difficulty']})"
                )
                
                # Update progress
                await self.jobs_repo.update_job_progress(
                    job_id=job.job_id,
                    current_step=idx,
                    step_name=f"Generating question {question_num}",
                    details=f"Type: {slot['type']}, Difficulty: {slot['difficulty']}",
                )
                
                # Refresh state from DB (in case of updates)
                state = await self.jobs_repo.get_generation_state(paper_id)
                
                # Generate the question
                # Use `or` instead of .get() fallback because slot["topic"] may be explicitly None
                record = await self.orchestrator.generate_single_question(
                    paper_id=paper_id,
                    section_index=slot["section_index"],
                    slot_index=slot["slot_index"],
                    question_type=slot["type"],
                    subject=metadata.subject,
                    grade=metadata.class_grade,
                    difficulty=slot["difficulty"],
                    marks=slot["marks"],
                    topic_focus=slot.get("topic") or metadata.subject,
                    diagram_required=slot.get("diagram") if slot.get("diagram") is not None else metadata.include_diagrams,
                    tenant_id=job.institution_id,
                    pdf_ids=metadata.pdf_ids,
                    memory_state=state,
                )
                
                # Render diagram if required and approved
                if record.approved and record.final_draft.diagram_required:
                    try:
                        from .diagram_renderer_service import render_question_diagram
                        success, diagram_data_url, error = await render_question_diagram(
                            record.final_draft
                        )
                        if success and diagram_data_url:
                            # Store the rendered diagram data URL in the spec
                            if record.final_draft.diagram_spec:
                                record.final_draft.diagram_spec["rendered_image"] = diagram_data_url
                            logger.info(f"Diagram rendered for question {question_num}")
                        elif error:
                            logger.warning(f"Diagram rendering failed for question {question_num}: {error}")
                            # Mark for review if diagram failed
                            record.needs_manual_review = True
                            record.manual_review_reason = f"Diagram rendering failed: {error}"
                    except Exception as diagram_err:
                        logger.warning(f"Diagram rendering error: {diagram_err}")
                
                generated_questions.append(record)
                total_llm_calls += record.total_iterations * 2  # Generate + Review per iteration
                
                # Update counters
                if record.approved:
                    questions_approved += 1
                if record.needs_manual_review:
                    questions_needing_review += 1

                
                # Update state
                question_summary = (
                    f"{slot['type'].upper()} on {record.final_draft.topic or 'topic'} "
                    f"({slot['difficulty']})"
                )
                topics = [record.final_draft.topic] if record.final_draft.topic else []
                state.add_completed_question(question_summary, topics)
                await self.jobs_repo.update_generation_state(state)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)
            
            # Final progress update
            await self.jobs_repo.update_job_progress(
                job_id=job.job_id,
                current_step=total_questions,
                step_name="Assembling paper",
                details="Building final paper structure",
            )
            
            # Assemble the paper
            paper = await self._assemble_paper(
                paper=paper,
                questions=generated_questions,
                blueprint=metadata.blueprint,
            )
            
            # Save updated paper
            await self.papers_repo.update(paper)
            
            # Calculate results
            duration_ms = int((time.time() - start_time) * 1000)
            
            result = PaperGenerationResult(
                paper_id=paper_id,
                total_questions=total_questions,
                questions_approved=questions_approved,
                questions_needing_review=questions_needing_review,
                total_marks=paper.total_marks,
                generation_time_ms=duration_ms,
                total_llm_calls=total_llm_calls,
                total_tokens_used=0,  # Would need to track from LLM calls
                questions_by_type=self._count_by_type(generated_questions),
                questions_by_difficulty=self._count_by_difficulty(generated_questions),
            )
            
            # Mark job as succeeded
            job.succeed(result.model_dump())
            await self.jobs_repo.update_job(job)
            
            # Clean up state
            await self.jobs_repo.delete_generation_state(paper_id)
            
            logger.info(
                f"Paper generation completed: {questions_approved}/{total_questions} approved, "
                f"{questions_needing_review} need review, {duration_ms}ms"
            )
            
        except Exception as e:
            logger.error(f"Paper generation failed: {e}", exc_info=True)
            job.fail(str(e), {"traceback": str(e)})
            await self.jobs_repo.update_job(job)
            
            # Update paper status
            try:
                # Note: get_by_id takes (tenant_id, paper_id) - tenant_id first
                paper = await self.papers_repo.get_by_id(job.institution_id, paper_id)
                if paper:
                    paper.status = PaperStatus.FAILED
                    paper.error_message = str(e)
                    await self.papers_repo.update(paper)
            except Exception:
                pass
    
    def _build_question_slots(self, blueprint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build a list of question slots from the blueprint.
        
        Args:
            blueprint: Paper blueprint configuration
            
        Returns:
            List of question slot definitions
        """
        slots = []
        sections = blueprint.get("sections", [])
        
        for section_idx, section in enumerate(sections):
            section_type = section.get("type", "mcq")
            count = section.get("count", 0)
            marks_each = section.get("marks_each", 1)
            difficulty_dist = section.get("difficulty", {"easy": 0.5, "med": 0.4, "hard": 0.1})
            
            # Normalize difficulty distribution
            total_dist = sum(difficulty_dist.values())
            if total_dist > 0:
                difficulty_dist = {k: v / total_dist for k, v in difficulty_dist.items()}
            
            # Calculate questions per difficulty
            easy_count = int(count * difficulty_dist.get("easy", 0.5))
            med_count = int(count * difficulty_dist.get("med", 0.4))
            hard_count = count - easy_count - med_count  # Remainder goes to hard
            
            slot_idx = 0
            
            # Easy questions
            for _ in range(easy_count):
                slots.append({
                    "section_index": section_idx,
                    "slot_index": slot_idx,
                    "type": section_type,
                    "marks": marks_each,
                    "difficulty": "easy",
                    "topic": section.get("topic"),
                    "diagram": section.get("include_diagram"),  # None means use global include_diagrams setting
                })
                slot_idx += 1
            
            # Medium questions
            for _ in range(med_count):
                slots.append({
                    "section_index": section_idx,
                    "slot_index": slot_idx,
                    "type": section_type,
                    "marks": marks_each,
                    "difficulty": "medium",
                    "topic": section.get("topic"),
                    "diagram": section.get("include_diagram"),  # None means use global include_diagrams setting
                })
                slot_idx += 1
            
            # Hard questions
            for _ in range(hard_count):
                slots.append({
                    "section_index": section_idx,
                    "slot_index": slot_idx,
                    "type": section_type,
                    "marks": marks_each,
                    "difficulty": "hard",
                    "topic": section.get("topic"),
                    "diagram": section.get("include_diagram"),  # None means use global include_diagrams setting
                })
                slot_idx += 1
        
        return slots
    
    def _build_blueprint_summary(self, metadata: PaperGenerationMetadata) -> str:
        """Build a compact blueprint summary for the state."""
        sections_summary = []
        for section in metadata.blueprint.get("sections", []):
            sections_summary.append(
                f"{section.get('count', 0)} {section.get('type', 'questions')}"
            )
        
        return (
            f"{metadata.subject} Grade {metadata.class_grade}: "
            f"{', '.join(sections_summary)}. "
            f"Style: {metadata.exam_style or 'Standard'}"
        )
    
    def _extract_constraints(self, metadata: PaperGenerationMetadata) -> List[str]:
        """Extract constraints from metadata."""
        constraints = []
        
        # Add exam style constraints
        exam_style = metadata.exam_style
        if exam_style:
            if exam_style.upper() == "JEE":
                constraints.append("Use JEE-style formatting with numerical precision")
                constraints.append("Include advanced problem-solving questions")
            elif exam_style.upper() == "NEET":
                constraints.append("Use NEET-style formatting")
                constraints.append("Focus on conceptual understanding")
            elif exam_style.upper() == "CBSE":
                constraints.append("Follow CBSE exam pattern")
                constraints.append("Include step-by-step solutions")
        
        # Add grade-appropriate constraints
        grade = metadata.class_grade
        if "10" in grade or "9" in grade:
            constraints.append("Questions should be appropriate for secondary school level")
        elif "11" in grade or "12" in grade:
            constraints.append("Questions can include higher-order thinking skills")
        
        return constraints
    
    async def _assemble_paper(
        self,
        paper: GeneratedPaper,
        questions: List[GeneratedQuestionRecord],
        blueprint: Dict[str, Any],
    ) -> GeneratedPaper:
        """
        Assemble the final paper from generated questions.
        
        Args:
            paper: Paper to update
            questions: Generated question records
            blueprint: Blueprint configuration
            
        Returns:
            Updated paper
        """
        sections_config = blueprint.get("sections", [])
        
        # Group questions by section
        questions_by_section: Dict[int, List[GeneratedQuestionRecord]] = {}
        for q in questions:
            section_idx = q.section_index
            if section_idx not in questions_by_section:
                questions_by_section[section_idx] = []
            questions_by_section[section_idx].append(q)
        
        # Build paper sections
        paper.sections = []
        
        for section_idx, section_config in enumerate(sections_config):
            section_type = section_config.get("type", "mcq")
            
            # Get section name
            type_names = {
                "mcq": "Multiple Choice Questions",
                "short": "Short Answer Questions",
                "short_answer": "Short Answer Questions",
                "long": "Long Answer Questions",
                "long_answer": "Long Answer Questions",
                "numerical": "Numerical Problems",
            }
            section_name = f"Section {chr(65 + section_idx)} - {type_names.get(section_type, 'Questions')}"
            
            # Get section instructions
            instructions = section_config.get("instructions", f"Answer all questions in this section.")
            
            # Convert records to GeneratedQuestion objects
            section_questions = []
            for record in questions_by_section.get(section_idx, []):
                draft = record.final_draft
                
                # Convert options
                options = None
                if draft.options:
                    options = [
                        QuestionOption(
                            label=opt.get("label", ""),
                            content=opt.get("content", ""),
                            is_correct=opt.get("is_correct", False),
                        )
                        for opt in draft.options
                    ]
                
                # Extract diagram URL from diagram_spec if available
                diagram_url = None
                if draft.diagram_spec and isinstance(draft.diagram_spec, dict):
                    diagram_url = draft.diagram_spec.get("rendered_image")
                
                question = GeneratedQuestion(
                    question_id=record.question_id,
                    question_text=draft.question_text,
                    question_type=draft.question_type,
                    options=options,
                    correct_answer=draft.correct_answer,
                    solution=draft.explanation,
                    marks=draft.marks,
                    difficulty=draft.difficulty,
                    bloom_level=draft.bloom_level,
                    topic=draft.topic,
                    has_diagram=draft.diagram_required,
                    diagram_spec=draft.diagram_spec,
                    diagram_url=diagram_url,
                    source_chunk_ids=draft.source_chunk_ids or [],
                )
                section_questions.append(question)
            
            # Create section
            section = PaperSection(
                name=section_name,
                instructions=instructions,
                questions=section_questions,
            )
            paper.sections.append(section)
        
        # Update paper status
        paper.status = PaperStatus.COMPLETED
        paper.completed_at = datetime.utcnow()
        
        # Calculate stats
        paper.generation_stats = {
            "total_questions": len(questions),
            "approved_questions": sum(1 for q in questions if q.approved),
            "questions_needing_review": sum(1 for q in questions if q.needs_manual_review),
            "total_iterations": sum(q.total_iterations for q in questions),
            "avg_iterations_per_question": (
                sum(q.total_iterations for q in questions) / len(questions)
                if questions else 0
            ),
        }
        
        return paper
    
    def _count_by_type(self, questions: List[GeneratedQuestionRecord]) -> Dict[str, int]:
        """Count questions by type."""
        counts: Dict[str, int] = {}
        for q in questions:
            qtype = q.final_draft.question_type
            counts[qtype] = counts.get(qtype, 0) + 1
        return counts
    
    def _count_by_difficulty(self, questions: List[GeneratedQuestionRecord]) -> Dict[str, int]:
        """Count questions by difficulty."""
        counts: Dict[str, int] = {}
        for q in questions:
            diff = q.final_draft.difficulty
            counts[diff] = counts.get(diff, 0) + 1
        return counts


async def create_paper_generation_job(
    institution_id: str,
    user_id: str,
    pdf_ids: List[str],
    subject: str,
    class_grade: str,
    blueprint: Dict[str, Any],
    include_diagrams: bool = True,
    exam_style: Optional[str] = None,
) -> tuple[Job, str]:
    """
    Create a new paper generation job.
    
    Args:
        institution_id: Institution identifier
        user_id: User who requested generation
        pdf_ids: Source PDF IDs
        subject: Subject name
        class_grade: Class/grade level
        blueprint: Paper blueprint configuration
        include_diagrams: Whether to include diagrams
        exam_style: Optional exam style (JEE, NEET, CBSE)
        
    Returns:
        Tuple of (Job, paper_id)
    """
    jobs_repo = await get_jobs_repository()
    papers_repo = await get_papers_repository()
    
    # Create paper record
    paper_id = str(uuid.uuid4())
    paper = GeneratedPaper(
        paper_id=paper_id,
        tenant_id=institution_id,
        teacher_id=user_id,
        source_type="notes",
        source_upload_ids=pdf_ids,
        source_subject=subject,
        status=PaperStatus.GENERATING,
        paper_config=PaperConfig(
            title=f"{subject} Question Paper",
            subject=subject,
            grade=class_grade,
        ),
    )
    await papers_repo.create(paper)
    
    # Create job
    metadata = PaperGenerationMetadata(
        paper_id=paper_id,
        pdf_ids=pdf_ids,
        subject=subject,
        class_grade=class_grade,
        blueprint=blueprint,
        include_diagrams=include_diagrams,
        exam_style=exam_style,
    )
    
    job = Job(
        job_type=JobType.GENERATE_PAPER,
        institution_id=institution_id,
        user_id=user_id,
        metadata=metadata.model_dump(),
    )
    
    await jobs_repo.create_job(job)
    
    return job, paper_id


async def run_paper_generation_worker(job_id: str) -> None:
    """
    Run the paper generation worker for a specific job.
    
    This is the entry point for the Celery task or background worker.
    
    Args:
        job_id: Job identifier to process
    """
    jobs_repo = await get_jobs_repository()
    papers_repo = await get_papers_repository()
    orchestrator = await get_orchestrator()
    
    # Get the job
    job = await jobs_repo.get_job(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return
    
    if job.status != JobStatus.QUEUED:
        logger.warning(f"Job {job_id} is not queued (status={job.status})")
        return
    
    # Create and run worker
    worker = PaperGenerationWorker(
        jobs_repo=jobs_repo,
        papers_repo=papers_repo,
        orchestrator=orchestrator,
    )
    
    await worker.process_job(job)
