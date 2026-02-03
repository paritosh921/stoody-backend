"""
Question Generator Service - Core RAG-based question generation.

This service implements two modes of question generation:
1. Mode 1 (Notes): Generate questions from uploaded notes
2. Mode 2 (Topic): Generate questions from topic using knowledge base search

Refactored to use modular handlers:
- constants.py: Diagram types, section config
- diagram_handler.py: Diagram generation/verification
- llm_handler.py: LLM interaction and parsing
- prompts.py: Prompt templates
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .qdrant_service import get_qdrant_service, QdrantService
from .embedding_service import get_embedding_service, EmbeddingService
from .document_ingestion_service import get_document_ingestion_service, DocumentIngestionService
from .models.config import QuestionGenerationConfig, PaperConfig, QuestionType
from .models.question import GeneratedQuestion, GenerationSource
from .models.paper import GeneratedPaper, PaperSection, PaperStatus
from .content_analyzer import ContentAnalyzerService
from .question_planner import QuestionPlannerService, PlannedQuestion
from .prompts import get_single_question_prompt, SYSTEM_PROMPT
from .paper_assembly import get_paper_assembly_service

# Import from new modular files
from .constants import SECTION_CONFIG, get_diagram_types_text
from .diagram_handler import DiagramHandler, get_diagram_handler
from .llm_handler import LLMHandler, get_llm_handler

# Tool-based diagram generation with verification
from services.diagram_engine.verified_diagram_service import VerifiedDiagramService


logger = logging.getLogger(__name__)

# Singleton instance
_question_generator: Optional["QuestionGeneratorService"] = None


class QuestionGeneratorError(Exception):
    """Exception for question generation errors."""
    pass


class QuestionGeneratorService:
    """
    RAG-based question generation service.
    
    Supports two modes:
    1. generate_from_notes(): Generate from uploaded notes content
    2. generate_from_topic(): Generate from topic using knowledge base search
    """
    
    def __init__(self):
        self._initialized = False
        self._qdrant_service: Optional[QdrantService] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._ingestion_service: Optional[DocumentIngestionService] = None
        self._openai_service = None
        self._diagram_engine = None
        self._verified_diagram_service = None
        
        # New architecture services
        self._content_analyzer: Optional[ContentAnalyzerService] = None
        self._question_planner: Optional[QuestionPlannerService] = None
        
        # Handlers (modular)
        self._diagram_handler: Optional[DiagramHandler] = None
        self._llm_handler: Optional[LLMHandler] = None
        
        # Configuration
        self._model = "gpt-5-mini"  # Cost-efficient model for structured tasks
        self._temperature = 1  # gpt-5-nano only supports temperature=1
        self._max_context_chunks = 10
    
    async def initialize(self) -> bool:
        """Initialize all dependent services."""
        if self._initialized:
            return True
        
        try:
            # Get service instances
            self._qdrant_service = get_qdrant_service()
            self._embedding_service = get_embedding_service()
            self._ingestion_service = get_document_ingestion_service()
            
            # Initialize async services
            await self._qdrant_service.initialize()
            await self._embedding_service.initialize()
            await self._ingestion_service.initialize()
            
            # Import OpenAI service
            from services.async_openai_service import AsyncOpenAIService
            self._openai_service = AsyncOpenAIService()
            
            # Check if Kimi is enabled for question generation and diagrams
            from services.kimi_service import get_kimi_service, is_kimi_enabled, is_kimi_enabled_for_diagrams
            kimi_service = None
            use_kimi = is_kimi_enabled()
            use_kimi_diagrams = is_kimi_enabled_for_diagrams()

            if use_kimi or use_kimi_diagrams:
                kimi_service = get_kimi_service()
                if use_kimi:
                    logger.info("Kimi K2.5 enabled for question generation")
                if use_kimi_diagrams:
                    logger.info("Kimi K2.5 enabled for diagram generation/verification")
            else:
                logger.info("Using OpenAI for question generation (Kimi not enabled)")

            # Initialize new architecture services
            self._content_analyzer = ContentAnalyzerService(
                self._openai_service,
                kimi_service=kimi_service,
                use_kimi=use_kimi
            )
            self._question_planner = QuestionPlannerService()

            # Initialize verified diagram service with Kimi K2.5 for all LLM calls
            self._verified_diagram_service = VerifiedDiagramService(
                openai_service=self._openai_service,  # Fallback only
                kimi_service=kimi_service,  # Primary LLM for diagrams
                use_kimi=use_kimi_diagrams,  # Use Kimi for all diagram operations
                max_attempts=5,  # Increased for better quality
                use_planning=True,  # Enable Plan → Generate → Verify workflow
                use_enhanced_verification=True,  # Enable category-based scoring
                use_specialized_renderers=True,  # Use SchemDraw, RDKit, etc.
            )
            if use_kimi_diagrams:
                logger.info("Verified diagram service initialized with Kimi K2.5")
            else:
                logger.info("Verified diagram service initialized with OpenAI (fallback)")
            
            # Initialize handlers
            self._diagram_handler = get_diagram_handler()
            self._diagram_handler.set_services(self._openai_service, self._verified_diagram_service)
            
            self._llm_handler = get_llm_handler()
            self._llm_handler.set_service(self._openai_service)
            
            # Set Kimi service on LLM handler if enabled
            if use_kimi and kimi_service:
                self._llm_handler.set_kimi_service(kimi_service, use_kimi=True)
            
            # Import legacy Diagram Engine (fallback)
            try:
                from services.diagram_engine import get_diagram_engine
                self._diagram_engine = get_diagram_engine()
            except Exception as e:
                logger.warning(f"Diagram engine not available: {e}")
                self._diagram_engine = None
            
            self._initialized = True
            logger.info("QuestionGeneratorService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize QuestionGeneratorService: {e}")
            raise QuestionGeneratorError(f"Initialization failed: {e}")
    
    # =========================================================================
    # MAIN GENERATION METHODS
    # =========================================================================
    
    async def generate_from_notes(
        self,
        content: str,
        generation_config: QuestionGenerationConfig,
        paper_config: PaperConfig,
        tenant_id: str,
        teacher_id: str,
        store_embeddings: bool = True,
        source_file_id: Optional[str] = None,
    ) -> GeneratedPaper:
        """
        Mode 1: Generate questions from uploaded notes content.
        
        Args:
            content: The extracted text content from notes (after OCR)
            generation_config: Configuration for question generation
            paper_config: Configuration for paper layout
            tenant_id: Tenant identifier
            teacher_id: Teacher identifier
            store_embeddings: Whether to store embeddings in Qdrant
            source_file_id: Optional source file ID for tracking
            
        Returns:
            GeneratedPaper with all questions organized into sections
        """
        await self.initialize()
        
        start_time = datetime.utcnow()
        logger.info(f"Generating questions from notes for tenant {tenant_id}")
        
        paper = GeneratedPaper(
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            paper_config=paper_config,
            generation_config=generation_config,
            source_type="notes",
            source_subject=paper_config.subject,
            status=PaperStatus.GENERATING,
        )
        
        if source_file_id:
            paper.source_upload_ids = [source_file_id]
        
        try:
            # Step 1: Store embeddings for RAG retrieval
            if store_embeddings:
                logger.info("Storing content embeddings for RAG...")
                await self._store_content_embeddings(
                    content=content,
                    tenant_id=tenant_id,
                    teacher_id=teacher_id,
                    subject=paper_config.subject,
                    grade=paper_config.grade,
                    chapter=paper_config.chapter,
                    source_file_id=source_file_id,
                )
            
            # Step 2: Analyze content to extract concepts
            logger.info("Analyzing content to extract concepts...")
            content_analysis = await self._content_analyzer.analyze_content(
                content=content,
                subject=paper_config.subject,
                grade=paper_config.grade,
                chapter=paper_config.chapter,
            )
            logger.info(f"Found {content_analysis.total_concepts} concepts")
            
            # Step 3: Create question plan
            logger.info("Creating question plan...")
            question_plan = self._question_planner.create_plan(
                content_analysis=content_analysis,
                generation_config=generation_config,
            )
            logger.info(f"Planned {question_plan.total_questions} questions")
            
            # Step 4: Generate questions
            logger.info("Generating questions with RAG...")
            questions = await self._generate_questions_from_plan(
                question_plan=question_plan,
                paper_config=paper_config,
                tenant_id=tenant_id,
                content_for_fallback=content,
            )
            
            # Step 5: Organize into sections
            paper.sections = self._organize_into_sections(questions, generation_config)
            
            # Step 6: Generate PDFs
            paper = await self._generate_pdfs(paper)
            
            # Update stats
            paper.generation_stats = {
                "total_questions": len(questions),
                "concepts_analyzed": content_analysis.total_concepts,
                "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "questions_with_diagrams": sum(1 for q in questions if q.has_diagram),
                "architecture": "per_question_with_rag",
                "pdfs_generated": paper.question_paper_url is not None,
            }
            
            logger.info(f"Successfully generated {len(questions)} questions from notes")
            return paper
            
        except Exception as e:
            paper.status = PaperStatus.FAILED
            paper.error_message = str(e)
            logger.error(f"Question generation failed: {e}", exc_info=True)
            raise QuestionGeneratorError(f"Generation failed: {e}")
    
    async def generate_from_topic(
        self,
        topic: str,
        subject: str,
        grade: str,
        generation_config: QuestionGenerationConfig,
        paper_config: PaperConfig,
        tenant_id: str,
        teacher_id: str,
        chapter: Optional[str] = None,
        top_k: int = 15,
        score_threshold: float = 0.4,
    ) -> GeneratedPaper:
        """
        Mode 2: Generate questions from topic using knowledge base search.
        
        Args:
            topic: The topic to generate questions about
            subject: Subject name
            grade: Grade/Class level
            generation_config: Configuration for question generation
            paper_config: Configuration for paper layout
            tenant_id: Tenant identifier
            teacher_id: Teacher identifier
            chapter: Optional chapter filter
            top_k: Number of context chunks to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            GeneratedPaper with all questions organized into sections
        """
        await self.initialize()
        
        start_time = datetime.utcnow()
        logger.info(f"Generating questions from topic '{topic}' for tenant {tenant_id}")
        
        paper = GeneratedPaper(
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            paper_config=paper_config,
            generation_config=generation_config,
            source_type="topic",
            source_topic=topic,
            source_subject=subject,
            status=PaperStatus.GENERATING,
        )
        
        try:
            # Step 1: Search knowledge base
            logger.info("Searching knowledge base...")
            search_results = await self._ingestion_service.search_knowledge_base(
                query=topic,
                tenant_id=tenant_id,
                subject=subject,
                chapter=chapter,
                grade=grade,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            
            if not search_results:
                logger.warning(
                    f"No relevant content found for topic '{topic}' in knowledge base. "
                    "Falling back to LLM internal knowledge."
                )
                
                # Generate a temporary syllabus/content for the analyzer
                fallback_prompt = (
                    f"Create a detailed study outline and summary for the topic '{topic}' "
                    f"in Subject: {subject}, Grade: {grade}, Chapter: {chapter or 'General'}. "
                    "Include key definitions, main concepts, formulas (if any), and principles "
                    "that are typically tested in exams. "
                    "This content will be used to generate exam questions."
                )
                
                try:
                    generated_content = await self._llm_handler.generate_completion(
                        prompt=fallback_prompt,
                        system_prompt="You are an expert educational content creator.",
                        max_tokens=2000
                    )
                    
                    if generated_content:
                        base_content = f"GENERATED CONTENT FOR TOPIC: {topic}\n\n{generated_content}"
                        logger.info("Successfully generated fallback content using LLM")
                    else:
                        # Fallback to simple instruction if generation fails
                        base_content = (
                            f"TOPIC: {topic}\n"
                            f"SUBJECT: {subject}\n"
                            f"GRADE: {grade}\n"
                            f"CHAPTER: {chapter or 'General'}\n\n"
                            "NOTE: Generate standard exam-style questions based on the general curriculum."
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate fallback content: {e}")
                    base_content = f"TOPIC: {topic}\nSUBJECT: {subject}\nGRADE: {grade}"
            else:
                base_content = self._build_context_from_search(search_results)
                logger.info(f"Found {len(search_results)} relevant chunks")
            
            # Step 2: Analyze content
            logger.info("Analyzing content...")
            content_analysis = await self._content_analyzer.analyze_content(
                content=base_content,
                subject=subject,
                grade=grade,
                chapter=chapter,
            )
            
            # Step 3: Create question plan
            logger.info("Creating question plan...")
            question_plan = self._question_planner.create_plan(
                content_analysis=content_analysis,
                generation_config=generation_config,
            )
            
            # Step 4: Generate questions
            logger.info("Generating questions...")
            questions = await self._generate_questions_from_plan(
                question_plan=question_plan,
                paper_config=paper_config,
                tenant_id=tenant_id,
                content_for_fallback=base_content,
            )
            
            # Step 5: Organize into sections
            paper.sections = self._organize_into_sections(questions, generation_config)
            
            # Step 6: Generate PDFs
            paper = await self._generate_pdfs(paper)
            
            # Update stats
            paper.generation_stats = {
                "total_questions": len(questions),
                "concepts_analyzed": content_analysis.total_concepts,
                "initial_chunks_retrieved": len(search_results),
                "generation_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "questions_with_diagrams": sum(1 for q in questions if q.has_diagram),
                "architecture": "per_question_with_rag",
                "pdfs_generated": paper.question_paper_url is not None,
            }
            
            logger.info(f"Successfully generated {len(questions)} questions from topic")
            return paper
            
        except QuestionGeneratorError:
            paper.status = PaperStatus.FAILED
            raise
        except Exception as e:
            paper.status = PaperStatus.FAILED
            paper.error_message = str(e)
            logger.error(f"Question generation failed: {e}", exc_info=True)
            raise QuestionGeneratorError(f"Generation failed: {e}")
    
    # =========================================================================
    # QUESTION GENERATION HELPERS
    # =========================================================================
    
    async def _generate_questions_from_plan(
        self,
        question_plan,
        paper_config: PaperConfig,
        tenant_id: str,
        content_for_fallback: str,
    ) -> List[GeneratedQuestion]:
        """Generate questions one at a time based on the question plan."""
        all_questions: List[GeneratedQuestion] = []
        total_planned = len(question_plan.planned_questions)
        
        for idx, planned_q in enumerate(question_plan.planned_questions, 1):
            logger.info(f"Generating question {idx}/{total_planned}: {planned_q.target_concept.name}")
            
            try:
                # Retrieve relevant context via RAG
                context = await self._retrieve_context_for_concept(
                    concept=planned_q.target_concept,
                    query=planned_q.additional_context_query,
                    tenant_id=tenant_id,
                    subject=paper_config.subject,
                    grade=paper_config.grade,
                    chapter=paper_config.chapter,
                    fallback_content=content_for_fallback,
                )
                
                # Generate ONE question
                question = await self._generate_single_question(
                    planned_question=planned_q,
                    context=context,
                    paper_config=paper_config,
                )
                
                if question:
                    # Generate diagram if required
                    if (planned_q.requires_diagram or question.has_diagram) and question.diagram_spec:
                        question = await self._diagram_handler.generate_diagram_for_question(question)
                    
                    all_questions.append(question)
                    logger.info(f"   Generated: {question.question_text[:50]}...")
                else:
                    logger.warning(f"   Failed for concept: {planned_q.target_concept.name}")
                    
            except Exception as e:
                logger.error(f"   Error generating question {idx}: {e}")
        
        return all_questions
    
    async def _retrieve_context_for_concept(
        self,
        concept,
        query: str,
        tenant_id: str,
        subject: str,
        grade: str,
        chapter: Optional[str],
        fallback_content: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve relevant context for a specific concept using RAG."""
        try:
            enhanced_query = f"{concept.name} {query}"
            
            search_results = await self._ingestion_service.search_knowledge_base(
                query=enhanced_query,
                tenant_id=tenant_id,
                subject=subject,
                chapter=chapter,
                grade=grade,
                top_k=top_k,
                score_threshold=0.4,
            )
            
            if search_results:
                context_parts = [r.get("text", "") for r in search_results if r.get("text")]
                if context_parts:
                    logger.info(f"   Retrieved {len(context_parts)} relevant chunks via RAG")
                    return "\n\n---\n\n".join(context_parts)
            
            # Fallback
            logger.info("   Using fallback context")
            return f"{concept.related_text}\n\n---\n\n{fallback_content[:4000]}"
            
        except Exception as e:
            logger.warning(f"   RAG retrieval failed: {e}, using fallback")
            return f"{concept.related_text}\n\n---\n\n{fallback_content[:4000]}"
    
    async def _generate_single_question(
        self,
        planned_question: PlannedQuestion,
        context: str,
        paper_config: PaperConfig,
    ) -> Optional[GeneratedQuestion]:
        """Generate a single question using a focused prompt."""
        try:
            prompt = get_single_question_prompt(
                question_type=planned_question.question_type,
                subject=paper_config.subject,
                grade=paper_config.grade,
                concept_name=planned_question.target_concept.name,
                concept_description=planned_question.target_concept.description,
                difficulty=planned_question.difficulty,
                marks=planned_question.marks,
                context=context,
                requires_diagram=planned_question.requires_diagram,
                diagram_type_hint=planned_question.diagram_type,
            )
            
            response_text = await self._llm_handler.generate_completion(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=2000,
            )
            
            if not response_text:
                return None
            
            return self._llm_handler.parse_single_question_response(
                response_text=response_text,
                question_type=planned_question.question_type,
                subject=paper_config.subject,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate single question: {e}")
            return None
    
    # =========================================================================
    # ORGANIZATION AND UTILITY METHODS
    # =========================================================================
    
    def _organize_into_sections(
        self,
        questions: List[GeneratedQuestion],
        generation_config: QuestionGenerationConfig,
    ) -> List[PaperSection]:
        """Organize questions into paper sections by type."""
        sections = []
        
        # Group questions by type
        questions_by_type: Dict[str, List[GeneratedQuestion]] = {}
        for q in questions:
            q_type = q.question_type
            if q_type not in questions_by_type:
                questions_by_type[q_type] = []
            questions_by_type[q_type].append(q)
        
        # Create sections in order
        section_order = [
            QuestionType.MCQ, QuestionType.TRUE_FALSE, QuestionType.FILL_IN_BLANKS,
            QuestionType.SHORT_ANSWER, QuestionType.LONG_ANSWER,
            QuestionType.NUMERICAL, QuestionType.MATCH_THE_FOLLOWING,
        ]
        
        for q_type in section_order:
            type_key = q_type.value
            if type_key in questions_by_type and questions_by_type[type_key]:
                config = SECTION_CONFIG.get(q_type, {})
                type_config = generation_config.question_types.get(q_type)
                marks = type_config.marks_per_question if type_config else 1
                
                section = PaperSection(
                    name=config.get("name", f"Section - {q_type.value}"),
                    instructions=config.get("instructions", "").format(marks=marks),
                    questions=questions_by_type[type_key],
                )
                sections.append(section)
        
        return sections
    
    async def _generate_pdfs(self, paper: GeneratedPaper) -> GeneratedPaper:
        """Generate PDF documents for the paper."""
        logger.info("Generating PDF documents...")
        try:
            assembly_service = await get_paper_assembly_service()
            paper = await assembly_service.assemble_paper(
                paper=paper,
                generate_pdfs=True,
                upload_to_s3=True,
            )
            logger.info(f"   Question Paper: {paper.question_paper_url}")
            logger.info(f"   Answer Key: {paper.answer_key_url}")
            logger.info(f"   Marking Scheme: {paper.marking_scheme_url}")
        except Exception as pdf_error:
            logger.warning(f"PDF generation failed (non-fatal): {pdf_error}")
        return paper
    
    async def _store_content_embeddings(
        self,
        content: str,
        tenant_id: str,
        teacher_id: str,
        subject: str,
        grade: str,
        chapter: Optional[str] = None,
        source_file_id: Optional[str] = None,
    ) -> None:
        """Store content embeddings in Qdrant for future use."""
        try:
            await self._ingestion_service.ingest_text(
                text=content,
                tenant_id=tenant_id,
                teacher_id=teacher_id,
                subject=subject,
                grade=grade,
                chapter=chapter,
                source_id=source_file_id or f"notes_{datetime.utcnow().isoformat()}",
            )
            logger.info("Stored content embeddings successfully")
        except Exception as e:
            logger.warning(f"Failed to store embeddings (non-critical): {e}")
    
    def _build_context_from_search(self, search_results: List[Dict[str, Any]]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            text = result.get("text", "")
            topic = result.get("topic", "")
            chapter = result.get("chapter", "")
            
            header = f"[Context {i}]"
            if chapter:
                header += f" Chapter: {chapter}"
            if topic:
                header += f" | Topic: {topic}"
            
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    async def preview_questions(
        self,
        questions: List[GeneratedQuestion],
        include_answers: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate preview data for questions (without PDF)."""
        preview = []
        
        for q in questions:
            data = {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "marks": q.marks,
                "difficulty": q.difficulty,
            }
            
            if q.options:
                data["options"] = [
                    {"label": o.label, "content": o.content}
                    for o in q.options
                ]
            
            if q.has_diagram and q.diagram_url:
                data["diagram_url"] = q.diagram_url
            
            if include_answers:
                data["correct_answer"] = q.correct_answer
                data["solution"] = q.solution
            
            preview.append(data)
        
        return preview
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all dependent services."""
        status = {
            "service": "QuestionGeneratorService",
            "initialized": self._initialized,
            "dependencies": {},
        }
        
        if self._initialized:
            try:
                status["dependencies"]["qdrant"] = await self._qdrant_service.health_check()
            except:
                status["dependencies"]["qdrant"] = {"healthy": False}
            
            try:
                status["dependencies"]["embedding"] = await self._embedding_service.health_check()
            except:
                status["dependencies"]["embedding"] = {"healthy": False}
            
            status["dependencies"]["diagram_engine"] = {
                "available": self._diagram_engine is not None
            }
        
        return status


# ============================================================================
# Module-level functions
# ============================================================================

async def get_question_generator() -> QuestionGeneratorService:
    """Get the singleton QuestionGeneratorService instance."""
    global _question_generator
    
    if _question_generator is None:
        _question_generator = QuestionGeneratorService()
        await _question_generator.initialize()
    
    return _question_generator


def get_question_generator_sync() -> QuestionGeneratorService:
    """Get the service without initialization (for dependency injection)."""
    global _question_generator
    
    if _question_generator is None:
        _question_generator = QuestionGeneratorService()
    
    return _question_generator
