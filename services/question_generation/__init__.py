"""
Question Generation System

A comprehensive system for generating educational question papers using:
- Qdrant vector database for storing and retrieving note embeddings
- OpenAI embeddings for semantic search
- GPT-4o for question generation
- Diagram Engine for visual content

Modules:
- qdrant_service: Vector database operations
- embedding_service: Text chunking and embedding generation
- document_ingestion_service: End-to-end document processing pipeline
- knowledge_base_repository: MongoDB CRUD operations for uploads
- background_processor: Async document processing
- question_generator: Core question generation logic (Phase 4)
- paper_assembly: PDF generation (Phase 5)
"""

from .qdrant_service import QdrantService, get_qdrant_service
from .embedding_service import EmbeddingService, get_embedding_service, TextChunker
from .document_ingestion_service import (
    DocumentIngestionService,
    get_document_ingestion_service,
    DocumentIngestionResult,
    DocumentIngestionError,
)
from .knowledge_base_repository import KnowledgeBaseRepository
from .papers_repository import PapersRepository, get_papers_repository, get_papers_repository_sync
from .background_processor import BackgroundProcessor, process_upload_background
from .question_generator import (
    QuestionGeneratorService,
    QuestionGeneratorError,
    get_question_generator,
    get_question_generator_sync,
)
from .content_analyzer import (
    ContentAnalyzerService,
    ContentAnalysis,
    ExtractedConcept,
    ConceptType,
)
from .question_planner import (
    QuestionPlannerService,
    QuestionPlan,
    PlannedQuestion,
)
from .paper_assembly import (
    PaperAssemblyService,
    PaperAssemblyError,
    get_paper_assembly_service,
    get_paper_assembly_service_sync,
)

# Import models
from .models import (
    # Config models
    QuestionType,
    DifficultyLevel,
    BloomLevel,
    QuestionTypeConfig,
    DifficultyDistribution,
    QuestionGenerationConfig,
    PaperConfig,
    # Question models
    QuestionOption,
    MarkingStep,
    DiagramSpec,
    GeneratedQuestion,
    GenerationSource,
    # Paper models
    PaperSection,
    GeneratedPaper,
    PaperStatus,
)

__all__ = [
    # Qdrant Service
    "QdrantService",
    "get_qdrant_service",
    # Embedding Service
    "EmbeddingService",
    "get_embedding_service",
    "TextChunker",
    # Document Ingestion Service
    "DocumentIngestionService",
    "get_document_ingestion_service",
    "DocumentIngestionResult",
    "DocumentIngestionError",
    # Knowledge Base Repository
    "KnowledgeBaseRepository",
    # Papers Repository (Phase 6)
    "PapersRepository",
    "get_papers_repository",
    "get_papers_repository_sync",
    # Background Processor
    "BackgroundProcessor",
    "process_upload_background",
    # Question Generator Service (Phase 4)
    "QuestionGeneratorService",
    "QuestionGeneratorError",
    "get_question_generator",
    "get_question_generator_sync",
    # Content Analyzer Service (Phase 8)
    "ContentAnalyzerService",
    "ContentAnalysis",
    "ExtractedConcept",
    "ConceptType",
    # Question Planner Service (Phase 8)
    "QuestionPlannerService",
    "QuestionPlan",
    "PlannedQuestion",
    # Paper Assembly Service (Phase 5)
    "PaperAssemblyService",
    "PaperAssemblyError",
    "get_paper_assembly_service",
    "get_paper_assembly_service_sync",
    # Config Models
    "QuestionType",
    "DifficultyLevel",
    "BloomLevel",
    "QuestionTypeConfig",
    "DifficultyDistribution",
    "QuestionGenerationConfig",
    "PaperConfig",
    # Question Models
    "QuestionOption",
    "MarkingStep",
    "DiagramSpec",
    "GeneratedQuestion",
    "GenerationSource",
    # Paper Models
    "PaperSection",
    "GeneratedPaper",
    "PaperStatus",
]

__version__ = "1.0.0"
