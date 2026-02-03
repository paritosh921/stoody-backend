"""
Tests for Question Generation System

Tests for:
- TextChunker
- EmbeddingService
- QdrantService
- DocumentIngestionService
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import the modules to test
from services.question_generation.embedding_service import TextChunker, EmbeddingService
from services.question_generation.models.embedding import (
    ChunkMetadata,
    DocumentChunk,
    EmbeddingResult,
    SearchQuery,
    SearchResult,
)
from services.question_generation.models.knowledge_base import UploadStatus


# =============================================================================
# TextChunker Tests
# =============================================================================

class TestTextChunker:
    """Tests for TextChunker class"""
    
    def test_chunk_empty_text(self):
        """Empty text should return empty list"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        result = chunker.chunk_text("")
        assert result == []
        
        result = chunker.chunk_text("   ")
        assert result == []
    
    def test_chunk_text_smaller_than_chunk_size(self):
        """Text smaller than chunk size should return single chunk"""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        text = "This is a short text."
        result = chunker.chunk_text(text)
        
        assert len(result) == 1
        assert result[0] == text
    
    def test_chunk_text_with_overlap(self):
        """Text should be split with proper overlap"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 100  # 100 characters
        result = chunker.chunk_text(text)
        
        # Should have multiple chunks
        assert len(result) >= 2
        
        # Each chunk should not exceed chunk_size
        for chunk in result:
            assert len(chunk) <= 50
    
    def test_chunk_text_sentence_boundary(self):
        """Chunker should try to break at sentence boundaries"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        result = chunker.chunk_text(text)
        
        # Text is 89 chars, should fit in one chunk
        assert len(result) == 1
    
    def test_chunk_text_multiple_paragraphs(self):
        """Multiple paragraphs should be handled correctly"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        result = chunker.chunk_text(text)
        
        assert len(result) >= 1
        # All chunks should be non-empty
        for chunk in result:
            assert len(chunk.strip()) > 0
    
    def test_create_document_chunks(self):
        """Create DocumentChunk objects from text"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test document with some content that needs to be chunked."
        
        metadata = ChunkMetadata(
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            source_file_id="test_file",
            source_type="typed",
            subject="Physics",
            grade="Class 11",
            chunk_index=0,
            total_chunks=1,
        )
        
        chunks = chunker.create_document_chunks(text, metadata)
        
        assert len(chunks) >= 1
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, DocumentChunk)
            assert chunk.text
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == len(chunks)
            assert chunk.metadata.tenant_id == "test_tenant"
            assert chunk.metadata.subject == "Physics"
    
    def test_chunker_invalid_overlap(self):
        """Overlap >= chunk_size should raise error"""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)


# =============================================================================
# EmbeddingService Tests
# =============================================================================

class TestEmbeddingService:
    """Tests for EmbeddingService class"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance"""
        return EmbeddingService()
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI embedding response"""
        mock_embedding = [0.1] * 1536  # 1536 dimensions
        mock_data = Mock()
        mock_data.embedding = mock_embedding
        
        mock_usage = Mock()
        mock_usage.total_tokens = 10
        
        mock_response = Mock()
        mock_response.data = [mock_data]
        mock_response.usage = mock_usage
        
        return mock_response
    
    @pytest.mark.asyncio
    async def test_embed_text_empty(self, embedding_service):
        """Empty text should raise error"""
        with pytest.raises(ValueError):
            await embedding_service.embed_text("")
    
    @pytest.mark.asyncio
    async def test_embed_text_success(self, embedding_service, mock_openai_response):
        """Successful embedding generation"""
        with patch.object(embedding_service, 'client') as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
            embedding_service._initialized = True
            
            result = await embedding_service.embed_text("Test text")
            
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) == 1536
            assert result.text == "Test text"
            assert result.tokens_used == 10
    
    @pytest.mark.asyncio
    async def test_embed_texts_batch_empty(self, embedding_service):
        """Empty list should return empty results"""
        result, tokens = await embedding_service.embed_texts_batch([])
        assert result == []
        assert tokens == 0
    
    @pytest.mark.asyncio
    async def test_embed_texts_batch_filters_empty(self, embedding_service, mock_openai_response):
        """Batch should filter out empty texts"""
        with patch.object(embedding_service, 'client') as mock_client:
            # Mock for non-empty texts
            mock_client.embeddings.create = AsyncMock(return_value=mock_openai_response)
            embedding_service._initialized = True
            
            result, tokens = await embedding_service.embed_texts_batch(["valid", "", "  "])
            
            # Only one valid text
            mock_client.embeddings.create.assert_called_once()
    
    def test_chunker_instance(self, embedding_service):
        """Service should have a TextChunker instance"""
        assert hasattr(embedding_service, 'chunker')
        assert isinstance(embedding_service.chunker, TextChunker)


# =============================================================================
# SearchQuery Tests
# =============================================================================

class TestSearchQuery:
    """Tests for SearchQuery model"""
    
    def test_get_qdrant_filter_empty(self):
        """No filters should return None"""
        query = SearchQuery(query_text="test")
        assert query.get_qdrant_filter() is None
    
    def test_get_qdrant_filter_subject(self):
        """Subject filter should work"""
        query = SearchQuery(query_text="test", subject="Physics")
        filter_dict = query.get_qdrant_filter()
        
        assert filter_dict is not None
        assert "must" in filter_dict
        assert len(filter_dict["must"]) == 1
        assert filter_dict["must"][0]["key"] == "subject"
        assert filter_dict["must"][0]["match"]["value"] == "Physics"
    
    def test_get_qdrant_filter_multiple(self):
        """Multiple filters should be combined"""
        query = SearchQuery(
            query_text="test",
            subject="Physics",
            chapter="Laws of Motion",
            grade="Class 11",
        )
        filter_dict = query.get_qdrant_filter()
        
        assert filter_dict is not None
        assert len(filter_dict["must"]) == 3
    
    def test_get_qdrant_filter_source_files(self):
        """Source file IDs filter should use 'any' match"""
        query = SearchQuery(
            query_text="test",
            source_file_ids=["file1", "file2"],
        )
        filter_dict = query.get_qdrant_filter()
        
        assert filter_dict is not None
        assert filter_dict["must"][0]["key"] == "source_file_id"
        assert "any" in filter_dict["must"][0]["match"]


# =============================================================================
# ChunkMetadata Tests
# =============================================================================

class TestChunkMetadata:
    """Tests for ChunkMetadata model"""
    
    def test_to_qdrant_payload(self):
        """Metadata should convert to Qdrant payload format"""
        metadata = ChunkMetadata(
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            source_file_id="test_file",
            source_type="pdf",
            subject="Physics",
            chapter="Chapter 1",
            topic="Topic A",
            grade="Class 11",
            chunk_index=0,
            total_chunks=5,
        )
        
        payload = metadata.to_qdrant_payload()
        
        assert payload["tenant_id"] == "test_tenant"
        assert payload["teacher_id"] == "test_teacher"
        assert payload["source_file_id"] == "test_file"
        assert payload["source_type"] == "pdf"
        assert payload["subject"] == "Physics"
        assert payload["chapter"] == "Chapter 1"
        assert payload["topic"] == "Topic A"
        assert payload["grade"] == "Class 11"
        assert payload["chunk_index"] == 0
        assert payload["total_chunks"] == 5
        assert "created_at" in payload


# =============================================================================
# DocumentChunk Tests
# =============================================================================

class TestDocumentChunk:
    """Tests for DocumentChunk model"""
    
    def test_has_embedding_false(self):
        """Chunk without embedding should return False"""
        metadata = ChunkMetadata(
            tenant_id="test",
            source_file_id="test",
            source_type="typed",
            subject="Physics",
            grade="Class 11",
            chunk_index=0,
            total_chunks=1,
        )
        chunk = DocumentChunk(text="Test", metadata=metadata)
        
        assert chunk.has_embedding is False
    
    def test_has_embedding_true(self):
        """Chunk with embedding should return True"""
        metadata = ChunkMetadata(
            tenant_id="test",
            source_file_id="test",
            source_type="typed",
            subject="Physics",
            grade="Class 11",
            chunk_index=0,
            total_chunks=1,
        )
        chunk = DocumentChunk(
            text="Test",
            metadata=metadata,
            embedding=[0.1] * 1536,
        )
        
        assert chunk.has_embedding is True
    
    def test_to_qdrant_point_without_embedding(self):
        """Converting to Qdrant point without embedding should raise error"""
        metadata = ChunkMetadata(
            tenant_id="test",
            source_file_id="test",
            source_type="typed",
            subject="Physics",
            grade="Class 11",
            chunk_index=0,
            total_chunks=1,
        )
        chunk = DocumentChunk(text="Test", metadata=metadata)
        
        with pytest.raises(ValueError):
            chunk.to_qdrant_point()
    
    def test_to_qdrant_point_with_embedding(self):
        """Converting to Qdrant point with embedding should work"""
        metadata = ChunkMetadata(
            tenant_id="test",
            source_file_id="test",
            source_type="typed",
            subject="Physics",
            grade="Class 11",
            chunk_index=0,
            total_chunks=1,
        )
        embedding = [0.1] * 1536
        chunk = DocumentChunk(
            text="Test content",
            metadata=metadata,
            embedding=embedding,
        )
        
        point = chunk.to_qdrant_point()
        
        assert "id" in point
        assert point["vector"] == embedding
        assert "payload" in point
        assert point["payload"]["text"] == "Test content"
        assert point["payload"]["subject"] == "Physics"


# =============================================================================
# SearchResult Tests  
# =============================================================================

class TestSearchResult:
    """Tests for SearchResult model"""
    
    def test_properties(self):
        """Property accessors should work"""
        result = SearchResult(
            id="test_id",
            score=0.95,
            text="Test content",
            metadata={
                "subject": "Physics",
                "chapter": "Chapter 1",
                "topic": "Topic A",
                "source_file_id": "file_123",
            }
        )
        
        assert result.subject == "Physics"
        assert result.chapter == "Chapter 1"
        assert result.topic == "Topic A"
        assert result.source_file_id == "file_123"
    
    def test_missing_metadata_fields(self):
        """Missing metadata fields should return defaults"""
        result = SearchResult(
            id="test_id",
            score=0.95,
            text="Test content",
            metadata={},
        )
        
        assert result.subject == ""
        assert result.chapter is None
        assert result.topic is None
        assert result.source_file_id == ""


# =============================================================================
# Integration Test Helpers
# =============================================================================

class TestIntegrationHelpers:
    """Helper tests for integration scenarios"""
    
    def test_chunk_and_embed_pipeline_structure(self):
        """Test the structure of chunk -> embed -> store pipeline"""
        # Create sample text
        text = """
        Newton's First Law of Motion states that an object at rest stays at rest 
        and an object in motion stays in motion with the same speed and in the 
        same direction unless acted upon by an unbalanced force.
        
        Newton's Second Law of Motion states that the acceleration of an object 
        is directly proportional to the net force acting on it and inversely 
        proportional to its mass. This can be expressed as F = ma.
        """
        
        # Create chunker
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        
        # Create metadata
        metadata = ChunkMetadata(
            tenant_id="school_123",
            teacher_id="teacher_abc",
            source_file_id="notes_physics_ch5",
            source_type="typed",
            subject="Physics",
            chapter="Laws of Motion",
            grade="Class 11",
            chunk_index=0,
            total_chunks=1,
        )
        
        # Create chunks
        chunks = chunker.create_document_chunks(text, metadata)
        
        # Verify structure
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.text
            assert chunk.metadata.tenant_id == "school_123"
            assert chunk.metadata.subject == "Physics"


# =============================================================================
# Phase 3: MongoDB Models Tests
# =============================================================================

class TestKnowledgeBaseUploadModel:
    """Tests for KnowledgeBaseUpload MongoDB model"""
    
    def test_create_upload(self):
        """Create upload with default values"""
        from models.knowledge_base import KnowledgeBaseUpload, UploadStatus
        
        upload = KnowledgeBaseUpload(
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            original_filename="notes.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            subject="Physics",
            grade="Class 11",
        )
        
        assert upload.id is not None
        assert upload.tenant_id == "test_tenant"
        assert upload.status == UploadStatus.PENDING.value
        assert upload.retry_count == 0
    
    def test_to_dict(self):
        """Convert upload to dict for MongoDB"""
        from models.knowledge_base import KnowledgeBaseUpload
        
        upload = KnowledgeBaseUpload(
            id="test_id",
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            original_filename="notes.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            subject="Physics",
            grade="Class 11",
        )
        
        data = upload.to_dict()
        
        assert data["_id"] == "test_id"
        assert data["tenant_id"] == "test_tenant"
        assert data["subject"] == "Physics"
        assert "created_at" in data
    
    def test_from_dict(self):
        """Create upload from MongoDB document"""
        from models.knowledge_base import KnowledgeBaseUpload, UploadStatus
        
        data = {
            "_id": "test_id",
            "tenant_id": "test_tenant",
            "teacher_id": "test_teacher",
            "original_filename": "notes.pdf",
            "file_type": "pdf",
            "file_size_bytes": 1024,
            "subject": "Physics",
            "grade": "Class 11",
            "status": UploadStatus.COMPLETED.value,
        }
        
        upload = KnowledgeBaseUpload.from_dict(data)
        
        assert upload.id == "test_id"
        assert upload.tenant_id == "test_tenant"
        assert upload.status == UploadStatus.COMPLETED.value
    
    def test_update_status(self):
        """Update upload status"""
        from models.knowledge_base import KnowledgeBaseUpload, UploadStatus
        
        upload = KnowledgeBaseUpload(
            tenant_id="test",
            teacher_id="test",
            original_filename="test.pdf",
            subject="Physics",
            grade="Class 11",
        )
        
        upload.update_status(UploadStatus.PROCESSING)
        assert upload.status == UploadStatus.PROCESSING.value
        assert upload.processing_started_at is not None
        
        upload.update_status(UploadStatus.COMPLETED)
        assert upload.status == UploadStatus.COMPLETED.value
        assert upload.processing_completed_at is not None
    
    def test_can_retry(self):
        """Check retry logic"""
        from models.knowledge_base import KnowledgeBaseUpload, UploadStatus
        
        upload = KnowledgeBaseUpload(
            tenant_id="test",
            teacher_id="test",
            original_filename="test.pdf",
            subject="Physics",
            grade="Class 11",
            status=UploadStatus.FAILED.value,
            retry_count=0,
            max_retries=3,
        )
        
        assert upload.can_retry() is True
        
        upload.retry_count = 3
        assert upload.can_retry() is False
        
        upload.status = UploadStatus.COMPLETED.value
        upload.retry_count = 0
        assert upload.can_retry() is False


class TestProcessingMetrics:
    """Tests for ProcessingMetrics model"""
    
    def test_create_metrics(self):
        """Create metrics with values"""
        from models.knowledge_base import ProcessingMetrics
        
        metrics = ProcessingMetrics(
            extraction_time_ms=100,
            embedding_time_ms=200,
            storage_time_ms=50,
            total_time_ms=350,
            tokens_used=1000,
        )
        
        assert metrics.extraction_time_ms == 100
        assert metrics.tokens_used == 1000
    
    def test_to_dict(self):
        """Convert metrics to dict"""
        from models.knowledge_base import ProcessingMetrics
        
        metrics = ProcessingMetrics(
            extraction_time_ms=100,
            tokens_used=500,
        )
        
        data = metrics.to_dict()
        
        assert data["extraction_time_ms"] == 100
        assert data["tokens_used"] == 500
    
    def test_from_dict(self):
        """Create metrics from dict"""
        from models.knowledge_base import ProcessingMetrics
        
        data = {
            "extraction_time_ms": 150,
            "embedding_time_ms": 250,
            "tokens_used": 1500,
        }
        
        metrics = ProcessingMetrics.from_dict(data)
        
        assert metrics.extraction_time_ms == 150
        assert metrics.tokens_used == 1500


# =============================================================================
# Phase 3: Background Processor Tests
# =============================================================================

class TestBackgroundProcessor:
    """Tests for BackgroundProcessor"""
    
    @pytest.mark.asyncio
    async def test_process_upload_structure(self):
        """Test that background processor has correct structure"""
        from services.question_generation.background_processor import BackgroundProcessor
        
        # Just verify the class can be instantiated with a mock db_manager
        mock_db = Mock()
        processor = BackgroundProcessor(mock_db)
        
        assert processor.db_manager == mock_db
        assert processor.repository is not None


# =============================================================================
# Phase 3: Repository Tests
# =============================================================================

class TestKnowledgeBaseRepository:
    """Tests for KnowledgeBaseRepository structure"""
    
    def test_repository_has_required_methods(self):
        """Repository should have all CRUD methods"""
        from services.question_generation.knowledge_base_repository import KnowledgeBaseRepository
        
        # Check that required methods exist
        assert hasattr(KnowledgeBaseRepository, 'create')
        assert hasattr(KnowledgeBaseRepository, 'get_by_id')
        assert hasattr(KnowledgeBaseRepository, 'update')
        assert hasattr(KnowledgeBaseRepository, 'delete')
        assert hasattr(KnowledgeBaseRepository, 'list_uploads')
        assert hasattr(KnowledgeBaseRepository, 'count_uploads')
        assert hasattr(KnowledgeBaseRepository, 'update_status')
        assert hasattr(KnowledgeBaseRepository, 'get_stats')
        assert hasattr(KnowledgeBaseRepository, 'get_pending_uploads')
        assert hasattr(KnowledgeBaseRepository, 'get_failed_uploads_for_retry')


# =============================================================================
# Phase 4: Question Generator Service Tests
# =============================================================================

class TestQuestionGenerationModels:
    """Tests for Phase 4 question generation models"""
    
    def test_question_type_enum(self):
        """Test QuestionType enum values"""
        from services.question_generation.models.config import QuestionType
        
        assert QuestionType.MCQ.value == "mcq"
        assert QuestionType.SHORT_ANSWER.value == "short_answer"
        assert QuestionType.LONG_ANSWER.value == "long_answer"
        assert QuestionType.NUMERICAL.value == "numerical"
    
    def test_difficulty_level_enum(self):
        """Test DifficultyLevel enum values"""
        from services.question_generation.models.config import DifficultyLevel
        
        assert DifficultyLevel.EASY.value == "easy"
        assert DifficultyLevel.MEDIUM.value == "medium"
        assert DifficultyLevel.HARD.value == "hard"
    
    def test_bloom_level_enum(self):
        """Test BloomLevel enum values"""
        from services.question_generation.models.config import BloomLevel
        
        assert BloomLevel.REMEMBER.value == "remember"
        assert BloomLevel.UNDERSTAND.value == "understand"
        assert BloomLevel.APPLY.value == "apply"
        assert BloomLevel.ANALYZE.value == "analyze"
        assert BloomLevel.EVALUATE.value == "evaluate"
        assert BloomLevel.CREATE.value == "create"


class TestQuestionTypeConfig:
    """Tests for QuestionTypeConfig model"""
    
    def test_create_config(self):
        """Create question type config"""
        from services.question_generation.models.config import QuestionTypeConfig
        
        config = QuestionTypeConfig(count=10, marks_per_question=2)
        
        assert config.count == 10
        assert config.marks_per_question == 2
        assert config.total_marks == 20
    
    def test_total_marks_property(self):
        """Total marks should be count * marks_per_question"""
        from services.question_generation.models.config import QuestionTypeConfig
        
        config = QuestionTypeConfig(count=5, marks_per_question=3)
        assert config.total_marks == 15


class TestDifficultyDistribution:
    """Tests for DifficultyDistribution model"""
    
    def test_default_distribution(self):
        """Default distribution should sum to 100"""
        from services.question_generation.models.config import DifficultyDistribution
        
        dist = DifficultyDistribution()
        
        assert dist.easy == 30
        assert dist.medium == 50
        assert dist.hard == 20
        assert dist.easy + dist.medium + dist.hard == 100
    
    def test_custom_distribution(self):
        """Custom distribution values"""
        from services.question_generation.models.config import DifficultyDistribution
        
        dist = DifficultyDistribution(easy=40, medium=40, hard=20)
        
        assert dist.easy == 40
        assert dist.medium == 40
        assert dist.hard == 20


class TestQuestionGenerationConfig:
    """Tests for QuestionGenerationConfig model"""
    
    def test_default_config(self):
        """Default config should have sensible values"""
        from services.question_generation.models.config import (
            QuestionGenerationConfig,
            QuestionType,
        )
        
        config = QuestionGenerationConfig()
        
        assert len(config.question_types) > 0
        assert QuestionType.MCQ in config.question_types
        assert config.include_diagrams is True
        assert config.include_solutions is True
    
    def test_total_questions_property(self):
        """Total questions should sum all types"""
        from services.question_generation.models.config import (
            QuestionGenerationConfig,
            QuestionType,
            QuestionTypeConfig,
        )
        
        config = QuestionGenerationConfig(
            question_types={
                QuestionType.MCQ: QuestionTypeConfig(count=10, marks_per_question=1),
                QuestionType.SHORT_ANSWER: QuestionTypeConfig(count=5, marks_per_question=2),
            }
        )
        
        assert config.total_questions == 15
        assert config.total_marks == 20  # 10*1 + 5*2
    
    def test_questions_per_difficulty(self):
        """Should calculate questions per difficulty level"""
        from services.question_generation.models.config import (
            QuestionGenerationConfig,
            QuestionType,
            QuestionTypeConfig,
            DifficultyLevel,
        )
        
        config = QuestionGenerationConfig(
            question_types={
                QuestionType.MCQ: QuestionTypeConfig(count=10, marks_per_question=1),
            }
        )
        
        per_difficulty = config.get_questions_per_difficulty()
        
        # With default distribution: easy=30%, medium=50%, hard=20%
        assert DifficultyLevel.EASY in per_difficulty
        assert DifficultyLevel.MEDIUM in per_difficulty
        assert DifficultyLevel.HARD in per_difficulty


class TestPaperConfig:
    """Tests for PaperConfig model"""
    
    def test_create_paper_config(self):
        """Create paper config with required fields"""
        from services.question_generation.models.config import PaperConfig
        
        config = PaperConfig(
            title="Unit Test - Chapter 5",
            subject="Physics",
            grade="Class 11",
        )
        
        assert config.title == "Unit Test - Chapter 5"
        assert config.subject == "Physics"
        assert config.grade == "Class 11"
        assert config.duration_minutes == 90  # default
    
    def test_optional_fields(self):
        """Optional fields should work"""
        from services.question_generation.models.config import PaperConfig
        
        config = PaperConfig(
            title="Test",
            subject="Physics",
            grade="Class 11",
            school_name="Test School",
            chapter="Laws of Motion",
            topics=["Newton's Laws", "Friction"],
        )
        
        assert config.school_name == "Test School"
        assert config.chapter == "Laws of Motion"
        assert len(config.topics) == 2
    
    def test_default_instructions(self):
        """Should have default general instructions"""
        from services.question_generation.models.config import PaperConfig
        
        config = PaperConfig(
            title="Test",
            subject="Physics",
            grade="Class 11",
        )
        
        assert config.general_instructions is not None
        assert len(config.general_instructions) > 0


class TestGeneratedQuestion:
    """Tests for GeneratedQuestion model"""
    
    def test_create_mcq(self):
        """Create MCQ question"""
        from services.question_generation.models.question import (
            GeneratedQuestion,
            QuestionOption,
        )
        
        options = [
            QuestionOption(label="A", content="10 m/s", is_correct=False),
            QuestionOption(label="B", content="15 m/s", is_correct=True),
            QuestionOption(label="C", content="20 m/s", is_correct=False),
            QuestionOption(label="D", content="25 m/s", is_correct=False),
        ]
        
        question = GeneratedQuestion(
            question_text="What is the velocity?",
            question_type="mcq",
            options=options,
            correct_answer="B",
            marks=1,
            difficulty="medium",
        )
        
        assert question.question_text == "What is the velocity?"
        assert question.question_type == "mcq"
        assert len(question.options) == 4
        assert question.correct_answer == "B"
    
    def test_to_dict(self):
        """Question should convert to dict"""
        from services.question_generation.models.question import GeneratedQuestion
        
        question = GeneratedQuestion(
            question_text="Test question",
            question_type="short_answer",
            solution="Test solution",
            marks=2,
            subject="Physics",
        )
        
        data = question.to_dict()
        
        assert data["question_text"] == "Test question"
        assert data["question_type"] == "short_answer"
        assert data["solution"] == "Test solution"
        assert data["marks"] == 2
        assert data["subject"] == "Physics"
    
    def test_from_dict(self):
        """Question should be created from dict"""
        from services.question_generation.models.question import GeneratedQuestion
        
        data = {
            "question_text": "Calculate the force",
            "question_type": "numerical",
            "solution": "F = ma = 5 * 10 = 50 N",
            "marks": 3,
            "difficulty": "hard",
            "subject": "Physics",
        }
        
        question = GeneratedQuestion.from_dict(data)
        
        assert question.question_text == "Calculate the force"
        assert question.question_type == "numerical"
        assert question.difficulty == "hard"
        assert question.marks == 3


class TestDiagramSpec:
    """Tests for DiagramSpec model"""
    
    def test_create_diagram_spec(self):
        """Create diagram specification"""
        from services.question_generation.models.question import DiagramSpec
        
        spec = DiagramSpec(
            subject="physics",
            diagram_type="circuit",
            title="Simple Circuit",
            description="A simple series circuit",
            parameters={"components": ["resistor", "battery"]},
        )
        
        assert spec.subject == "physics"
        assert spec.diagram_type == "circuit"
        assert spec.title == "Simple Circuit"
    
    def test_to_dict(self):
        """Diagram spec should convert to dict"""
        from services.question_generation.models.question import DiagramSpec
        
        spec = DiagramSpec(
            subject="math",
            diagram_type="graph",
            title="Quadratic Function",
            parameters={"equation": "y = x^2"},
        )
        
        data = spec.to_dict()
        
        assert data["subject"] == "math"
        assert data["diagram_type"] == "graph"
        assert data["parameters"]["equation"] == "y = x^2"


class TestPaperSection:
    """Tests for PaperSection model"""
    
    def test_create_section(self):
        """Create paper section"""
        from services.question_generation.models.paper import PaperSection
        from services.question_generation.models.question import GeneratedQuestion
        
        questions = [
            GeneratedQuestion(question_text="Q1", marks=1),
            GeneratedQuestion(question_text="Q2", marks=1),
            GeneratedQuestion(question_text="Q3", marks=1),
        ]
        
        section = PaperSection(
            name="Section A - MCQ",
            instructions="Choose the correct option",
            questions=questions,
        )
        
        assert section.name == "Section A - MCQ"
        assert section.question_count == 3
        assert section.total_marks == 3
    
    def test_to_dict(self):
        """Section should convert to dict"""
        from services.question_generation.models.paper import PaperSection
        from services.question_generation.models.question import GeneratedQuestion
        
        questions = [GeneratedQuestion(question_text="Q1", marks=2)]
        section = PaperSection(
            name="Section B",
            instructions="Answer briefly",
            questions=questions,
        )
        
        data = section.to_dict()
        
        assert data["name"] == "Section B"
        assert data["question_count"] == 1
        assert data["total_marks"] == 2
        assert len(data["questions"]) == 1


class TestGeneratedPaper:
    """Tests for GeneratedPaper model"""
    
    def test_create_paper(self):
        """Create generated paper"""
        from services.question_generation.models.paper import (
            GeneratedPaper,
            PaperSection,
            PaperStatus,
        )
        from services.question_generation.models.config import PaperConfig
        from services.question_generation.models.question import GeneratedQuestion
        
        paper_config = PaperConfig(
            title="Unit Test",
            subject="Physics",
            grade="Class 11",
        )
        
        questions = [
            GeneratedQuestion(question_text="Q1", marks=1),
            GeneratedQuestion(question_text="Q2", marks=2),
        ]
        
        section = PaperSection(
            name="Section A",
            instructions="Answer all",
            questions=questions,
        )
        
        paper = GeneratedPaper(
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            paper_config=paper_config,
            sections=[section],
        )
        
        assert paper.paper_id is not None
        assert paper.tenant_id == "test_tenant"
        assert paper.title == "Unit Test"
        assert paper.total_questions == 2
        assert paper.total_marks == 3
        assert paper.status == PaperStatus.GENERATING
    
    def test_get_all_questions(self):
        """Should get all questions from all sections"""
        from services.question_generation.models.paper import (
            GeneratedPaper,
            PaperSection,
        )
        from services.question_generation.models.question import GeneratedQuestion
        
        section1 = PaperSection(
            name="Section A",
            instructions="",
            questions=[
                GeneratedQuestion(question_text="Q1", question_type="mcq"),
                GeneratedQuestion(question_text="Q2", question_type="mcq"),
            ],
        )
        
        section2 = PaperSection(
            name="Section B",
            instructions="",
            questions=[
                GeneratedQuestion(question_text="Q3", question_type="short_answer"),
            ],
        )
        
        paper = GeneratedPaper(
            tenant_id="test",
            sections=[section1, section2],
        )
        
        all_questions = paper.get_all_questions()
        
        assert len(all_questions) == 3
    
    def test_to_dict(self):
        """Paper should convert to dict"""
        from services.question_generation.models.paper import GeneratedPaper
        from services.question_generation.models.config import PaperConfig
        
        paper = GeneratedPaper(
            tenant_id="test",
            paper_config=PaperConfig(
                title="Test Paper",
                subject="Physics",
                grade="Class 11",
            ),
        )
        
        data = paper.to_dict()
        
        assert data["paper_id"] == paper.paper_id
        assert data["title"] == "Test Paper"
        assert data["subject"] == "Physics"
        assert "created_at" in data


class TestQuestionGeneratorService:
    """Tests for QuestionGeneratorService"""
    
    def test_service_has_required_methods(self):
        """Service should have required methods"""
        from services.question_generation.question_generator import QuestionGeneratorService
        
        assert hasattr(QuestionGeneratorService, 'initialize')
        assert hasattr(QuestionGeneratorService, 'generate_from_notes')
        assert hasattr(QuestionGeneratorService, 'generate_from_topic')
        assert hasattr(QuestionGeneratorService, 'preview_questions')
        assert hasattr(QuestionGeneratorService, 'health_check')
    
    def test_service_instantiation(self):
        """Service should be instantiable"""
        from services.question_generation.question_generator import QuestionGeneratorService
        
        service = QuestionGeneratorService()
        
        assert service._initialized is False
        assert service._model == "gpt-4o"
    
    def test_extract_json_simple(self):
        """Should extract JSON from simple text"""
        from services.question_generation.question_generator import QuestionGeneratorService
        
        service = QuestionGeneratorService()
        
        text = '{"questions": [{"text": "test"}]}'
        result = service._extract_json(text)
        
        assert '"questions"' in result
    
    def test_extract_json_with_markdown(self):
        """Should extract JSON from markdown code block"""
        from services.question_generation.question_generator import QuestionGeneratorService
        
        service = QuestionGeneratorService()
        
        text = '''Here is the JSON:
```json
{"questions": [{"text": "test"}]}
```
That's all.'''
        
        result = service._extract_json(text)
        assert '"questions"' in result


class TestQuestionGenerationAPIModels:
    """Tests for API request/response models"""
    
    def test_generation_config_request_conversion(self):
        """Request should convert to internal config"""
        from api.v1.question_generation import (
            GenerationConfigRequest,
            QuestionTypeConfigRequest,
        )
        
        request = GenerationConfigRequest(
            mcq=QuestionTypeConfigRequest(count=5, marks_per_question=1),
            short_answer=QuestionTypeConfigRequest(count=3, marks_per_question=2),
        )
        
        internal = request.to_internal_config()
        
        assert internal.total_questions == 8
        assert internal.total_marks == 11  # 5*1 + 3*2
    
    def test_paper_config_request_conversion(self):
        """Paper request should convert to internal config"""
        from api.v1.question_generation import PaperConfigRequest
        
        request = PaperConfigRequest(
            title="Test Paper",
            subject="Physics",
            grade="Class 11",
            chapter="Motion",
        )
        
        internal = request.to_internal_config()
        
        assert internal.title == "Test Paper"
        assert internal.subject == "Physics"
        assert internal.chapter == "Motion"


# =============================================================================
# Phase 5: Paper Assembly Service Tests
# =============================================================================

class TestPaperAssemblyService:
    """Tests for PaperAssemblyService"""
    
    def test_service_has_required_methods(self):
        """Service should have required methods"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        assert hasattr(PaperAssemblyService, 'initialize')
        assert hasattr(PaperAssemblyService, 'assemble_paper')
        assert hasattr(PaperAssemblyService, 'generate_question_paper_pdf')
        assert hasattr(PaperAssemblyService, 'generate_answer_key_pdf')
        assert hasattr(PaperAssemblyService, 'generate_marking_scheme_pdf')
        assert hasattr(PaperAssemblyService, 'health_check')
    
    def test_service_instantiation(self):
        """Service should be instantiable"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        service = PaperAssemblyService()
        
        assert service._initialized is False
        assert service._page_size is not None
    
    def test_escape_html(self):
        """HTML characters should be escaped"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        service = PaperAssemblyService()
        
        assert service._escape_html("<test>") == "&lt;test&gt;"
        assert service._escape_html("a & b") == "a &amp; b"
        assert service._escape_html("") == ""
        assert service._escape_html(None) == ""


class TestPDFStyles:
    """Tests for PDF style configuration"""
    
    def test_get_pdf_styles(self):
        """Should return dictionary of styles"""
        from services.question_generation.paper_assembly import get_pdf_styles
        
        styles = get_pdf_styles()
        
        assert isinstance(styles, dict)
        assert "SchoolName" in styles
        assert "ExamTitle" in styles
        assert "SectionHeader" in styles
        assert "QuestionText" in styles
        assert "AnswerText" in styles
        assert "MarkingStep" in styles


class TestPaperAssemblyPDFGeneration:
    """Tests for PDF generation functionality"""
    
    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing"""
        from services.question_generation.models.paper import (
            GeneratedPaper,
            PaperSection,
        )
        from services.question_generation.models.config import PaperConfig
        from services.question_generation.models.question import (
            GeneratedQuestion,
            QuestionOption,
            MarkingStep,
        )
        
        questions = [
            GeneratedQuestion(
                question_text="What is Newton's First Law?",
                question_type="mcq",
                options=[
                    QuestionOption(label="A", content="Law of Inertia", is_correct=True),
                    QuestionOption(label="B", content="F = ma", is_correct=False),
                    QuestionOption(label="C", content="Action-Reaction", is_correct=False),
                    QuestionOption(label="D", content="Law of Gravitation", is_correct=False),
                ],
                correct_answer="A",
                solution="Newton's First Law is also known as the Law of Inertia.",
                marks=1,
                difficulty="easy",
            ),
            GeneratedQuestion(
                question_text="Calculate the force on a 5 kg mass with acceleration 10 m/s².",
                question_type="numerical",
                solution="F = ma = 5 × 10 = 50 N",
                solution_steps=["Given: m = 5 kg, a = 10 m/s²", "Formula: F = ma", "F = 50 N"],
                marking_scheme=[
                    MarkingStep(step="Identify given values", marks=0.5),
                    MarkingStep(step="Apply formula F = ma", marks=1.0),
                    MarkingStep(step="Calculate correctly", marks=1.5),
                ],
                marks=3,
                difficulty="medium",
            ),
        ]
        
        section = PaperSection(
            name="Section A - Mixed Questions",
            instructions="Answer all questions",
            questions=questions,
        )
        
        paper = GeneratedPaper(
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            paper_config=PaperConfig(
                title="Physics Unit Test",
                subject="Physics",
                grade="Class 11",
                school_name="Test School",
                duration_minutes=45,
                general_instructions=[
                    "All questions are compulsory.",
                    "Write neatly.",
                ],
            ),
            sections=[section],
        )
        
        return paper
    
    @pytest.mark.asyncio
    async def test_generate_question_paper_pdf(self, sample_paper):
        """Should generate question paper PDF"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        service = PaperAssemblyService()
        await service.initialize()
        
        pdf_bytes = await service.generate_question_paper_pdf(sample_paper)
        
        # Check PDF was generated
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        
        # Check PDF header
        assert pdf_bytes[:4] == b'%PDF'
    
    @pytest.mark.asyncio
    async def test_generate_answer_key_pdf(self, sample_paper):
        """Should generate answer key PDF"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        service = PaperAssemblyService()
        await service.initialize()
        
        pdf_bytes = await service.generate_answer_key_pdf(sample_paper)
        
        # Check PDF was generated
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b'%PDF'
    
    @pytest.mark.asyncio
    async def test_generate_marking_scheme_pdf(self, sample_paper):
        """Should generate marking scheme PDF"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        service = PaperAssemblyService()
        await service.initialize()
        
        pdf_bytes = await service.generate_marking_scheme_pdf(sample_paper)
        
        # Check PDF was generated
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b'%PDF'
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Service health check should work"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        
        service = PaperAssemblyService()
        await service.initialize()
        
        health = await service.health_check()
        
        assert health["service"] == "PaperAssemblyService"
        assert health["initialized"] is True


class TestPapersAPI:
    """Tests for papers API endpoints"""
    
    def test_paper_storage_functions(self):
        """In-memory paper storage should work"""
        from api.v1.papers import (
            store_generated_paper,
            _get_paper,
            _delete_paper,
            _list_papers,
        )
        from services.question_generation.models.paper import GeneratedPaper
        
        # Create test paper
        paper = GeneratedPaper(
            tenant_id="test_tenant",
            teacher_id="test_teacher",
        )
        
        # Store
        store_generated_paper(paper)
        
        # Retrieve
        retrieved = _get_paper(paper.paper_id)
        assert retrieved is not None
        assert retrieved.paper_id == paper.paper_id
        
        # List
        papers = _list_papers(tenant_id="test_tenant")
        assert len(papers) >= 1
        
        # Delete
        deleted = _delete_paper(paper.paper_id)
        assert deleted is True
        
        # Verify deleted
        assert _get_paper(paper.paper_id) is None
    
    def test_paper_summary_model(self):
        """PaperSummary model should work"""
        from api.v1.papers import PaperSummary
        
        summary = PaperSummary(
            paper_id="test_id",
            title="Test Paper",
            subject="Physics",
            grade="Class 11",
            total_questions=10,
            total_marks=30,
            duration_minutes=60,
            status="completed",
            source_type="notes",
            created_at="2025-01-31T12:00:00",
        )
        
        assert summary.paper_id == "test_id"
        assert summary.total_questions == 10
    
    def test_paper_details_model(self):
        """PaperDetails model should work"""
        from api.v1.papers import PaperDetails
        
        details = PaperDetails(
            paper_id="test_id",
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            title="Test Paper",
            subject="Physics",
            grade="Class 11",
            total_questions=10,
            total_marks=30,
            duration_minutes=60,
            status="completed",
            source_type="topic",
            sections=[],
            created_at="2025-01-31T12:00:00",
        )
        
        assert details.paper_id == "test_id"
        assert details.source_type == "topic"


class TestPaperAssemblyIntegration:
    """Integration tests for paper assembly with question generator"""
    
    @pytest.mark.asyncio
    async def test_full_paper_generation_flow(self):
        """Test complete flow from questions to PDF"""
        from services.question_generation.paper_assembly import PaperAssemblyService
        from services.question_generation.models.paper import (
            GeneratedPaper,
            PaperSection,
            PaperStatus,
        )
        from services.question_generation.models.config import PaperConfig
        from services.question_generation.models.question import (
            GeneratedQuestion,
            QuestionOption,
        )
        
        # Create questions
        mcq_questions = [
            GeneratedQuestion(
                question_text=f"MCQ Question {i+1}",
                question_type="mcq",
                options=[
                    QuestionOption(label="A", content="Option A", is_correct=i % 4 == 0),
                    QuestionOption(label="B", content="Option B", is_correct=i % 4 == 1),
                    QuestionOption(label="C", content="Option C", is_correct=i % 4 == 2),
                    QuestionOption(label="D", content="Option D", is_correct=i % 4 == 3),
                ],
                correct_answer=["A", "B", "C", "D"][i % 4],
                marks=1,
            )
            for i in range(5)
        ]
        
        short_answer_questions = [
            GeneratedQuestion(
                question_text=f"Short Answer Question {i+1}",
                question_type="short_answer",
                solution=f"Sample answer for question {i+1}",
                marks=2,
            )
            for i in range(3)
        ]
        
        # Create sections
        sections = [
            PaperSection(
                name="Section A - Multiple Choice",
                instructions="Choose the correct option",
                questions=mcq_questions,
            ),
            PaperSection(
                name="Section B - Short Answer",
                instructions="Answer briefly",
                questions=short_answer_questions,
            ),
        ]
        
        # Create paper
        paper = GeneratedPaper(
            tenant_id="integration_test",
            teacher_id="test_teacher",
            paper_config=PaperConfig(
                title="Integration Test Paper",
                subject="Science",
                grade="Class 10",
                duration_minutes=60,
            ),
            sections=sections,
        )
        
        # Generate PDFs
        service = PaperAssemblyService()
        await service.initialize()
        
        result = await service.assemble_paper(
            paper,
            generate_pdfs=True,
            upload_to_s3=False,  # Don't upload in tests
        )
        
        # Verify result
        assert result.status == PaperStatus.COMPLETED
        assert result.completed_at is not None
        assert result.total_questions == 8  # 5 MCQ + 3 Short
        assert result.total_marks == 11  # 5*1 + 3*2


# =============================================================================
# Phase 6: API Integration Tests
# =============================================================================

class TestPapersRepository:
    """Tests for PapersRepository"""
    
    def test_repository_has_required_methods(self):
        """Repository should have all CRUD methods"""
        from services.question_generation.papers_repository import PapersRepository
        
        # Check that required methods exist
        assert hasattr(PapersRepository, 'create')
        assert hasattr(PapersRepository, 'get_by_id')
        assert hasattr(PapersRepository, 'update')
        assert hasattr(PapersRepository, 'delete')
        assert hasattr(PapersRepository, 'list_papers')
        assert hasattr(PapersRepository, 'count_papers')
        assert hasattr(PapersRepository, 'update_status')
        assert hasattr(PapersRepository, 'update_urls')
        assert hasattr(PapersRepository, 'get_stats')
        assert hasattr(PapersRepository, 'get_papers_by_teacher')
        assert hasattr(PapersRepository, 'get_recent_papers')
        assert hasattr(PapersRepository, 'get_published_papers')
        assert hasattr(PapersRepository, 'ensure_indexes')
    
    def test_repository_instantiation(self):
        """Repository should be instantiable"""
        from services.question_generation.papers_repository import PapersRepository
        
        mock_db = Mock()
        repo = PapersRepository(mock_db)
        
        assert repo.db_manager == mock_db


class TestGeneratedPaperMongoConversion:
    """Tests for GeneratedPaper MongoDB conversion"""
    
    def test_to_mongo_dict(self):
        """Paper should convert to MongoDB document format"""
        from services.question_generation.models.paper import (
            GeneratedPaper,
            PaperSection,
            PaperStatus,
        )
        from services.question_generation.models.config import PaperConfig
        from services.question_generation.models.question import GeneratedQuestion
        
        questions = [
            GeneratedQuestion(question_text="Q1", marks=1),
        ]
        
        section = PaperSection(
            name="Section A",
            instructions="Answer all",
            questions=questions,
        )
        
        paper = GeneratedPaper(
            paper_id="test_paper_123",
            tenant_id="test_tenant",
            teacher_id="test_teacher",
            paper_config=PaperConfig(
                title="Test Paper",
                subject="Physics",
                grade="Class 11",
            ),
            sections=[section],
            status=PaperStatus.COMPLETED,
        )
        
        mongo_doc = paper.to_mongo_dict()
        
        assert mongo_doc["_id"] == "test_paper_123"
        assert mongo_doc["tenant_id"] == "test_tenant"
        assert mongo_doc["status"] == "completed"
        assert "paper_config" in mongo_doc
        assert mongo_doc["paper_config"]["title"] == "Test Paper"
        assert "sections" in mongo_doc
        assert len(mongo_doc["sections"]) == 1
    
    def test_from_mongo_dict(self):
        """Paper should be created from MongoDB document"""
        from services.question_generation.models.paper import GeneratedPaper, PaperStatus
        from datetime import datetime
        
        mongo_doc = {
            "_id": "test_paper_456",
            "tenant_id": "test_tenant",
            "teacher_id": "test_teacher",
            "status": "completed",
            "source_type": "topic",
            "source_topic": "Newton's Laws",
            "created_at": datetime.utcnow(),
            "paper_config": {
                "title": "Physics Test",
                "subject": "Physics",
                "grade": "Class 11",
                "duration_minutes": 60,
            },
            "generation_config": {
                "question_types": {
                    "mcq": {"count": 5, "marks_per_question": 1},
                },
                "difficulty_distribution": {
                    "easy": 30,
                    "medium": 50,
                    "hard": 20,
                },
                "bloom_levels": ["remember", "understand"],
                "include_diagrams": True,
                "include_solutions": True,
                "include_marking_scheme": True,
            },
            "sections": [
                {
                    "name": "Section A",
                    "instructions": "Answer all",
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_text": "Test question",
                            "question_type": "mcq",
                            "marks": 1,
                            "difficulty": "easy",
                            "options": [
                                {"label": "A", "content": "Option A", "is_correct": True},
                                {"label": "B", "content": "Option B", "is_correct": False},
                            ],
                            "correct_answer": "A",
                            "solution": "The answer is A",
                            "marking_scheme": [],
                        }
                    ],
                }
            ],
        }
        
        paper = GeneratedPaper.from_mongo_dict(mongo_doc)
        
        assert paper.paper_id == "test_paper_456"
        assert paper.tenant_id == "test_tenant"
        assert paper.status == PaperStatus.COMPLETED
        assert paper.source_topic == "Newton's Laws"
        assert paper.paper_config is not None
        assert paper.paper_config.title == "Physics Test"
        assert len(paper.sections) == 1
        assert len(paper.sections[0].questions) == 1
    
    def test_roundtrip_conversion(self):
        """Converting to MongoDB and back should preserve data"""
        from services.question_generation.models.paper import (
            GeneratedPaper,
            PaperSection,
            PaperStatus,
        )
        from services.question_generation.models.config import (
            PaperConfig,
            QuestionGenerationConfig,
            QuestionType,
            QuestionTypeConfig,
        )
        from services.question_generation.models.question import (
            GeneratedQuestion,
            QuestionOption,
            MarkingStep,
        )
        
        # Create a complete paper
        question = GeneratedQuestion(
            question_id="q123",
            question_text="What is 2+2?",
            question_type="mcq",
            options=[
                QuestionOption(label="A", content="3", is_correct=False),
                QuestionOption(label="B", content="4", is_correct=True),
                QuestionOption(label="C", content="5", is_correct=False),
                QuestionOption(label="D", content="6", is_correct=False),
            ],
            correct_answer="B",
            solution="2+2=4",
            marking_scheme=[
                MarkingStep(step_number=1, description="Correct answer", marks=1),
            ],
            marks=1,
            difficulty="easy",
        )
        
        section = PaperSection(
            name="Section A",
            instructions="Choose the correct option",
            questions=[question],
        )
        
        original = GeneratedPaper(
            paper_id="roundtrip_test",
            tenant_id="tenant123",
            teacher_id="teacher456",
            paper_config=PaperConfig(
                title="Math Test",
                subject="Mathematics",
                grade="Class 5",
                duration_minutes=30,
                school_name="Test School",
            ),
            generation_config=QuestionGenerationConfig(
                question_types={
                    QuestionType.MCQ: QuestionTypeConfig(count=10, marks_per_question=1),
                },
            ),
            sections=[section],
            status=PaperStatus.COMPLETED,
            source_type="notes",
        )
        
        # Convert to MongoDB and back
        mongo_doc = original.to_mongo_dict()
        restored = GeneratedPaper.from_mongo_dict(mongo_doc)
        
        # Verify key fields
        assert restored.paper_id == original.paper_id
        assert restored.tenant_id == original.tenant_id
        assert restored.teacher_id == original.teacher_id
        assert restored.status == original.status
        assert restored.paper_config.title == original.paper_config.title
        assert restored.paper_config.school_name == original.paper_config.school_name
        assert len(restored.sections) == len(original.sections)
        assert len(restored.sections[0].questions) == len(original.sections[0].questions)
        assert restored.sections[0].questions[0].question_text == question.question_text
        assert restored.sections[0].questions[0].correct_answer == question.correct_answer


class TestAPIAuthenticationPatterns:
    """Tests for API authentication patterns"""
    
    def test_require_teacher_or_admin_function(self):
        """require_teacher_or_admin should validate role"""
        from api.v1.papers import require_teacher_or_admin
        from fastapi import HTTPException
        
        # Teacher should pass
        teacher_user = {"role": "tutor", "tenant_id": "t1", "user_id": "u1"}
        result = require_teacher_or_admin(teacher_user)
        assert result == teacher_user
        
        # Admin should pass
        admin_user = {"role": "admin", "tenant_id": "t1", "user_id": "u2"}
        result = require_teacher_or_admin(admin_user)
        assert result == admin_user
        
        # Student should fail
        student_user = {"role": "student", "tenant_id": "t1", "user_id": "u3"}
        try:
            require_teacher_or_admin(student_user)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 403
    
    def test_verify_paper_access_function(self):
        """_verify_paper_access should validate paper ownership"""
        from api.v1.papers import _verify_paper_access
        from services.question_generation.models.paper import GeneratedPaper
        from fastapi import HTTPException
        
        paper = GeneratedPaper(
            paper_id="test",
            tenant_id="tenant1",
            teacher_id="teacher1",
        )
        
        # Owner should have access
        owner = {"role": "tutor", "tenant_id": "tenant1", "user_id": "teacher1"}
        try:
            _verify_paper_access(paper, owner)
        except HTTPException:
            assert False, "Owner should have access"
        
        # Admin should have access
        admin = {"role": "admin", "tenant_id": "tenant1", "user_id": "admin1"}
        try:
            _verify_paper_access(paper, admin)
        except HTTPException:
            assert False, "Admin should have access"
        
        # Different teacher should not have access
        other_teacher = {"role": "tutor", "tenant_id": "tenant1", "user_id": "teacher2"}
        try:
            _verify_paper_access(paper, other_teacher)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 403
        
        # Different tenant should not have access
        different_tenant = {"role": "admin", "tenant_id": "tenant2", "user_id": "admin1"}
        try:
            _verify_paper_access(paper, different_tenant)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 403


class TestQuestionGenerationAPIAuth:
    """Tests for question generation API authentication"""
    
    def test_api_has_rate_limiting(self):
        """API endpoints should have rate limiting decorators"""
        from api.v1.question_generation import (
            generate_from_notes,
            generate_from_topic,
            preview_questions,
            get_templates,
            get_question_types,
            health_check,
        )
        
        # Check that functions exist and are decorated
        assert callable(generate_from_notes)
        assert callable(generate_from_topic)
        assert callable(preview_questions)
        assert callable(get_templates)
        assert callable(get_question_types)
        assert callable(health_check)
    
    def test_api_limiter_configured(self):
        """API should have limiter configured"""
        from api.v1 import question_generation
        
        assert hasattr(question_generation, 'limiter')


class TestPapersAPIAuth:
    """Tests for papers API authentication"""
    
    def test_api_has_rate_limiting(self):
        """API endpoints should have rate limiting"""
        from api.v1.papers import (
            list_papers,
            get_paper_details,
            download_question_paper,
            download_answer_key,
            download_marking_scheme,
            regenerate_paper,
            delete_paper,
            publish_paper,
            archive_paper,
        )
        
        # Check that functions exist
        assert callable(list_papers)
        assert callable(get_paper_details)
        assert callable(download_question_paper)
        assert callable(download_answer_key)
        assert callable(download_marking_scheme)
        assert callable(regenerate_paper)
        assert callable(delete_paper)
        assert callable(publish_paper)
        assert callable(archive_paper)
    
    def test_api_limiter_configured(self):
        """API should have limiter configured"""
        from api.v1 import papers
        
        assert hasattr(papers, 'limiter')


class TestPhase6Integration:
    """Integration tests for Phase 6 features"""
    
    def test_all_phase6_exports_available(self):
        """All Phase 6 exports should be available"""
        from services.question_generation import (
            PapersRepository,
            get_papers_repository,
        )
        
        assert PapersRepository is not None
        assert callable(get_papers_repository)
    
    def test_paper_status_enum_values(self):
        """PaperStatus should have all required values"""
        from services.question_generation.models.paper import PaperStatus
        
        assert PaperStatus.GENERATING.value == "generating"
        assert PaperStatus.DRAFT.value == "draft"
        assert PaperStatus.COMPLETED.value == "completed"
        assert PaperStatus.PUBLISHED.value == "published"
        assert PaperStatus.ARCHIVED.value == "archived"
        assert PaperStatus.FAILED.value == "failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
