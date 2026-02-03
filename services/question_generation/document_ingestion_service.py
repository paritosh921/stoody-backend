"""
Document Ingestion Service for Question Generation

Orchestrates the complete pipeline from file upload to vector storage:
1. Document processing (PDF, Word, Images)
2. OCR for handwritten/image content
3. Text chunking and embedding
4. Storage in Qdrant vector database

This service integrates:
- DocumentProcessor for file handling
- OCRService for image text extraction
- EmbeddingService for vectorization
- QdrantService for vector storage
"""

import asyncio
import logging
import time
import base64
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import uuid

from services.document_processor import get_document_processor, DocumentProcessor
from core.ocr_service import get_ocr_service, OCRService
from .embedding_service import get_embedding_service, EmbeddingService
from .qdrant_service import get_qdrant_service, QdrantService
from .models.embedding import ChunkMetadata, DocumentProcessingResult
from .models.knowledge_base import (
    UploadStatus,
    KnowledgeBaseUpload,
)

logger = logging.getLogger(__name__)


class DocumentIngestionError(Exception):
    """Raised when document ingestion fails"""
    pass


class DocumentIngestionResult:
    """Result of document ingestion process"""
    
    def __init__(
        self,
        upload_id: str,
        status: UploadStatus,
        extracted_text: str = "",
        total_chunks: int = 0,
        chunks_embedded: int = 0,
        chunks_stored: int = 0,
        point_ids: List[str] = None,
        total_tokens: int = 0,
        processing_time_ms: int = 0,
        errors: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.upload_id = upload_id
        self.status = status
        self.extracted_text = extracted_text
        self.total_chunks = total_chunks
        self.chunks_embedded = chunks_embedded
        self.chunks_stored = chunks_stored
        self.point_ids = point_ids or []
        self.total_tokens = total_tokens
        self.processing_time_ms = processing_time_ms
        self.errors = errors or []
        self.metadata = metadata or {}
    
    @property
    def success(self) -> bool:
        return self.status == UploadStatus.COMPLETED
    
    @property
    def partial_success(self) -> bool:
        return self.status == UploadStatus.PARTIALLY_COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "status": self.status.value,
            "extracted_text_length": len(self.extracted_text),
            "total_chunks": self.total_chunks,
            "chunks_embedded": self.chunks_embedded,
            "chunks_stored": self.chunks_stored,
            "point_ids": self.point_ids,
            "total_tokens": self.total_tokens,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class DocumentIngestionService:
    """
    Service for ingesting documents into the knowledge base.
    
    Handles the complete pipeline from file upload to vector storage,
    including OCR for image-based content.
    """
    
    # OCR prompt for extracting educational content
    OCR_PROMPT = """Extract ALL text from this image including:
- Mathematical equations and formulas (preserve LaTeX notation if possible)
- Scientific notation and symbols
- Diagrams labels and annotations
- Handwritten text (interpret carefully)
- Printed text
- Tables and structured content

Format the output as clean, readable text. Preserve paragraph structure.
For mathematical content, use standard notation (e.g., x^2 for x squared, sqrt(x) for square root).
"""
    
    def __init__(self):
        self._doc_processor: Optional[DocumentProcessor] = None
        self._ocr_service: Optional[OCRService] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._qdrant_service: Optional[QdrantService] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize all required services."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                # Get service instances
                self._doc_processor = get_document_processor()
                self._ocr_service = get_ocr_service()
                self._embedding_service = get_embedding_service()
                self._qdrant_service = get_qdrant_service()
                
                # Initialize async services
                await self._embedding_service.initialize()
                await self._qdrant_service.initialize()
                
                self._initialized = True
                logger.info("✅ DocumentIngestionService initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize DocumentIngestionService: {e}")
                raise
    
    async def ingest_file(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        tenant_id: str,
        teacher_id: str,
        subject: str,
        grade: str,
        chapter: Optional[str] = None,
        topic: Optional[str] = None,
        upload_id: Optional[str] = None,
        use_ocr_for_images: bool = True,
    ) -> DocumentIngestionResult:
        """
        Ingest a document file into the knowledge base.
        
        Complete pipeline:
        1. Process file to extract text (using DocumentProcessor or OCR)
        2. Chunk and embed the text
        3. Store embeddings in Qdrant
        
        Args:
            file_content: Binary file content
            filename: Original filename
            mime_type: MIME type of the file
            tenant_id: Tenant identifier
            teacher_id: Teacher who uploaded
            subject: Subject area
            grade: Grade/class level
            chapter: Optional chapter name
            topic: Optional topic name
            upload_id: Optional pre-generated upload ID
            use_ocr_for_images: Whether to use OCR for image files
            
        Returns:
            DocumentIngestionResult with processing details
        """
        await self.initialize()
        
        start_time = time.time()
        upload_id = upload_id or str(uuid.uuid4())
        errors = []
        
        try:
            # Step 1: Extract text from the file
            extracted_text, source_type, extraction_metadata = await self._extract_text(
                file_content=file_content,
                filename=filename,
                mime_type=mime_type,
                use_ocr_for_images=use_ocr_for_images,
            )
            
            if not extracted_text or not extracted_text.strip():
                return DocumentIngestionResult(
                    upload_id=upload_id,
                    status=UploadStatus.FAILED,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    errors=["No text could be extracted from the document"],
                    metadata=extraction_metadata,
                )
            
            logger.info(f"📝 Extracted {len(extracted_text)} chars from {filename}")
            
            # Step 2: Create metadata for chunks
            chunk_metadata = ChunkMetadata(
                tenant_id=tenant_id,
                teacher_id=teacher_id,
                source_file_id=upload_id,
                source_type=source_type,
                subject=subject,
                chapter=chapter,
                topic=topic,
                grade=grade,
                chunk_index=0,  # Will be updated during chunking
                total_chunks=1,  # Will be updated during chunking
            )
            
            # Step 3: Process document (chunk, embed, store)
            processing_result = await self._embedding_service.process_document(
                text=extracted_text,
                metadata=chunk_metadata,
                store_in_qdrant=True,
            )
            
            # Combine errors
            errors.extend(processing_result.errors)
            
            # Determine final status
            if processing_result.success:
                status = UploadStatus.COMPLETED
            elif processing_result.partial_success:
                status = UploadStatus.PARTIALLY_COMPLETED
            else:
                status = UploadStatus.FAILED
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return DocumentIngestionResult(
                upload_id=upload_id,
                status=status,
                extracted_text=extracted_text,
                total_chunks=processing_result.total_chunks,
                chunks_embedded=processing_result.chunks_embedded,
                chunks_stored=processing_result.chunks_stored,
                point_ids=processing_result.point_ids,
                total_tokens=processing_result.total_tokens,
                processing_time_ms=processing_time_ms,
                errors=errors,
                metadata={
                    **extraction_metadata,
                    "filename": filename,
                    "mime_type": mime_type,
                    "source_type": source_type,
                },
            )
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed for {filename}: {e}", exc_info=True)
            return DocumentIngestionResult(
                upload_id=upload_id,
                status=UploadStatus.FAILED,
                processing_time_ms=int((time.time() - start_time) * 1000),
                errors=[str(e)],
            )
    
    async def _extract_text(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        use_ocr_for_images: bool = True,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract text from a file.
        
        For PDFs and Word docs, uses DocumentProcessor.
        For images, optionally uses OCR service.
        
        Returns:
            Tuple of (extracted_text, source_type, metadata)
        """
        metadata = {}
        
        # Determine file type
        is_image = mime_type.startswith("image/")
        
        if is_image and use_ocr_for_images:
            # Use OCR for image files
            return await self._extract_with_ocr(file_content, filename, metadata)
        else:
            # Use DocumentProcessor for PDFs, Word, and images (as base64)
            return await self._extract_with_doc_processor(
                file_content, filename, mime_type, metadata
            )
    
    async def _extract_with_ocr(
        self,
        file_content: bytes,
        filename: str,
        metadata: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Extract text from image using OCR service."""
        try:
            # Convert to base64
            image_b64 = base64.b64encode(file_content).decode("utf-8")
            
            # Call OCR service
            result = await self._ocr_service.analyze_image(
                image_b64=image_b64,
                prompt=self.OCR_PROMPT,
            )
            
            if not result.get("success"):
                raise DocumentIngestionError(
                    f"OCR failed: {result.get('error', 'Unknown error')}"
                )
            
            text = result.get("text", "")
            metadata["ocr_provider"] = result.get("provider", "unknown")
            metadata["extraction_method"] = "ocr"
            
            # Determine if handwritten (based on OCR provider or heuristics)
            source_type = "handwritten" if "handwritten" in text.lower() else "image"
            
            return text, source_type, metadata
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise DocumentIngestionError(f"OCR extraction failed: {str(e)}")
    
    async def _extract_with_doc_processor(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        metadata: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Extract text using DocumentProcessor."""
        try:
            result = await self._doc_processor.process_file(
                file_content=file_content,
                filename=filename,
                mime_type=mime_type,
            )
            
            if not result.get("success"):
                raise DocumentIngestionError(
                    f"Document processing failed: {result.get('error', 'Unknown error')}"
                )
            
            text = result.get("text", "")
            file_type = result.get("file_type", "unknown")
            
            # Map file type to source type
            source_type_map = {
                "pdf": "pdf",
                "docx": "typed",
                "doc": "typed",
                "image": "image",
            }
            source_type = source_type_map.get(file_type, "typed")
            
            metadata["extraction_method"] = "document_processor"
            metadata["file_type"] = file_type
            metadata["char_count"] = result.get("char_count", 0)
            metadata["num_chunks_raw"] = result.get("num_chunks", 0)
            
            # For images processed by doc processor, the text is just a placeholder
            # We should use OCR instead
            if file_type == "image" and "[Image:" in text:
                # This is just image metadata, not actual text
                raise DocumentIngestionError(
                    "Image requires OCR for text extraction. Enable use_ocr_for_images=True"
                )
            
            return text, source_type, metadata
            
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise DocumentIngestionError(f"Document processing failed: {str(e)}")
    
    async def ingest_text(
        self,
        text: str,
        tenant_id: str,
        teacher_id: str,
        subject: str,
        grade: str,
        source_id: Optional[str] = None,
        source_type: str = "typed",
        chapter: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> DocumentIngestionResult:
        """
        Ingest pre-extracted text into the knowledge base.
        
        Use this when text is already available (e.g., from frontend OCR,
        copy-paste, or external extraction).
        
        Args:
            text: Pre-extracted text content
            tenant_id: Tenant identifier
            teacher_id: Teacher who provided the content
            subject: Subject area
            grade: Grade/class level
            source_id: Optional source identifier
            source_type: Type of source (typed, handwritten, etc.)
            chapter: Optional chapter name
            topic: Optional topic name
            
        Returns:
            DocumentIngestionResult with processing details
        """
        await self.initialize()
        
        start_time = time.time()
        upload_id = source_id or str(uuid.uuid4())
        
        if not text or not text.strip():
            return DocumentIngestionResult(
                upload_id=upload_id,
                status=UploadStatus.FAILED,
                processing_time_ms=int((time.time() - start_time) * 1000),
                errors=["Text content cannot be empty"],
            )
        
        try:
            # Create metadata for chunks
            chunk_metadata = ChunkMetadata(
                tenant_id=tenant_id,
                teacher_id=teacher_id,
                source_file_id=upload_id,
                source_type=source_type,
                subject=subject,
                chapter=chapter,
                topic=topic,
                grade=grade,
                chunk_index=0,
                total_chunks=1,
            )
            
            # Process document
            processing_result = await self._embedding_service.process_document(
                text=text,
                metadata=chunk_metadata,
                store_in_qdrant=True,
            )
            
            # Determine status
            if processing_result.success:
                status = UploadStatus.COMPLETED
            elif processing_result.partial_success:
                status = UploadStatus.PARTIALLY_COMPLETED
            else:
                status = UploadStatus.FAILED
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return DocumentIngestionResult(
                upload_id=upload_id,
                status=status,
                extracted_text=text,
                total_chunks=processing_result.total_chunks,
                chunks_embedded=processing_result.chunks_embedded,
                chunks_stored=processing_result.chunks_stored,
                point_ids=processing_result.point_ids,
                total_tokens=processing_result.total_tokens,
                processing_time_ms=processing_time_ms,
                errors=processing_result.errors,
                metadata={
                    "source_type": source_type,
                    "text_length": len(text),
                },
            )
            
        except Exception as e:
            logger.error(f"❌ Text ingestion failed: {e}", exc_info=True)
            return DocumentIngestionResult(
                upload_id=upload_id,
                status=UploadStatus.FAILED,
                processing_time_ms=int((time.time() - start_time) * 1000),
                errors=[str(e)],
            )
    
    async def search_knowledge_base(
        self,
        query: str,
        tenant_id: str,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        topic: Optional[str] = None,
        grade: Optional[str] = None,
        top_k: int = 10,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using semantic similarity.
        
        Args:
            query: Search query text
            tenant_id: Tenant identifier
            subject: Optional filter by subject
            chapter: Optional filter by chapter
            topic: Optional filter by topic
            grade: Optional filter by grade
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with text and metadata
        """
        await self.initialize()
        
        try:
            # Generate embedding for query
            query_result = await self._embedding_service.embed_text(query)
            
            # Build filter
            filter_conditions = []
            if subject:
                filter_conditions.append({"key": "subject", "match": {"value": subject}})
            if chapter:
                filter_conditions.append({"key": "chapter", "match": {"value": chapter}})
            if topic:
                filter_conditions.append({"key": "topic", "match": {"value": topic}})
            if grade:
                filter_conditions.append({"key": "grade", "match": {"value": grade}})
            
            query_filter = {"must": filter_conditions} if filter_conditions else None
            
            # Search Qdrant - use correct parameter names matching QdrantService.search()
            results = await self._qdrant_service.search(
                tenant_id=tenant_id,
                query_vector=query_result.embedding,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_conditions=query_filter,
            )
            
            # Results are already dicts with keys: id, score, text, metadata
            return [
                {
                    "id": r.get("id"),
                    "score": r.get("score"),
                    "text": r.get("text", ""),
                    "subject": r.get("metadata", {}).get("subject"),
                    "chapter": r.get("metadata", {}).get("chapter"),
                    "topic": r.get("metadata", {}).get("topic"),
                    "source_file_id": r.get("metadata", {}).get("source_file_id"),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}", exc_info=True)
            raise
    
    async def delete_document(
        self,
        tenant_id: str,
        source_file_id: str,
    ) -> int:
        """
        Delete a document and all its chunks from the knowledge base.
        
        Args:
            tenant_id: Tenant identifier
            source_file_id: ID of the source document to delete
            
        Returns:
            Number of points deleted
        """
        await self.initialize()
        
        collection_name = self._qdrant_service.get_collection_name(tenant_id)
        return await self._qdrant_service.delete_by_source(
            collection_name=collection_name,
            source_file_id=source_file_id,
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services."""
        results = {
            "document_ingestion": "healthy",
            "services": {},
        }
        
        try:
            await self.initialize()
            
            # Check embedding service
            embedding_health = await self._embedding_service.health_check()
            results["services"]["embedding"] = embedding_health
            
            # Check Qdrant service
            qdrant_health = await self._qdrant_service.health_check()
            results["services"]["qdrant"] = qdrant_health
            
            # Check if all services are healthy
            all_healthy = all(
                svc.get("status") == "healthy"
                for svc in results["services"].values()
            )
            
            results["document_ingestion"] = "healthy" if all_healthy else "degraded"
            
        except Exception as e:
            results["document_ingestion"] = "unhealthy"
            results["error"] = str(e)
        
        return results


# Singleton accessor
_document_ingestion_service: Optional[DocumentIngestionService] = None


def get_document_ingestion_service() -> DocumentIngestionService:
    """Get the singleton DocumentIngestionService instance."""
    global _document_ingestion_service
    if _document_ingestion_service is None:
        _document_ingestion_service = DocumentIngestionService()
    return _document_ingestion_service
