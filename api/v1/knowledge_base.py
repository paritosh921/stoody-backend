"""
Knowledge Base API Router

Provides endpoints for managing the knowledge base (notes upload, embedding, and search).

Endpoints:
- POST /upload: Upload and process a document (synchronous)
- POST /upload-async: Upload and process a document (background)
- POST /upload-text: Upload pre-extracted text
- POST /search: Semantic search in the knowledge base
- GET /uploads: List uploaded documents
- GET /uploads/{upload_id}: Get upload details
- DELETE /documents/{document_id}: Delete a document
- GET /stats: Get knowledge base statistics
- GET /upload-stats: Get MongoDB upload statistics
- GET /health: Health check for services
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.question_generation.embedding_service import get_embedding_service, EmbeddingService
from services.question_generation.qdrant_service import get_qdrant_service, QdrantService
from services.question_generation.document_ingestion_service import (
    get_document_ingestion_service,
    DocumentIngestionService,
)
from services.question_generation.knowledge_base_repository import KnowledgeBaseRepository
from services.question_generation.background_processor import process_upload_background
from services.question_generation.models.embedding import (
    ChunkMetadata,
    SearchQuery,
    SearchResult,
)
from services.question_generation.models.knowledge_base import (
    UploadStatus,
    KnowledgeBaseUpload,
    KnowledgeBaseUploadCreate,
    KnowledgeBaseSearchRequest,
    KnowledgeBaseSearchResult,
)
from models.knowledge_base import (
    KnowledgeBaseUpload as KBUploadModel,
    UploadStatus as KBUploadStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])


# ============================================================================
# Request/Response Models
# ============================================================================

class UploadResponse(BaseModel):
    """Response from document upload"""
    success: bool
    upload_id: Optional[str] = None
    message: str
    status: str = UploadStatus.PENDING.value
    total_chunks: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    processing_time_ms: int = 0
    errors: List[str] = []
    extracted_content: Optional[str] = Field(default=None, description="Extracted text content from the document")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "upload_id": "upload_abc123",
                "message": "Document processed successfully",
                "status": "completed",
                "total_chunks": 15,
                "chunks_embedded": 15,
                "chunks_stored": 15,
                "processing_time_ms": 2345,
                "extracted_content": "Chapter 1: Introduction to Physics..."
            }
        }
    }


class SearchResponse(BaseModel):
    """Response from knowledge base search"""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: int
    error: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "query": "Newton's second law of motion",
                "results": [
                    {
                        "id": "point_123",
                        "score": 0.92,
                        "text": "Newton's second law states that F = ma...",
                        "metadata": {
                            "subject": "Physics",
                            "chapter": "Laws of Motion"
                        }
                    }
                ],
                "total_results": 1,
                "search_time_ms": 45
            }
        }
    }


class CollectionStatsResponse(BaseModel):
    """Response with collection statistics"""
    success: bool
    tenant_id: str
    collection_name: str
    total_points: int = 0
    indexed_points: int = 0
    status: str = "unknown"
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response from health check"""
    status: str
    services: Dict[str, Dict[str, Any]]
    timestamp: str


class UploadDetailsResponse(BaseModel):
    """Response with upload details"""
    success: bool
    upload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class UploadListResponse(BaseModel):
    """Response with list of uploads"""
    success: bool
    uploads: List[Dict[str, Any]]
    total: int
    skip: int
    limit: int


class UploadStatsResponse(BaseModel):
    """Response with upload statistics from MongoDB"""
    success: bool
    tenant_id: str
    total_uploads: int = 0
    completed_uploads: int = 0
    failed_uploads: int = 0
    pending_uploads: int = 0
    total_chunks: int = 0
    uploads_by_subject: Dict[str, int] = {}
    uploads_by_status: Dict[str, int] = {}
    error: Optional[str] = None


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_embedding_svc() -> EmbeddingService:
    """Get embedding service instance."""
    svc = get_embedding_service()
    await svc.initialize()
    return svc


async def get_qdrant_svc() -> QdrantService:
    """Get Qdrant service instance."""
    svc = get_qdrant_service()
    await svc.initialize()
    return svc


async def get_ingestion_svc() -> DocumentIngestionService:
    """Get document ingestion service instance."""
    svc = get_document_ingestion_service()
    await svc.initialize()
    return svc


def get_repository(request: Request) -> KnowledgeBaseRepository:
    """Get knowledge base repository instance."""
    db_manager = request.app.state.db
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return KnowledgeBaseRepository(db_manager)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, image, etc.)"),
    tenant_id: str = Form(..., description="Tenant identifier"),
    teacher_id: str = Form(..., description="Teacher identifier"),
    subject: str = Form(..., description="Subject area"),
    grade: str = Form(..., description="Grade/class level"),
    chapter: Optional[str] = Form(default=None, description="Chapter name"),
    topic: Optional[str] = Form(default=None, description="Topic name"),
    use_ocr: bool = Form(default=True, description="Use OCR for image files"),
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_svc),
):
    """
    Upload and process a document for the knowledge base.
    
    This endpoint accepts a document file along with metadata, extracts text
    (using OCR for images), chunks the text, generates embeddings, and stores
    them in Qdrant for semantic search.
    
    Supported file types: PDF, PNG, JPG, JPEG, WEBP, TXT, DOCX
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        file_content = await file.read()
        
        # Get MIME type
        mime_type = file.content_type or "application/octet-stream"
        
        # Process document using ingestion service
        result = await ingestion_svc.ingest_file(
            file_content=file_content,
            filename=file.filename,
            mime_type=mime_type,
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            subject=subject,
            grade=grade,
            chapter=chapter,
            topic=topic,
            use_ocr_for_images=use_ocr,
        )
        
        return UploadResponse(
            success=result.success or result.partial_success,
            upload_id=result.upload_id,
            message="Document processed successfully" if result.success else (
                "Document partially processed" if result.partial_success else "Document processing failed"
            ),
            status=result.status.value,
            total_chunks=result.total_chunks,
            chunks_embedded=result.chunks_embedded,
            chunks_stored=result.chunks_stored,
            processing_time_ms=result.processing_time_ms,
            errors=result.errors,
            extracted_content=result.extracted_text,
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload-text", response_model=UploadResponse)
async def upload_text(
    tenant_id: str = Form(..., description="Tenant identifier"),
    teacher_id: str = Form(..., description="Teacher identifier"),
    subject: str = Form(..., description="Subject area"),
    grade: str = Form(..., description="Grade/class level"),
    text: str = Form(..., description="Text content to index"),
    source_id: Optional[str] = Form(default=None, description="Source document ID"),
    source_type: str = Form(default="typed", description="Source type (typed, handwritten, etc.)"),
    chapter: Optional[str] = Form(default=None, description="Chapter name"),
    topic: Optional[str] = Form(default=None, description="Topic name"),
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_svc),
):
    """
    Upload pre-extracted text directly to the knowledge base.
    
    This is a simpler endpoint that accepts plain text (e.g., from OCR output)
    and processes it for embedding and storage.
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
        
        # Process text using ingestion service
        result = await ingestion_svc.ingest_text(
            text=text,
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            subject=subject,
            grade=grade,
            source_id=source_id,
            source_type=source_type,
            chapter=chapter,
            topic=topic,
        )
        
        return UploadResponse(
            success=result.success or result.partial_success,
            upload_id=result.upload_id,
            message="Text processed successfully" if result.success else (
                "Text partially processed" if result.partial_success else "Text processing failed"
            ),
            status=result.status.value,
            total_chunks=result.total_chunks,
            chunks_embedded=result.chunks_embedded,
            chunks_stored=result.chunks_stored,
            processing_time_ms=result.processing_time_ms,
            errors=result.errors,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text upload failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(
    request: KnowledgeBaseSearchRequest,
    tenant_id: str = Query(..., description="Tenant identifier"),
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_svc),
):
    """
    Search the knowledge base using semantic similarity.
    
    Takes a query and optional filters, generates an embedding for the query,
    and searches Qdrant for the most similar documents.
    """
    start_time = time.time()
    
    try:
        # Search using ingestion service
        search_results = await ingestion_svc.search_knowledge_base(
            query=request.query,
            tenant_id=tenant_id,
            subject=request.subject,
            chapter=request.chapter,
            topic=request.topic,
            grade=request.grade,
            top_k=request.top_k,
            score_threshold=request.min_score,
        )
        
        search_time_ms = int((time.time() - start_time) * 1000)
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        search_time_ms = int((time.time() - start_time) * 1000)
        return SearchResponse(
            success=False,
            query=request.query,
            results=[],
            total_results=0,
            search_time_ms=search_time_ms,
            error=str(e),
        )


@router.delete("/documents/{source_file_id}")
async def delete_document(
    source_file_id: str,
    tenant_id: str = Query(..., description="Tenant identifier"),
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_svc),
):
    """
    Delete a document and all its chunks from the knowledge base.
    """
    try:
        deleted_count = await ingestion_svc.delete_document(
            tenant_id=tenant_id,
            source_file_id=source_file_id,
        )
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} points for document {source_file_id}",
            "deleted_count": deleted_count,
        }
        
    except Exception as e:
        logger.error(f"Delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.get("/stats", response_model=CollectionStatsResponse)
async def get_collection_stats(
    tenant_id: str = Query(..., description="Tenant identifier"),
    qdrant_svc: QdrantService = Depends(get_qdrant_svc),
):
    """
    Get statistics for a tenant's knowledge base collection.
    """
    try:
        collection_name = qdrant_svc.get_collection_name(tenant_id)
        stats = await qdrant_svc.get_collection_stats(collection_name)
        
        return CollectionStatsResponse(
            success=True,
            tenant_id=tenant_id,
            collection_name=collection_name,
            total_points=stats.get("total_points", 0),
            indexed_points=stats.get("indexed_points", 0),
            status=stats.get("status", "unknown"),
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        return CollectionStatsResponse(
            success=False,
            tenant_id=tenant_id,
            collection_name=qdrant_svc.get_collection_name(tenant_id),
            error=str(e),
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_svc),
):
    """
    Health check for all knowledge base services.
    """
    from datetime import datetime
    
    health_result = await ingestion_svc.health_check()
    
    return HealthCheckResponse(
        status=health_result.get("document_ingestion", "unknown"),
        services=health_result.get("services", {}),
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post("/collections/create")
async def create_collection(
    tenant_id: str = Query(..., description="Tenant identifier"),
    qdrant_svc: QdrantService = Depends(get_qdrant_svc),
):
    """
    Create a new collection for a tenant.
    
    Collections are automatically created during document upload, but this
    endpoint allows pre-creating collections.
    """
    try:
        collection_name = qdrant_svc.get_collection_name(tenant_id)
        created = await qdrant_svc.create_collection(collection_name)
        
        return {
            "success": True,
            "collection_name": collection_name,
            "created": created,
            "message": "Collection created" if created else "Collection already exists",
        }
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


# ============================================================================
# Background Upload Endpoints
# ============================================================================

@router.post("/upload-async", response_model=UploadResponse)
async def upload_document_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (PDF, image, etc.)"),
    tenant_id: str = Form(..., description="Tenant identifier"),
    teacher_id: str = Form(..., description="Teacher identifier"),
    subject: str = Form(..., description="Subject area"),
    grade: str = Form(..., description="Grade/class level"),
    chapter: Optional[str] = Form(default=None, description="Chapter name"),
    topics: Optional[str] = Form(default=None, description="Comma-separated topics"),
):
    """
    Upload a document for background processing.
    
    This endpoint immediately returns with a pending status while the document
    is processed in the background. Use GET /uploads/{upload_id} to check status.
    
    This is the preferred method for large files or when immediate response is needed.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        file_content = await file.read()
        
        # Get MIME type
        mime_type = file.content_type or "application/octet-stream"
        
        # Generate upload ID
        upload_id = str(uuid.uuid4())
        
        # Determine file type
        file_extension = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        
        # Parse topics
        topic_list = [t.strip() for t in topics.split(",")] if topics else []
        
        # Create upload record in MongoDB
        repository = get_repository(request)
        upload = KBUploadModel(
            id=upload_id,
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            original_filename=file.filename,
            file_type=file_extension,
            file_size_bytes=len(file_content),
            mime_type=mime_type,
            subject=subject,
            chapter=chapter,
            topics=topic_list,
            grade=grade,
            status=KBUploadStatus.PENDING.value,
        )
        
        await repository.create(upload)
        
        # Schedule background processing
        db_manager = request.app.state.db
        background_tasks.add_task(
            process_upload_background,
            db_manager=db_manager,
            upload_id=upload_id,
            tenant_id=tenant_id,
            file_content=file_content,
            filename=file.filename,
            mime_type=mime_type,
        )
        
        logger.info(f"📤 Queued upload {upload_id} for background processing")
        
        return UploadResponse(
            success=True,
            upload_id=upload_id,
            message="Document queued for processing",
            status=KBUploadStatus.PENDING.value,
            processing_time_ms=0,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ============================================================================
# Upload Management Endpoints
# ============================================================================

@router.get("/uploads", response_model=UploadListResponse)
async def list_uploads(
    request: Request,
    tenant_id: str = Query(..., description="Tenant identifier"),
    teacher_id: Optional[str] = Query(default=None, description="Filter by teacher"),
    subject: Optional[str] = Query(default=None, description="Filter by subject"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum records to return"),
):
    """
    List uploaded documents with optional filters.
    
    Returns a paginated list of uploads for the specified tenant.
    """
    try:
        repository = get_repository(request)
        
        # Convert status string to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = KBUploadStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        uploads = await repository.list_uploads(
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            subject=subject,
            status=status_enum,
            skip=skip,
            limit=limit,
        )
        
        total = await repository.count_uploads(
            tenant_id=tenant_id,
            teacher_id=teacher_id,
            subject=subject,
            status=status_enum,
        )
        
        return UploadListResponse(
            success=True,
            uploads=[u.to_dict() for u in uploads],
            total=total,
            skip=skip,
            limit=limit,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list uploads: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")


@router.get("/uploads/{upload_id}", response_model=UploadDetailsResponse)
async def get_upload_details(
    request: Request,
    upload_id: str,
    tenant_id: str = Query(..., description="Tenant identifier"),
):
    """
    Get details of a specific upload.
    
    Use this endpoint to check the status of an async upload.
    """
    try:
        repository = get_repository(request)
        upload = await repository.get_by_id(tenant_id, upload_id)
        
        if upload is None:
            return UploadDetailsResponse(
                success=False,
                error=f"Upload not found: {upload_id}",
            )
        
        # Convert to dict but exclude large text content
        upload_dict = upload.to_dict()
        if upload_dict.get("extracted_text"):
            # Include just the length, not the full text
            upload_dict["extracted_text_preview"] = upload_dict["extracted_text"][:500] + "..." if len(upload_dict["extracted_text"]) > 500 else upload_dict["extracted_text"]
            upload_dict["extracted_text"] = None  # Remove full text from response
        
        return UploadDetailsResponse(
            success=True,
            upload=upload_dict,
        )
        
    except Exception as e:
        logger.error(f"Failed to get upload details: {e}", exc_info=True)
        return UploadDetailsResponse(
            success=False,
            error=str(e),
        )


@router.get("/upload-stats", response_model=UploadStatsResponse)
async def get_upload_stats(
    request: Request,
    tenant_id: str = Query(..., description="Tenant identifier"),
):
    """
    Get upload statistics from MongoDB.
    
    Returns aggregated statistics about uploads including counts by status and subject.
    """
    try:
        repository = get_repository(request)
        stats = await repository.get_stats(tenant_id)
        
        return UploadStatsResponse(
            success=True,
            tenant_id=tenant_id,
            total_uploads=stats.total_uploads,
            completed_uploads=stats.completed_uploads,
            failed_uploads=stats.failed_uploads,
            pending_uploads=stats.pending_uploads,
            total_chunks=stats.total_chunks,
            uploads_by_subject=stats.uploads_by_subject,
            uploads_by_status=stats.uploads_by_status,
        )
        
    except Exception as e:
        logger.error(f"Failed to get upload stats: {e}", exc_info=True)
        return UploadStatsResponse(
            success=False,
            tenant_id=tenant_id,
            error=str(e),
        )


@router.post("/uploads/{upload_id}/retry")
async def retry_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    upload_id: str,
    tenant_id: str = Query(..., description="Tenant identifier"),
):
    """
    Retry processing a failed upload.
    
    The file content must still be available in S3.
    """
    try:
        repository = get_repository(request)
        upload = await repository.get_by_id(tenant_id, upload_id)
        
        if upload is None:
            raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")
        
        if not upload.can_retry():
            raise HTTPException(
                status_code=400, 
                detail=f"Upload cannot be retried. Status: {upload.status}, Retries: {upload.retry_count}/{upload.max_retries}"
            )
        
        if not upload.s3_key:
            raise HTTPException(
                status_code=400,
                detail="File content not available in S3 for retry"
            )
        
        # Download file from S3
        from utils.s3_storage import download_file
        file_content = await download_file(upload.s3_key)
        
        if file_content is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to download file from S3"
            )
        
        # Increment retry count
        upload.increment_retry()
        await repository.update(upload)
        
        # Schedule background processing
        db_manager = request.app.state.db
        background_tasks.add_task(
            process_upload_background,
            db_manager=db_manager,
            upload_id=upload_id,
            tenant_id=tenant_id,
            file_content=file_content,
            filename=upload.original_filename,
            mime_type=upload.mime_type,
        )
        
        return {
            "success": True,
            "message": f"Retry queued for upload {upload_id}",
            "retry_count": upload.retry_count,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retry upload: {str(e)}")


@router.delete("/uploads/{upload_id}")
async def delete_upload(
    request: Request,
    upload_id: str,
    tenant_id: str = Query(..., description="Tenant identifier"),
    delete_vectors: bool = Query(default=True, description="Also delete vectors from Qdrant"),
    ingestion_svc: DocumentIngestionService = Depends(get_ingestion_svc),
):
    """
    Delete an upload record and optionally its vectors from Qdrant.
    """
    try:
        repository = get_repository(request)
        
        # Get upload to check it exists
        upload = await repository.get_by_id(tenant_id, upload_id)
        if upload is None:
            raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")
        
        # Delete vectors from Qdrant if requested
        deleted_vectors = 0
        if delete_vectors:
            deleted_vectors = await ingestion_svc.delete_document(
                tenant_id=tenant_id,
                source_file_id=upload_id,
            )
        
        # Delete from S3 if exists
        if upload.s3_key:
            from utils.s3_storage import delete_file
            await delete_file(upload.s3_key)
        
        # Delete from MongoDB
        await repository.delete(tenant_id, upload_id)
        
        return {
            "success": True,
            "message": f"Upload {upload_id} deleted",
            "vectors_deleted": deleted_vectors,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete upload: {str(e)}")
