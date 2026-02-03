"""
Background Processor for Knowledge Base

Handles asynchronous processing of document uploads including:
- File storage to S3
- Text extraction (OCR for images)
- Embedding generation
- Vector storage in Qdrant
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from models.knowledge_base import (
    KnowledgeBaseUpload,
    UploadStatus,
    ProcessingMetrics,
)
from .knowledge_base_repository import KnowledgeBaseRepository
from .document_ingestion_service import get_document_ingestion_service
from utils.s3_storage import upload_file, is_s3_enabled, get_public_url

logger = logging.getLogger(__name__)


class BackgroundProcessor:
    """
    Processor for handling knowledge base uploads in the background.
    
    This class manages the complete lifecycle of document processing:
    1. Upload file to S3
    2. Extract text (OCR for images)
    3. Chunk and embed text
    4. Store embeddings in Qdrant
    5. Update MongoDB with results
    """
    
    def __init__(self, db_manager):
        """
        Initialize processor with database manager.
        
        Args:
            db_manager: DatabaseManager instance from app.state.db
        """
        self.db_manager = db_manager
        self.repository = KnowledgeBaseRepository(db_manager)
        self._ingestion_service = None
    
    async def _get_ingestion_service(self):
        """Get or create ingestion service instance."""
        if self._ingestion_service is None:
            self._ingestion_service = get_document_ingestion_service()
            await self._ingestion_service.initialize()
        return self._ingestion_service
    
    async def process_upload(
        self,
        upload_id: str,
        tenant_id: str,
        file_content: bytes,
        filename: str,
        mime_type: str,
    ) -> Dict[str, Any]:
        """
        Process a document upload in the background.
        
        This is the main entry point for background processing.
        
        Args:
            upload_id: Upload identifier
            tenant_id: Tenant identifier
            file_content: Raw file bytes
            filename: Original filename
            mime_type: MIME type
            
        Returns:
            Processing result dict
        """
        start_time = time.time()
        metrics = ProcessingMetrics()
        
        try:
            # Get upload record
            upload = await self.repository.get_by_id(tenant_id, upload_id)
            if upload is None:
                raise ValueError(f"Upload not found: {upload_id}")
            
            # Update status to processing
            await self.repository.update_status(
                tenant_id=tenant_id,
                upload_id=upload_id,
                status=UploadStatus.PROCESSING,
            )
            
            logger.info(f"🔄 Starting background processing for upload {upload_id}")
            
            # Step 1: Upload to S3 (if enabled)
            s3_key = None
            s3_url = None
            
            if is_s3_enabled():
                s3_start = time.time()
                s3_key = await self._upload_to_s3(
                    file_content=file_content,
                    filename=filename,
                    tenant_id=tenant_id,
                    upload_id=upload_id,
                )
                if s3_key:
                    s3_url = get_public_url(s3_key)
                    await self.repository.update_status(
                        tenant_id=tenant_id,
                        upload_id=upload_id,
                        status=UploadStatus.EXTRACTING_TEXT,
                        extra_updates={"s3_key": s3_key, "s3_url": s3_url},
                    )
                metrics.storage_time_ms = int((time.time() - s3_start) * 1000)
            
            # Step 2: Extract text and process
            extraction_start = time.time()
            
            ingestion_service = await self._get_ingestion_service()
            
            # Use ingestion service for text extraction, embedding, and storage
            result = await ingestion_service.ingest_file(
                file_content=file_content,
                filename=filename,
                mime_type=mime_type,
                tenant_id=tenant_id,
                teacher_id=upload.teacher_id,
                subject=upload.subject,
                grade=upload.grade,
                chapter=upload.chapter,
                topic=upload.topics[0] if upload.topics else None,
                upload_id=upload_id,
                use_ocr_for_images=True,
            )
            
            metrics.extraction_time_ms = int((time.time() - extraction_start) * 1000)
            metrics.embedding_time_ms = result.processing_time_ms - metrics.extraction_time_ms
            metrics.tokens_used = result.total_tokens
            metrics.total_time_ms = int((time.time() - start_time) * 1000)
            
            # Determine final status
            if result.success:
                final_status = UploadStatus.COMPLETED
                error_message = None
            elif result.partial_success:
                final_status = UploadStatus.PARTIALLY_COMPLETED
                error_message = "; ".join(result.errors) if result.errors else None
            else:
                final_status = UploadStatus.FAILED
                error_message = "; ".join(result.errors) if result.errors else "Processing failed"
            
            # Update with final results
            await self.repository.update_processing_result(
                tenant_id=tenant_id,
                upload_id=upload_id,
                extracted_text=result.extracted_text,
                total_chunks=result.total_chunks,
                chunks_embedded=result.chunks_embedded,
                chunks_stored=result.chunks_stored,
                qdrant_point_ids=result.point_ids,
                status=final_status,
                metrics=metrics.to_dict(),
                error_message=error_message,
            )
            
            logger.info(
                f"✅ Completed processing upload {upload_id}: "
                f"{result.chunks_stored} chunks stored in {metrics.total_time_ms}ms"
            )
            
            return {
                "success": result.success or result.partial_success,
                "upload_id": upload_id,
                "status": final_status.value,
                "chunks_stored": result.chunks_stored,
                "processing_time_ms": metrics.total_time_ms,
                "errors": result.errors,
            }
            
        except Exception as e:
            logger.error(f"❌ Background processing failed for {upload_id}: {e}", exc_info=True)
            
            # Update status to failed
            await self.repository.update_status(
                tenant_id=tenant_id,
                upload_id=upload_id,
                status=UploadStatus.FAILED,
                error_message=str(e),
            )
            
            return {
                "success": False,
                "upload_id": upload_id,
                "status": UploadStatus.FAILED.value,
                "error": str(e),
            }
    
    async def _upload_to_s3(
        self,
        file_content: bytes,
        filename: str,
        tenant_id: str,
        upload_id: str,
    ) -> Optional[str]:
        """
        Upload file to S3.
        
        Returns:
            S3 key if successful, None otherwise
        """
        try:
            # Generate S3 key
            extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
            s3_key = f"knowledge_base/{tenant_id}/{upload_id}/{filename}"
            
            # Upload to S3
            result = await upload_file(
                file_content=file_content,
                destination_path=s3_key,
            )
            
            if result:
                logger.info(f"📤 Uploaded to S3: {s3_key}")
                return s3_key
            else:
                logger.warning(f"Failed to upload to S3: {s3_key}")
                return None
                
        except Exception as e:
            logger.error(f"S3 upload error: {e}")
            return None
    
    async def retry_failed_upload(
        self,
        tenant_id: str,
        upload_id: str,
        file_content: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Retry processing a failed upload.
        
        If file_content is not provided, attempts to download from S3.
        
        Args:
            tenant_id: Tenant identifier
            upload_id: Upload identifier
            file_content: Optional file content (if not stored in S3)
            
        Returns:
            Processing result dict
        """
        try:
            # Get upload record
            upload = await self.repository.get_by_id(tenant_id, upload_id)
            if upload is None:
                raise ValueError(f"Upload not found: {upload_id}")
            
            if not upload.can_retry():
                return {
                    "success": False,
                    "error": "Upload cannot be retried (max retries exceeded)",
                }
            
            # Increment retry count
            upload.increment_retry()
            await self.repository.update(upload)
            
            # Get file content from S3 if not provided
            if file_content is None and upload.s3_key:
                from utils.s3_storage import download_file
                file_content = await download_file(upload.s3_key)
            
            if file_content is None:
                raise ValueError("File content not available for retry")
            
            # Process again
            return await self.process_upload(
                upload_id=upload_id,
                tenant_id=tenant_id,
                file_content=file_content,
                filename=upload.original_filename,
                mime_type=upload.mime_type,
            )
            
        except Exception as e:
            logger.error(f"Retry failed for {upload_id}: {e}")
            return {
                "success": False,
                "upload_id": upload_id,
                "error": str(e),
            }


async def process_upload_background(
    db_manager,
    upload_id: str,
    tenant_id: str,
    file_content: bytes,
    filename: str,
    mime_type: str,
) -> None:
    """
    Background task function for processing uploads.
    
    This function is designed to be called from FastAPI's BackgroundTasks.
    
    Args:
        db_manager: DatabaseManager instance
        upload_id: Upload identifier
        tenant_id: Tenant identifier
        file_content: Raw file bytes
        filename: Original filename
        mime_type: MIME type
    """
    processor = BackgroundProcessor(db_manager)
    await processor.process_upload(
        upload_id=upload_id,
        tenant_id=tenant_id,
        file_content=file_content,
        filename=filename,
        mime_type=mime_type,
    )
