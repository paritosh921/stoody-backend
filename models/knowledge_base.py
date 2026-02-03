"""
Knowledge Base MongoDB Models

Data models for storing knowledge base uploads and their processing status in MongoDB.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class UploadStatus(str, Enum):
    """Status of a knowledge base upload"""
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTING_TEXT = "extracting_text"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class SourceType(str, Enum):
    """Type of source document"""
    PDF = "pdf"
    IMAGE = "image"
    HANDWRITTEN = "handwritten"
    TYPED = "typed"
    DOCX = "docx"


@dataclass
class ProcessingMetrics:
    """Metrics from document processing"""
    extraction_time_ms: int = 0
    embedding_time_ms: int = 0
    storage_time_ms: int = 0
    total_time_ms: int = 0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingMetrics':
        if data is None:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeBaseUpload:
    """
    A document uploaded to the knowledge base.
    
    This model tracks the complete lifecycle of a document from upload
    through OCR/text extraction, embedding generation, and storage in Qdrant.
    """
    # Identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    teacher_id: str = ""
    
    # File information
    original_filename: str = ""
    file_type: str = ""  # pdf, image, docx, txt
    file_size_bytes: int = 0
    mime_type: str = ""
    s3_key: Optional[str] = None  # S3 storage key
    s3_url: Optional[str] = None  # Public URL if available
    
    # Content extraction
    extracted_text: Optional[str] = None
    text_length: int = 0
    extraction_method: str = ""  # ocr, document_processor, direct
    ocr_provider: Optional[str] = None  # openai, mistral, etc.
    
    # Chunking and embedding
    total_chunks: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    qdrant_point_ids: List[str] = field(default_factory=list)
    
    # Classification
    subject: str = ""
    chapter: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    grade: str = ""
    source_type: str = SourceType.TYPED.value
    
    # Processing status
    status: str = UploadStatus.PENDING.value
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Processing metrics
    metrics: Optional[ProcessingMetrics] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage"""
        data = {
            "_id": self.id,
            "tenant_id": self.tenant_id,
            "teacher_id": self.teacher_id,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "s3_key": self.s3_key,
            "s3_url": self.s3_url,
            "extracted_text": self.extracted_text,
            "text_length": self.text_length,
            "extraction_method": self.extraction_method,
            "ocr_provider": self.ocr_provider,
            "total_chunks": self.total_chunks,
            "chunks_embedded": self.chunks_embedded,
            "chunks_stored": self.chunks_stored,
            "qdrant_point_ids": self.qdrant_point_ids,
            "subject": self.subject,
            "chapter": self.chapter,
            "topics": self.topics,
            "grade": self.grade,
            "source_type": self.source_type,
            "status": self.status,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processing_started_at": self.processing_started_at,
            "processing_completed_at": self.processing_completed_at,
            "metadata": self.metadata,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeBaseUpload':
        """Create instance from MongoDB document"""
        if data is None:
            raise ValueError("Cannot create KnowledgeBaseUpload from None")
        
        # Handle _id field
        if "_id" in data:
            data["id"] = data.pop("_id")
        
        # Handle metrics
        if "metrics" in data and data["metrics"]:
            data["metrics"] = ProcessingMetrics.from_dict(data["metrics"])
        
        # Filter to only known fields
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered_data)
    
    def update_status(self, status: UploadStatus, error: Optional[str] = None) -> None:
        """Update processing status"""
        self.status = status.value
        self.updated_at = datetime.utcnow()
        
        if status == UploadStatus.PROCESSING:
            self.processing_started_at = datetime.utcnow()
        elif status in (UploadStatus.COMPLETED, UploadStatus.FAILED, UploadStatus.PARTIALLY_COMPLETED):
            self.processing_completed_at = datetime.utcnow()
        
        if error:
            self.error_message = error
    
    def can_retry(self) -> bool:
        """Check if processing can be retried"""
        return self.retry_count < self.max_retries and self.status == UploadStatus.FAILED.value
    
    def increment_retry(self) -> None:
        """Increment retry counter"""
        self.retry_count += 1
        self.updated_at = datetime.utcnow()
    
    @property
    def is_completed(self) -> bool:
        return self.status == UploadStatus.COMPLETED.value
    
    @property
    def is_failed(self) -> bool:
        return self.status == UploadStatus.FAILED.value
    
    @property
    def is_processing(self) -> bool:
        return self.status in (
            UploadStatus.PROCESSING.value,
            UploadStatus.EXTRACTING_TEXT.value,
            UploadStatus.EMBEDDING.value,
            UploadStatus.STORING.value,
        )


@dataclass
class KnowledgeBaseStats:
    """Statistics for a tenant's knowledge base"""
    tenant_id: str
    total_uploads: int = 0
    completed_uploads: int = 0
    failed_uploads: int = 0
    pending_uploads: int = 0
    total_chunks: int = 0
    total_text_chars: int = 0
    total_tokens_used: int = 0
    uploads_by_subject: Dict[str, int] = field(default_factory=dict)
    uploads_by_status: Dict[str, int] = field(default_factory=dict)
    last_upload_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
