"""
Knowledge Base Models

Models for managing uploaded documents in the knowledge base.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class UploadStatus(str, Enum):
    """Status of a knowledge base upload"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


class KnowledgeBaseUpload(BaseModel):
    """A document uploaded to the knowledge base"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., description="Tenant identifier")
    teacher_id: str = Field(..., description="Teacher who uploaded")
    
    # File information
    original_filename: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type: pdf, image, docx")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    s3_path: Optional[str] = Field(default=None, description="S3 storage path")
    
    # Extracted content
    extracted_text: Optional[str] = Field(default=None, description="OCR/extracted text")
    total_chunks: int = Field(default=0, ge=0, description="Number of text chunks")
    qdrant_point_ids: List[str] = Field(default_factory=list, description="Qdrant point IDs")
    
    # Classification
    subject: str = Field(..., description="Subject area")
    chapter: Optional[str] = Field(default=None, description="Chapter name")
    topics: List[str] = Field(default_factory=list, description="Topics covered")
    grade: str = Field(..., description="Grade/class level")
    
    # Status
    status: UploadStatus = Field(default=UploadStatus.PENDING)
    error_message: Optional[str] = Field(default=None, description="Error if failed")
    
    # Timestamps
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(default=None)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_mongo_dict(self) -> Dict[str, Any]:
        """Convert to MongoDB document format"""
        return {
            "_id": self.id,
            "tenant_id": self.tenant_id,
            "teacher_id": self.teacher_id,
            "original_filename": self.original_filename,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "s3_path": self.s3_path,
            "extracted_text": self.extracted_text,
            "total_chunks": self.total_chunks,
            "qdrant_point_ids": self.qdrant_point_ids,
            "subject": self.subject,
            "chapter": self.chapter,
            "topics": self.topics,
            "grade": self.grade,
            "status": self.status.value,
            "error_message": self.error_message,
            "uploaded_at": self.uploaded_at,
            "processed_at": self.processed_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_mongo_dict(cls, data: Dict[str, Any]) -> "KnowledgeBaseUpload":
        """Create from MongoDB document"""
        if "_id" in data:
            data["id"] = data.pop("_id")
        if "status" in data and isinstance(data["status"], str):
            data["status"] = UploadStatus(data["status"])
        return cls(**data)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "upload_abc123",
                "tenant_id": "AAAA-1234",
                "teacher_id": "teacher_xyz",
                "original_filename": "chapter5_notes.pdf",
                "file_type": "pdf",
                "file_size_bytes": 1234567,
                "subject": "Physics",
                "chapter": "Laws of Motion",
                "topics": ["Newton's First Law", "Newton's Second Law"],
                "grade": "Class 11",
                "status": "completed"
            }
        }
    }


class KnowledgeBaseUploadCreate(BaseModel):
    """Request model for creating a new upload"""
    subject: str = Field(..., min_length=1, max_length=100, description="Subject area")
    grade: str = Field(..., min_length=1, max_length=50, description="Grade/class level")
    chapter: Optional[str] = Field(default=None, max_length=200, description="Chapter name")
    topics: Optional[List[str]] = Field(default=None, description="Topics covered")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "Physics",
                "grade": "Class 11",
                "chapter": "Laws of Motion",
                "topics": ["Newton's Laws", "Friction"]
            }
        }
    }


class KnowledgeBaseSearchRequest(BaseModel):
    """Request model for searching the knowledge base"""
    query: str = Field(..., min_length=1, max_length=5000, description="Search query")
    subject: Optional[str] = Field(default=None, description="Filter by subject")
    chapter: Optional[str] = Field(default=None, description="Filter by chapter")
    topic: Optional[str] = Field(default=None, description="Filter by topic")
    grade: Optional[str] = Field(default=None, description="Filter by grade")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    min_score: float = Field(default=0.5, ge=0, le=1, description="Minimum similarity score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Newton's second law of motion",
                "subject": "Physics",
                "grade": "Class 11",
                "top_k": 10
            }
        }
    }


class KnowledgeBaseSearchResult(BaseModel):
    """Result from knowledge base search"""
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results returned")
    search_time_ms: int = Field(default=0, description="Search time in milliseconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Newton's second law",
                "results": [
                    {
                        "id": "point_123",
                        "score": 0.92,
                        "text": "Newton's second law states that F = ma...",
                        "subject": "Physics",
                        "chapter": "Laws of Motion"
                    }
                ],
                "total_results": 1,
                "search_time_ms": 45
            }
        }
    }
