"""
Embedding Models for Question Generation

Models for text chunks, embeddings, and search results.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class ChunkMetadata(BaseModel):
    """Metadata associated with a text chunk"""
    tenant_id: str = Field(..., description="Tenant identifier")
    teacher_id: Optional[str] = Field(default=None, description="Teacher who uploaded")
    source_file_id: str = Field(..., description="ID of the source file")
    source_type: str = Field(..., description="Type: pdf, image, handwritten, typed")
    subject: str = Field(..., description="Subject area")
    chapter: Optional[str] = Field(default=None, description="Chapter name")
    topic: Optional[str] = Field(default=None, description="Specific topic")
    grade: str = Field(..., description="Grade/class level")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk in the document")
    total_chunks: int = Field(..., ge=1, description="Total chunks in the document")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant-compatible payload dict"""
        return {
            "tenant_id": self.tenant_id,
            "teacher_id": self.teacher_id,
            "source_file_id": self.source_file_id,
            "source_type": self.source_type,
            "subject": self.subject,
            "chapter": self.chapter,
            "topic": self.topic,
            "grade": self.grade,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "created_at": self.created_at.isoformat(),
        }


class DocumentChunk(BaseModel):
    """A chunk of text from a document"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, description="The text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    
    @property
    def has_embedding(self) -> bool:
        return self.embedding is not None and len(self.embedding) > 0
    
    def to_qdrant_point(self) -> Dict[str, Any]:
        """Convert to Qdrant point format"""
        if not self.has_embedding:
            raise ValueError("Cannot create Qdrant point without embedding")
        
        payload = self.metadata.to_qdrant_payload()
        payload["text"] = self.text
        
        return {
            "id": self.id,
            "vector": self.embedding,
            "payload": payload,
        }


class EmbeddingResult(BaseModel):
    """Result of embedding generation"""
    text: str = Field(..., description="Original text")
    embedding: List[float] = Field(..., description="Generated embedding vector")
    model: str = Field(..., description="Model used for embedding")
    dimensions: int = Field(..., description="Embedding dimensions")
    tokens_used: int = Field(default=0, description="Tokens consumed")


class SearchResult(BaseModel):
    """Result from semantic search"""
    id: str = Field(..., description="Point ID")
    score: float = Field(..., ge=0, le=1, description="Similarity score")
    text: str = Field(..., description="Retrieved text content")
    metadata: Dict[str, Any] = Field(..., description="Associated metadata")
    
    @property
    def subject(self) -> str:
        return self.metadata.get("subject", "")
    
    @property
    def chapter(self) -> Optional[str]:
        return self.metadata.get("chapter")
    
    @property
    def topic(self) -> Optional[str]:
        return self.metadata.get("topic")
    
    @property
    def source_file_id(self) -> str:
        return self.metadata.get("source_file_id", "")


class SearchQuery(BaseModel):
    """Query for semantic search"""
    query_text: str = Field(..., min_length=1, max_length=5000, description="Search query")
    
    # Filters
    subject: Optional[str] = Field(default=None, description="Filter by subject")
    chapter: Optional[str] = Field(default=None, description="Filter by chapter")
    topic: Optional[str] = Field(default=None, description="Filter by topic")
    grade: Optional[str] = Field(default=None, description="Filter by grade")
    source_file_ids: Optional[List[str]] = Field(default=None, description="Filter by source files")
    
    # Search options
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    score_threshold: float = Field(default=0.5, ge=0, le=1, description="Minimum similarity score")
    
    def get_qdrant_filter(self) -> Optional[Dict[str, Any]]:
        """Build Qdrant filter from query parameters"""
        conditions = []
        
        if self.subject:
            conditions.append({
                "key": "subject",
                "match": {"value": self.subject}
            })
        
        if self.chapter:
            conditions.append({
                "key": "chapter",
                "match": {"value": self.chapter}
            })
        
        if self.topic:
            conditions.append({
                "key": "topic",
                "match": {"value": self.topic}
            })
        
        if self.grade:
            conditions.append({
                "key": "grade",
                "match": {"value": self.grade}
            })
        
        if self.source_file_ids:
            conditions.append({
                "key": "source_file_id",
                "match": {"any": self.source_file_ids}
            })
        
        if not conditions:
            return None
        
        return {"must": conditions}


class DocumentProcessingResult(BaseModel):
    """Result of processing a document for embedding"""
    source_file_id: str = Field(..., description="ID of the processed file")
    total_chunks: int = Field(..., ge=0, description="Number of chunks created")
    chunks_embedded: int = Field(..., ge=0, description="Number of chunks embedded")
    chunks_stored: int = Field(..., ge=0, description="Number of chunks stored in Qdrant")
    point_ids: List[str] = Field(default_factory=list, description="Qdrant point IDs")
    total_tokens: int = Field(default=0, description="Total tokens used for embeddings")
    processing_time_ms: int = Field(default=0, description="Processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    
    @property
    def success(self) -> bool:
        return self.chunks_stored > 0 and len(self.errors) == 0
    
    @property
    def partial_success(self) -> bool:
        return self.chunks_stored > 0 and len(self.errors) > 0
