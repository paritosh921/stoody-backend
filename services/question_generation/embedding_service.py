"""
Embedding Service for Question Generation

Handles text chunking and OpenAI embedding generation for semantic search.
"""

import asyncio
import logging
import time
from typing import List, Optional, Tuple
import httpx
from openai import AsyncOpenAI

from config_async import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .models.embedding import (
    DocumentChunk,
    ChunkMetadata,
    EmbeddingResult,
    DocumentProcessingResult,
)
from .qdrant_service import get_qdrant_service

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text splitting into overlapping chunks for embedding."""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Uses sentence-aware splitting to avoid breaking mid-sentence.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = text.strip()
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        prev_start = -1
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to find a good break point (sentence or paragraph boundary)
            if end < text_length:
                # Look for sentence boundaries within the last 20% of the chunk
                search_start = start + int(self.chunk_size * 0.8)
                best_break = end
                
                # Priority order: paragraph break, period, newline, comma, space
                for separator, offset in [('\n\n', 2), ('. ', 2), ('.\n', 2), ('\n', 1), (', ', 2), (' ', 1)]:
                    pos = text.rfind(separator, search_start, end)
                    if pos > search_start:
                        best_break = pos + offset
                        break
                
                end = best_break
            
            # Extract chunk and clean it
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position with overlap
            if end >= text_length:
                break
            
            prev_start = start
            start = end - self.chunk_overlap
            
            # Ensure we make progress (avoid infinite loop)
            if start <= prev_start:
                start = end
        
        return chunks
    
    def create_document_chunks(
        self,
        text: str,
        metadata: ChunkMetadata,
    ) -> List[DocumentChunk]:
        """
        Create DocumentChunk objects from text.
        
        Args:
            text: The full document text
            metadata: Base metadata for all chunks
            
        Returns:
            List of DocumentChunk objects
        """
        text_chunks = self.chunk_text(text)
        total_chunks = len(text_chunks)
        
        document_chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            # Create chunk-specific metadata
            chunk_metadata = ChunkMetadata(
                tenant_id=metadata.tenant_id,
                teacher_id=metadata.teacher_id,
                source_file_id=metadata.source_file_id,
                source_type=metadata.source_type,
                subject=metadata.subject,
                chapter=metadata.chapter,
                topic=metadata.topic,
                grade=metadata.grade,
                chunk_index=idx,
                total_chunks=total_chunks,
            )
            
            document_chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )
        
        return document_chunks


class EmbeddingService:
    """
    Service for generating embeddings using OpenAI's API.
    
    Features:
    - Lazy initialization of OpenAI client
    - Batch embedding for efficiency
    - Connection pooling with httpx
    - Automatic retry with exponential backoff
    """
    
    _instance: Optional['EmbeddingService'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Configuration
        self.model = EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        self.batch_size = EMBEDDING_BATCH_SIZE
        self.api_key = OPENAI_API_KEY
        
        # Chunker instance
        self.chunker = TextChunker()
        
        logger.info(f"EmbeddingService created with model={self.model}, dimensions={self.dimensions}")
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client with connection pooling."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                # Create HTTP client with connection pooling
                self._http_client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=50,
                        keepalive_expiry=30,
                    ),
                    timeout=httpx.Timeout(60.0),  # Longer timeout for embeddings
                )
                
                # Initialize OpenAI client
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    http_client=self._http_client,
                )
                
                self._initialized = True
                logger.info("✅ EmbeddingService initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize EmbeddingService: {e}")
                raise
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with the generated embedding
        """
        await self.initialize()
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            response = await self.client.embeddings.create(
                input=text.strip(),
                model=self.model,
                dimensions=self.dimensions,
            )
            
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                tokens_used=tokens_used,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def embed_texts_batch(
        self,
        texts: List[str],
        max_retries: int = 3,
    ) -> Tuple[List[EmbeddingResult], int]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum retry attempts per batch
            
        Returns:
            Tuple of (list of EmbeddingResults, total tokens used)
        """
        await self.initialize()
        
        if not texts:
            return [], 0
        
        # Filter empty texts and track indices
        valid_texts = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
        
        if not valid_texts:
            return [], 0
        
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)
        total_tokens = 0
        
        # Process in batches
        for batch_start in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[batch_start:batch_start + self.batch_size]
            batch_indices = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            
            # Retry logic
            for attempt in range(max_retries):
                try:
                    response = await self.client.embeddings.create(
                        input=batch_texts,
                        model=self.model,
                        dimensions=self.dimensions,
                    )
                    
                    tokens_used = response.usage.total_tokens if response.usage else 0
                    total_tokens += tokens_used
                    
                    # Map embeddings back to original indices
                    for data_item, original_idx, text in zip(response.data, batch_indices, batch_texts):
                        results[original_idx] = EmbeddingResult(
                            text=text,
                            embedding=data_item.embedding,
                            model=self.model,
                            dimensions=len(data_item.embedding),
                            tokens_used=tokens_used // len(batch),  # Approximate per-text tokens
                        )
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to embed batch after {max_retries} attempts: {e}")
                        raise
                    
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Embedding batch failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
        
        # Filter out None values (should not happen if all texts were valid)
        final_results = [r for r in results if r is not None]
        
        return final_results, total_tokens
    
    async def embed_document_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> Tuple[List[DocumentChunk], int]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Tuple of (chunks with embeddings, total tokens used)
        """
        if not chunks:
            return [], 0
        
        texts = [chunk.text for chunk in chunks]
        embedding_results, total_tokens = await self.embed_texts_batch(texts)
        
        # Map embeddings back to chunks
        result_map = {r.text: r.embedding for r in embedding_results}
        
        embedded_chunks = []
        for chunk in chunks:
            if chunk.text in result_map:
                chunk.embedding = result_map[chunk.text]
                embedded_chunks.append(chunk)
            else:
                logger.warning(f"No embedding found for chunk {chunk.id}")
        
        return embedded_chunks, total_tokens
    
    async def process_document(
        self,
        text: str,
        metadata: ChunkMetadata,
        store_in_qdrant: bool = True,
    ) -> DocumentProcessingResult:
        """
        Full pipeline: chunk text, generate embeddings, and optionally store in Qdrant.
        
        Args:
            text: Full document text
            metadata: Base metadata for the document
            store_in_qdrant: Whether to store embeddings in Qdrant
            
        Returns:
            DocumentProcessingResult with processing details
        """
        start_time = time.time()
        errors = []
        
        try:
            # Step 1: Chunk the text
            chunks = self.chunker.create_document_chunks(text, metadata)
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                return DocumentProcessingResult(
                    source_file_id=metadata.source_file_id,
                    total_chunks=0,
                    chunks_embedded=0,
                    chunks_stored=0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    errors=["No chunks created from document - text may be empty"],
                )
            
            logger.info(f"Created {total_chunks} chunks from document {metadata.source_file_id}")
            
            # Step 2: Generate embeddings
            embedded_chunks, total_tokens = await self.embed_document_chunks(chunks)
            chunks_embedded = len(embedded_chunks)
            
            logger.info(f"Generated {chunks_embedded} embeddings using {total_tokens} tokens")
            
            # Step 3: Store in Qdrant (if requested)
            chunks_stored = 0
            point_ids = []
            
            if store_in_qdrant and embedded_chunks:
                try:
                    qdrant_service = get_qdrant_service()
                    await qdrant_service.initialize()

                    # Prepare points for upsert
                    points = [chunk.to_qdrant_point() for chunk in embedded_chunks]
                    point_ids = [p["id"] for p in points]

                    # Upsert to Qdrant - pass tenant_id, NOT collection_name
                    # (upsert_points internally calls _get_collection_name)
                    chunks_stored, _ = await qdrant_service.upsert_points(metadata.tenant_id, points)

                    collection_name = qdrant_service.get_collection_name(metadata.tenant_id)
                    logger.info(f"Stored {chunks_stored} points in Qdrant collection {collection_name}")
                    
                except Exception as e:
                    error_msg = f"Failed to store in Qdrant: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            return DocumentProcessingResult(
                source_file_id=metadata.source_file_id,
                total_chunks=total_chunks,
                chunks_embedded=chunks_embedded,
                chunks_stored=chunks_stored,
                point_ids=point_ids,
                total_tokens=total_tokens,
                processing_time_ms=int((time.time() - start_time) * 1000),
                errors=errors,
            )
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return DocumentProcessingResult(
                source_file_id=metadata.source_file_id,
                total_chunks=0,
                chunks_embedded=0,
                chunks_stored=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                errors=[str(e)],
            )
    
    async def health_check(self) -> dict:
        """Check if the embedding service is healthy."""
        try:
            await self.initialize()
            
            # Test with a simple embedding
            result = await self.embed_text("test")
            
            return {
                "status": "healthy",
                "model": self.model,
                "dimensions": result.dimensions,
                "initialized": self._initialized,
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized,
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        
        self.client = None
        self._initialized = False
        logger.info("EmbeddingService closed")


# Singleton accessor
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton EmbeddingService instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
