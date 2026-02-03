"""
Qdrant Vector Database Service

Handles all interactions with Qdrant for storing and retrieving
document embeddings for the Question Generation System.

Features:
- Async operations using thread pool
- Multi-tenant collection management
- Semantic search with filtering
- Batch upsert operations
- Connection pooling and retry logic
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config_async import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_PREFIX,
    EMBEDDING_DIMENSIONS,
)

logger = logging.getLogger(__name__)


class QdrantServiceError(Exception):
    """Base exception for Qdrant service errors"""
    pass


class QdrantConnectionError(QdrantServiceError):
    """Failed to connect to Qdrant"""
    pass


class QdrantCollectionError(QdrantServiceError):
    """Error with collection operations"""
    pass


class QdrantService:
    """
    Service for managing vector embeddings in Qdrant.
    
    Provides:
    - Connection management with lazy initialization
    - Collection creation per tenant
    - Embedding storage and retrieval
    - Semantic search with filtering
    - Batch operations
    """
    
    # Default distance metric for similarity search
    DISTANCE_METRIC = qdrant_models.Distance.COSINE
    
    # Maximum points per batch upsert
    BATCH_SIZE = 100
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.0
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_prefix: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Initialize the Qdrant service.
        
        Args:
            url: Qdrant server URL (defaults to config)
            api_key: API key for authentication (defaults to config)
            collection_prefix: Prefix for collection names (defaults to config)
            vector_size: Size of embedding vectors (defaults to config)
        """
        self._url = url or QDRANT_URL
        self._api_key = api_key or QDRANT_API_KEY
        self._collection_prefix = collection_prefix or QDRANT_COLLECTION_PREFIX
        self._vector_size = vector_size or EMBEDDING_DIMENSIONS
        
        self._client: Optional[QdrantClient] = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
        logger.info(
            f"QdrantService configured: url={self._url[:50]}..., "
            f"prefix={self._collection_prefix}, vector_size={self._vector_size}"
        )
    
    async def initialize(self) -> bool:
        """
        Initialize the Qdrant client connection.
        
        Returns:
            True if initialization was successful
            
        Raises:
            QdrantConnectionError: If connection fails
        """
        async with self._lock:
            if self._initialized:
                return True
            
            try:
                # Create client in thread pool (sync operation)
                loop = asyncio.get_event_loop()
                self._client = await loop.run_in_executor(
                    None,
                    lambda: QdrantClient(
                        url=self._url,
                        api_key=self._api_key,
                        timeout=30,
                        prefer_grpc=False,
                    )
                )
                
                # Test connection by getting collections
                await self._run_sync(lambda: self._client.get_collections())
                
                self._initialized = True
                logger.info("✅ Qdrant client initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Qdrant client: {e}")
                raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}")
    
    async def _run_sync(self, func) -> Any:
        """Run a synchronous function in the thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)
    
    async def _ensure_initialized(self) -> None:
        """Ensure the client is initialized"""
        if not self._initialized:
            await self.initialize()
    
    def _get_collection_name(self, tenant_id: str) -> str:
        """
        Get the collection name for a tenant.
        
        Args:
            tenant_id: The tenant identifier
            
        Returns:
            Full collection name
        """
        # Sanitize tenant_id for collection name
        safe_tenant = tenant_id.replace("-", "_").lower()
        return f"{self._collection_prefix}_{safe_tenant}"

    def get_collection_name(self, tenant_id: str) -> str:
        """
        Public method to get the collection name for a tenant.
        
        Args:
            tenant_id: The tenant identifier
            
        Returns:
            Full collection name
        """
        return self._get_collection_name(tenant_id)
    
    async def collection_exists(self, tenant_id: str) -> bool:
        """
        Check if a collection exists for a tenant.
        
        Args:
            tenant_id: The tenant identifier
            
        Returns:
            True if collection exists
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        try:
            result = await self._run_sync(
                lambda: self._client.collection_exists(collection_name)
            )
            return result
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    async def create_collection(
        self,
        tenant_id: str,
        recreate: bool = False,
    ) -> bool:
        """
        Create a collection for a tenant.
        
        Args:
            tenant_id: The tenant identifier
            recreate: If True, delete existing collection first
            
        Returns:
            True if collection was created successfully
            
        Raises:
            QdrantCollectionError: If creation fails
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        try:
            # Check if already exists
            exists = await self.collection_exists(tenant_id)
            
            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    await self._run_sync(
                        lambda: self._client.delete_collection(collection_name)
                    )
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    return True
            
            # Create collection
            await self._run_sync(
                lambda: self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self._vector_size,
                        distance=self.DISTANCE_METRIC,
                    ),
                    # Optimize for accuracy
                    hnsw_config=qdrant_models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                    ),
                    # Enable payload indexing for filtering
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=10000,
                    ),
                )
            )
            
            # Create payload indexes for common filter fields
            await self._create_payload_indexes(collection_name)
            
            logger.info(f"✅ Created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection {collection_name}: {e}")
            raise QdrantCollectionError(f"Failed to create collection: {e}")
    
    async def _create_payload_indexes(self, collection_name: str) -> None:
        """Create indexes on payload fields for efficient filtering"""
        index_fields = [
            ("subject", qdrant_models.PayloadSchemaType.KEYWORD),
            ("chapter", qdrant_models.PayloadSchemaType.KEYWORD),
            ("topic", qdrant_models.PayloadSchemaType.KEYWORD),
            ("grade", qdrant_models.PayloadSchemaType.KEYWORD),
            ("teacher_id", qdrant_models.PayloadSchemaType.KEYWORD),
            ("source_file_id", qdrant_models.PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, field_type in index_fields:
            try:
                await self._run_sync(
                    lambda fn=field_name, ft=field_type: self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=fn,
                        field_schema=ft,
                    )
                )
            except Exception as e:
                # Index might already exist
                logger.debug(f"Could not create index for {field_name}: {e}")
    
    async def delete_collection(self, tenant_id: str) -> bool:
        """
        Delete a tenant's collection.
        
        Args:
            tenant_id: The tenant identifier
            
        Returns:
            True if deleted successfully
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        try:
            await self._run_sync(
                lambda: self._client.delete_collection(collection_name)
            )
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
    async def upsert_points(
        self,
        tenant_id: str,
        points: List[Dict[str, Any]],
    ) -> Tuple[int, List[str]]:
        """
        Upsert points (embeddings) into the collection.
        
        Args:
            tenant_id: The tenant identifier
            points: List of points, each with 'id', 'vector', and 'payload'
            
        Returns:
            Tuple of (number of points upserted, list of point IDs)
            
        Raises:
            QdrantServiceError: If upsert fails
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        # Ensure collection exists
        if not await self.collection_exists(tenant_id):
            await self.create_collection(tenant_id)
        
        try:
            # Convert to Qdrant PointStruct
            qdrant_points = []
            point_ids = []
            
            for point in points:
                point_id = point.get("id") or str(uuid.uuid4())
                point_ids.append(point_id)
                
                qdrant_points.append(
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=point["vector"],
                        payload=point.get("payload", {}),
                    )
                )
            
            # Batch upsert
            total_upserted = 0
            for i in range(0, len(qdrant_points), self.BATCH_SIZE):
                batch = qdrant_points[i:i + self.BATCH_SIZE]
                
                await self._run_sync(
                    lambda b=batch: self._client.upsert(
                        collection_name=collection_name,
                        points=b,
                        wait=True,
                    )
                )
                
                total_upserted += len(batch)
                logger.debug(f"Upserted batch: {total_upserted}/{len(qdrant_points)}")
            
            logger.info(f"✅ Upserted {total_upserted} points to {collection_name}")
            return total_upserted, point_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert points: {e}")
            raise QdrantServiceError(f"Failed to upsert points: {e}")
    
    async def search(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: float = 0.5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the collection.
        
        Args:
            tenant_id: The tenant identifier
            query_vector: The query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Qdrant filter conditions
            
        Returns:
            List of search results with id, score, and payload
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        # Check if collection exists
        if not await self.collection_exists(tenant_id):
            logger.warning(f"Collection does not exist: {collection_name}")
            return []
        
        try:
            # Build filter
            qdrant_filter = None
            if filter_conditions:
                qdrant_filter = qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key=cond["key"],
                            match=qdrant_models.MatchValue(value=cond["match"]["value"])
                            if "value" in cond["match"]
                            else qdrant_models.MatchAny(any=cond["match"]["any"]),
                        )
                        for cond in filter_conditions.get("must", [])
                    ]
                )
            
            # Perform search using query_points (new Qdrant API)
            response = await self._run_sync(
                lambda: self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            )
            
            # Convert to dict - response.points contains the results
            results = response.points if hasattr(response, 'points') else response
            return [
                {
                    "id": str(result.id),
                    "score": result.score,
                    "text": result.payload.get("text", "") if result.payload else "",
                    "metadata": {
                        k: v for k, v in (result.payload.items() if result.payload else []) if k != "text"
                    },
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise QdrantServiceError(f"Search failed: {e}")
    
    async def delete_by_source(
        self,
        tenant_id: str,
        source_file_id: str,
    ) -> int:
        """
        Delete all points from a specific source file.
        
        Args:
            tenant_id: The tenant identifier
            source_file_id: The source file ID
            
        Returns:
            Number of points deleted
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        if not await self.collection_exists(tenant_id):
            return 0
        
        try:
            # First, count points to delete
            count_result = await self._run_sync(
                lambda: self._client.count(
                    collection_name=collection_name,
                    count_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="source_file_id",
                                match=qdrant_models.MatchValue(value=source_file_id),
                            )
                        ]
                    ),
                )
            )
            count = count_result.count
            
            if count == 0:
                return 0
            
            # Delete points
            await self._run_sync(
                lambda: self._client.delete(
                    collection_name=collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="source_file_id",
                                    match=qdrant_models.MatchValue(value=source_file_id),
                                )
                            ]
                        )
                    ),
                )
            )
            
            logger.info(f"Deleted {count} points for source {source_file_id}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete by source: {e}")
            raise QdrantServiceError(f"Failed to delete by source: {e}")
    
    async def get_collection_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get statistics about a tenant's collection.
        
        Args:
            tenant_id: The tenant identifier
            
        Returns:
            Collection statistics
        """
        await self._ensure_initialized()
        collection_name = self._get_collection_name(tenant_id)
        
        if not await self.collection_exists(tenant_id):
            return {
                "exists": False,
                "collection_name": collection_name,
            }
        
        try:
            info = await self._run_sync(
                lambda: self._client.get_collection(collection_name)
            )
            
            return {
                "exists": True,
                "collection_name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value if info.status else "unknown",
                "vector_size": self._vector_size,
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "exists": True,
                "collection_name": collection_name,
                "error": str(e),
            }
    
    async def health_check(self) -> bool:
        """
        Check if the Qdrant service is healthy.
        
        Returns:
            True if healthy
        """
        try:
            await self._ensure_initialized()
            await self._run_sync(lambda: self._client.get_collections())
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the Qdrant client connection"""
        if self._client:
            try:
                await self._run_sync(lambda: self._client.close())
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
            finally:
                self._client = None
                self._initialized = False
                logger.info("Qdrant client closed")


# ============================================================================
# Singleton instance
# ============================================================================

_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service() -> QdrantService:
    """
    Get the singleton Qdrant service instance.
    
    Returns:
        QdrantService instance
    """
    global _qdrant_service
    
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    
    return _qdrant_service


async def initialize_qdrant_service() -> QdrantService:
    """
    Initialize and return the Qdrant service.
    
    Returns:
        Initialized QdrantService instance
    """
    service = get_qdrant_service()
    await service.initialize()
    return service
