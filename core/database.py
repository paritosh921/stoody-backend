"""
Async Database Manager for SkillBot
Handles ChromaDB and MongoDB connections with connection pooling
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
import chromadb
from chromadb.config import Settings as ChromaSettings
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
import time
import certifi

from config_async import (
    CHROMADB_PATH,
    CHROMADB_COLLECTION_NAME,
    MONGODB_URL,
    MONGODB_DB_NAME,
    DISABLE_MONGODB,
    settings
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database connections with async support"""

    def __init__(self):
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.chroma_collection = None
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.mongo_db: Optional[AsyncIOMotorDatabase] = None
        self._chroma_lock = asyncio.Lock()
        self._mongo_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize all database connections"""
        try:
            # Initialize ChromaDB (non-blocking - failure won't stop startup)
            try:
                await self._init_chromadb()
            except Exception as chroma_error:
                logger.warning(f"⚠️  ChromaDB initialization failed (continuing without it): {str(chroma_error)}")
                self.chroma_client = None
                self.chroma_collection = None

            # Initialize MongoDB (if configured and not disabled)
            if MONGODB_URL and not DISABLE_MONGODB:
                await self._init_mongodb()

            logger.info("✅ Database manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize database manager: {str(e)}")
            raise

    async def _init_chromadb(self) -> None:
        """Initialize ChromaDB with thread safety"""
        async with self._chroma_lock:
            try:
                # Create ChromaDB client in thread executor to avoid blocking
                loop = asyncio.get_event_loop()

                def create_chroma_client():
                    return chromadb.PersistentClient(
                        path=str(CHROMADB_PATH),
                        settings=ChromaSettings(
                            anonymized_telemetry=False,
                            # Allow reset to gracefully handle mixed ChromaDB versions/schemas
                            allow_reset=True,
                            is_persistent=True,
                        )
                    )

                self.chroma_client = await loop.run_in_executor(None, create_chroma_client)

                # Get or create collection
                def get_or_create_collection():
                    return self.chroma_client.get_or_create_collection(
                        name=CHROMADB_COLLECTION_NAME,
                        metadata={"description": "SkillBot async questions collection"}
                    )

                self.chroma_collection = await loop.run_in_executor(None, get_or_create_collection)

                logger.info(f"✅ ChromaDB initialized - Collection: {CHROMADB_COLLECTION_NAME}")

            except Exception as e:
                logger.error(f"❌ ChromaDB initialization failed: {str(e)}")
                raise

    async def _init_mongodb(self) -> None:
        """Initialize MongoDB with connection pooling"""
        async with self._mongo_lock:
            try:
                # Skip MongoDB entirely if disabled or not configured
                if not MONGODB_URL or DISABLE_MONGODB:
                    logger.info("MongoDB disabled or not configured; skipping initialization")
                    self.mongo_client = None
                    self.mongo_db = None
                    return
                # Create MongoDB client with async motor
                # Reduced timeouts to fail fast and not hang the UI
                min_pool_size = max(0, settings.MONGODB_MIN_POOL_SIZE)
                max_pool_size = max(1, settings.MONGODB_MAX_POOL_SIZE)

                if min_pool_size > max_pool_size:
                    logger.warning(
                        "Configured MongoDB min pool size (%d) exceeds max pool size (%d); trimming min to match max",
                        min_pool_size,
                        max_pool_size
                    )
                    min_pool_size = max_pool_size

                self.mongo_client = AsyncIOMotorClient(
                    MONGODB_URL,
                    minPoolSize=min_pool_size,
                    maxPoolSize=max_pool_size,
                    maxIdleTimeMS=30000,
                    serverSelectionTimeoutMS=5000,  # Reduced from 15s to 5s
                    connectTimeoutMS=5000,  # Reduced from 15s to 5s
                    socketTimeoutMS=10000,  # Reduced from 45s to 10s
                    waitQueueTimeoutMS=5000,  # Reduced from 20s to 5s
                    tls=True,
                    tlsCAFile=certifi.where(),
                )

                # Get database instance
                self.mongo_db = self.mongo_client[MONGODB_DB_NAME]

                # Test connection
                await self.mongo_client.admin.command('ping')

                logger.info(f"✅ MongoDB initialized - Database: {MONGODB_DB_NAME}")

                # Ensure indexes (unique constraints, performance)
                try:
                    await self.ensure_indexes()
                except Exception as e:
                    logger.warning(f"⚠️ Failed to ensure indexes: {str(e)}")

            except ServerSelectionTimeoutError:
                logger.warning("⚠️ MongoDB connection failed - continuing without MongoDB")
                self.mongo_client = None
                self.mongo_db = None
            except Exception as e:
                logger.error(f"❌ MongoDB initialization failed: {str(e)}")
                raise

    async def get_chroma_collection(self):
        """Get ChromaDB collection with lazy initialization"""
        if self.chroma_collection is None:
            await self._init_chromadb()
        return self.chroma_collection

    async def get_mongo_db(self) -> Optional[AsyncIOMotorDatabase]:
        """Get MongoDB database with lazy initialization"""
        if self.mongo_db is None and MONGODB_URL and not DISABLE_MONGODB:
            await self._init_mongodb()
        return self.mongo_db

    async def get_mongo_collection(self, collection_name: str):
        """Get MongoDB collection"""
        db = await self.get_mongo_db()
        if db is None:
            return None
        return db[collection_name]

    async def ensure_indexes(self) -> None:
        """Create necessary indexes if they don't exist."""
        if self.mongo_db is None:
            return
        try:
            students = self.mongo_db["students"]
            # Unique index on business key student_id
            await students.create_index(
                [("student_id", 1)],
                unique=True,
                name="uniq_student_id"
            )
            logger.info("✅ Ensured unique index on students.student_id")

            # Tutor indexes
            tutors = self.mongo_db["tutors"]
            await tutors.create_index(
                [("username", 1)],
                unique=True,
                name="uniq_tutor_username"
            )
            await tutors.create_index(
                [("tutor_id", 1)],
                unique=True,
                name="uniq_tutor_id"
            )
            logger.info("✅ Ensured indexes on tutors.username and tutors.tutor_id")
        except OperationFailure as e:
            # Likely existing duplicates preventing index creation
            logger.warning(f"⚠️ Could not create one or more indexes: {str(e)}")

    # ChromaDB async operations
    async def chroma_add(self, ids: List[str], documents: List[str],
                        metadatas: Optional[List[Dict[str, Any]]] = None,
                        embeddings: Optional[List[List[float]]] = None) -> bool:
        """Add documents to ChromaDB asynchronously"""
        try:
            collection = await self.get_chroma_collection()
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                lambda: collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB add failed: {str(e)}")
            return False

    async def chroma_query(self, query_texts: List[str], n_results: int = 10,
                          where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query ChromaDB asynchronously"""
        try:
            collection = await self.get_chroma_collection()
            loop = asyncio.get_event_loop()

            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where
                )
            )
            return results
        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    async def chroma_get(self, ids: Optional[List[str]] = None,
                        where: Optional[Dict[str, Any]] = None,
                        limit: Optional[int] = None) -> Dict[str, Any]:
        """Get documents from ChromaDB asynchronously"""
        try:
            collection = await self.get_chroma_collection()
            loop = asyncio.get_event_loop()

            results = await loop.run_in_executor(
                None,
                lambda: collection.get(ids=ids, where=where, limit=limit)
            )
            return results
        except Exception as e:
            logger.error(f"ChromaDB get failed: {str(e)}")
            return {"ids": [], "documents": [], "metadatas": []}

    async def chroma_delete(self, ids: Optional[List[str]] = None,
                           where: Optional[Dict[str, Any]] = None) -> bool:
        """Delete documents from ChromaDB asynchronously"""
        try:
            collection = await self.get_chroma_collection()
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                None,
                lambda: collection.delete(ids=ids, where=where)
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {str(e)}")
            return False

    async def chroma_count(self) -> int:
        """Get document count from ChromaDB asynchronously"""
        try:
            collection = await self.get_chroma_collection()
            loop = asyncio.get_event_loop()

            count = await loop.run_in_executor(
                None,
                lambda: collection.count()
            )
            return count
        except Exception as e:
            logger.error(f"ChromaDB count failed: {str(e)}")
            return 0

    async def chroma_reset(self) -> bool:
        """Reset ChromaDB collection (delete all documents)"""
        try:
            if not self.chroma_client:
                logger.warning("ChromaDB client not initialized")
                return False

            loop = asyncio.get_event_loop()

            # Delete the collection and recreate it
            await loop.run_in_executor(
                None,
                lambda: self.chroma_client.delete_collection(
                    name=CHROMADB_COLLECTION_NAME
                )
            )

            # Recreate the collection
            self.chroma_collection = await loop.run_in_executor(
                None,
                lambda: self.chroma_client.get_or_create_collection(
                    name=CHROMADB_COLLECTION_NAME,
                    metadata={"description": "SkillBot async questions collection"}
                )
            )

            logger.info(f"✅ ChromaDB collection reset - {CHROMADB_COLLECTION_NAME}")
            return True

        except Exception as e:
            logger.error(f"ChromaDB reset failed: {str(e)}")
            return False

    # MongoDB async operations wrapper
    async def mongo_find_one(self, collection_name: str, filter_dict: Dict[str, Any],
                            projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find one document in MongoDB"""
        try:
            # Quick check if MongoDB is available
            if self.mongo_client is None or self.mongo_db is None:
                logger.warning(f"MongoDB not connected - cannot query {collection_name} collection")
                return None

            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return None
            return await collection.find_one(filter_dict, projection)
        except Exception as e:
            logger.error(f"MongoDB find_one failed: {str(e)}")
            return None

    async def mongo_find(self, collection_name: str, filter_dict: Dict[str, Any],
                        projection: Optional[Dict[str, Any]] = None,
                        sort: Optional[List[tuple]] = None,
                        limit: Optional[int] = None,
                        skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents in MongoDB"""
        try:
            # Quick check if MongoDB is available
            if self.mongo_client is None or self.mongo_db is None:
                logger.warning(f"MongoDB not connected - cannot query {collection_name} collection")
                return []

            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return []

            cursor = collection.find(filter_dict, projection)

            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            # Use a reasonable max limit to avoid hanging on large collections
            max_length = limit if limit is not None else 1000
            return await cursor.to_list(length=max_length)
        except Exception as e:
            logger.error(f"MongoDB find failed: {str(e)}")
            return []

    async def mongo_insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert one document to MongoDB"""
        try:
            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return None
            result = await collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB insert_one failed: {str(e)}")
            return None

    async def mongo_insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Insert many documents to MongoDB"""
        try:
            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return None
            result = await collection.insert_many(documents)
            return [str(id_) for id_ in result.inserted_ids]
        except Exception as e:
            logger.error(f"MongoDB insert_many failed: {str(e)}")
            return None

    async def mongo_update_one(self, collection_name: str, filter_dict: Dict[str, Any],
                              update_dict: Dict[str, Any], upsert: bool = False) -> bool:
        """Update one document in MongoDB"""
        try:
            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return False
            result = await collection.update_one(filter_dict, update_dict, upsert=upsert)
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
        except Exception as e:
            logger.error(f"MongoDB update_one failed: {str(e)}")
            return False

    async def mongo_delete_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete one document from MongoDB"""
        try:
            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return False
            result = await collection.delete_one(filter_dict)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"MongoDB delete_one failed: {str(e)}")
            return False

    async def mongo_delete_many(self, collection_name: str, filter_dict: Dict[str, Any]) -> int:
        """Delete multiple documents from MongoDB"""
        try:
            collection = await self.get_mongo_collection(collection_name)
            if collection is None:
                return 0
            result = await collection.delete_many(filter_dict)
            return result.deleted_count
        except Exception as e:
            logger.error(f"MongoDB delete_many failed: {str(e)}")
            return 0

    async def health_check(self) -> bool:
        """Check health of all database connections"""
        try:
            # Check ChromaDB
            chroma_healthy = False
            try:
                collection = await self.get_chroma_collection()
                if collection:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: collection.count()
                    )
                    chroma_healthy = True
            except Exception as e:
                logger.error(f"ChromaDB health check failed: {str(e)}")

            # Check MongoDB only if enabled
            mongo_healthy = False
            if not DISABLE_MONGODB and self.mongo_client:
                try:
                    await asyncio.wait_for(
                        self.mongo_client.admin.command('ping'),
                        timeout=5.0
                    )
                    mongo_healthy = True
                except Exception as e:
                    logger.error(f"MongoDB health check failed: {str(e)}")
                    # Close client to stop background threads from retrying
                    try:
                        self.mongo_client.close()
                    except Exception:
                        pass
                    self.mongo_client = None
                    self.mongo_db = None
                    mongo_healthy = False

            # Consider overall healthy if ChromaDB is healthy (Mongo is optional)
            return chroma_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    async def close(self) -> None:
        """Close all database connections"""
        try:
            # Close MongoDB connection
            if self.mongo_client:
                self.mongo_client.close()
                logger.info("🔌 MongoDB connection closed")

            # ChromaDB doesn't need explicit closing
            logger.info("🔌 Database connections closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
