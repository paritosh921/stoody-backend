"""
Chat MongoDB Client - MongoDB-based storage for chat conversations
Replaces ChromaDB for conversation memory and document storage.
Uses sentence-transformers for embeddings and MongoDB for persistence.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)

# Lazy load embeddings to avoid startup overhead
_embeddings_model = None
_embeddings_lock = asyncio.Lock()


async def get_embeddings_model():
    """Lazy load the sentence-transformers model for embeddings"""
    global _embeddings_model
    if _embeddings_model is None:
        async with _embeddings_lock:
            if _embeddings_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    # Use the same model as ChromaDB for consistency
                    _embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    logger.info("✅ Sentence Transformers model loaded for embeddings")
                except ImportError:
                    logger.warning("⚠️ sentence-transformers not available, semantic search disabled")
                    _embeddings_model = None
    return _embeddings_model


def compute_embedding(text: str, model) -> Optional[List[float]]:
    """Compute embedding for a text using sentence-transformers"""
    if model is None:
        return None
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error computing embedding: {e}")
        return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    try:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception:
        return 0.0


class ChatMongoDBClient:
    """
    MongoDB client for managing chat conversations and RAG documents.
    Provides similar interface to ChatChromaDBClient for easy migration.
    
    Collections used:
    - chat_conversations: Stores conversation messages with embeddings
    - chat_documents: Stores uploaded document chunks for RAG
    """
    
    CONVERSATIONS_COLLECTION = "chat_conversations"
    DOCUMENTS_COLLECTION = "chat_documents"
    
    def __init__(self, db_manager=None):
        """
        Initialize chat MongoDB client
        
        Args:
            db_manager: DatabaseManager instance (if None, will get from app state)
        """
        self._db_manager = db_manager
        self._embeddings_model = None
        self._initialized = False
        
    async def _ensure_initialized(self):
        """Ensure the client is initialized with database manager and embeddings"""
        if self._initialized:
            return
            
        # Get database manager if not provided
        if self._db_manager is None:
            try:
                from main_async import app
                if hasattr(app, 'state') and hasattr(app.state, 'db'):
                    self._db_manager = app.state.db
            except Exception as e:
                logger.warning(f"Could not get database manager from app state: {e}")
        
        # Load embeddings model
        self._embeddings_model = await get_embeddings_model()
        
        # Ensure indexes exist for efficient querying
        await self._ensure_indexes()
        
        self._initialized = True
        logger.info("✅ Chat MongoDB Client initialized")

    async def _get_db(self):
        """Resolve the context-aware MongoDB database."""
        if self._db_manager is None:
            return None
        return await self._db_manager.get_context_db()

    
    async def _ensure_indexes(self):
        """Create necessary indexes for efficient querying"""
        db = await self._get_db()
        if db is None:
            return
        
        try:
            
            # Conversations collection indexes
            conv_collection = db[self.CONVERSATIONS_COLLECTION]
            await conv_collection.create_index([("session_id", 1)])
            await conv_collection.create_index([("session_id", 1), ("timestamp", 1)])
            await conv_collection.create_index([("user_id", 1)])
            
            # Documents collection indexes
            doc_collection = db[self.DOCUMENTS_COLLECTION]
            await doc_collection.create_index([("session_id", 1)])
            await doc_collection.create_index([("session_id", 1), ("document_id", 1)])
            
            logger.debug("✅ Chat MongoDB indexes ensured")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def _extract_user_id_from_session(self, session_id: str) -> Optional[str]:
        """
        Extract user_id from session_id format: user_{userId}_{timestamp}_{random}
        """
        try:
            if session_id.startswith("user_"):
                parts = session_id.split("_")
                if len(parts) >= 2:
                    return parts[1]
        except Exception as e:
            logger.warning(f"Failed to extract user_id from session_id: {session_id}, error: {e}")
        return None
    
    # ==================== CONVERSATION HISTORY METHODS ====================
    
    async def save_conversation_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a single conversation message to MongoDB with optional embedding
        
        Args:
            session_id: Session identifier
            message_id: Unique message ID
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            logger.warning("MongoDB not available for saving conversation")
            return False
            
        try:
            user_id = self._extract_user_id_from_session(session_id)
            timestamp = metadata.get("timestamp", datetime.now().isoformat()) if metadata else datetime.now().isoformat()
            
            # Compute embedding for semantic search
            embedding = None
            if self._embeddings_model is not None:
                embedding = compute_embedding(content, self._embeddings_model)
            
            document = {
                "_id": f"{session_id}_{message_id}",
                "session_id": session_id,
                "user_id": user_id,
                "message_id": message_id,
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "embedding": embedding,
                **(metadata or {})
            }
            
            collection = db[self.CONVERSATIONS_COLLECTION]
            
            # Use upsert to handle duplicates
            await collection.update_one(
                {"_id": document["_id"]},
                {"$set": document},
                upsert=True
            )
            
            logger.debug(f"💾 Saved message to MongoDB: {document['_id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save conversation message: {str(e)}")
            return False
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session, sorted by timestamp
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages
            
        Returns:
            List of messages with content and metadata
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            logger.warning("MongoDB not available for getting conversation history")
            return []
            
        try:
            collection = db[self.CONVERSATIONS_COLLECTION]
            
            cursor = collection.find(
                {"session_id": session_id},
                {"embedding": 0}  # Exclude embedding from results
            ).sort("timestamp", 1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            messages = []
            async for doc in cursor:
                messages.append({
                    "id": doc.get("_id"),
                    "role": doc.get("role"),
                    "content": doc.get("content"),
                    "timestamp": doc.get("timestamp"),
                    "metadata": {
                        k: v for k, v in doc.items() 
                        if k not in ["_id", "role", "content", "timestamp", "embedding"]
                    }
                })
            
            # If limit specified, return last N messages (most recent)
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            logger.debug(f"📥 Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {str(e)}")
            return []
    
    async def search_conversation_context(
        self,
        session_id: str,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Semantic search within a conversation for relevant context
        
        Args:
            session_id: Session identifier
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant messages sorted by relevance
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            return []
        
        if self._embeddings_model is None:
            # Fallback to simple text matching if embeddings not available
            return await self._simple_text_search(session_id, query, n_results)
            
        try:
            # Compute query embedding
            query_embedding = compute_embedding(query, self._embeddings_model)
            if query_embedding is None:
                return await self._simple_text_search(session_id, query, n_results)
            
            collection = db[self.CONVERSATIONS_COLLECTION]
            
            # Get all messages with embeddings for this session
            cursor = collection.find(
                {"session_id": session_id, "embedding": {"$ne": None}}
            )
            
            # Calculate similarities
            scored_messages = []
            async for doc in cursor:
                if doc.get("embedding"):
                    similarity = cosine_similarity(query_embedding, doc["embedding"])
                    scored_messages.append({
                        "content": doc.get("content"),
                        "metadata": {
                            k: v for k, v in doc.items()
                            if k not in ["content", "embedding", "_id"]
                        },
                        "relevance_score": similarity
                    })
            
            # Sort by relevance and take top N
            scored_messages.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = scored_messages[:n_results]
            
            logger.debug(f"🔍 Found {len(results)} relevant messages for query")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search conversation context: {str(e)}")
            return []
    
    async def _simple_text_search(
        self,
        session_id: str,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Simple text-based search fallback when embeddings are not available"""
        try:
            collection = db[self.CONVERSATIONS_COLLECTION]
            
            # Use MongoDB text search or simple regex
            cursor = collection.find({
                "session_id": session_id,
                "content": {"$regex": query, "$options": "i"}
            }).limit(n_results)
            
            results = []
            async for doc in cursor:
                results.append({
                    "content": doc.get("content"),
                    "metadata": {
                        k: v for k, v in doc.items()
                        if k not in ["content", "embedding", "_id"]
                    },
                    "relevance_score": 0.5  # Default score for text match
                })
            
            return results
        except Exception as e:
            logger.error(f"Simple text search failed: {e}")
            return []
    
    async def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all unique sessions that have messages in MongoDB
        
        Returns:
            List of session info dicts
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            return []
            
        try:
            collection = db[self.CONVERSATIONS_COLLECTION]
            
            # Aggregate to get session statistics
            pipeline = [
                {
                    "$group": {
                        "_id": "$session_id",
                        "message_count": {"$sum": 1},
                        "created_at": {"$min": "$timestamp"},
                        "last_updated": {"$max": "$timestamp"}
                    }
                },
                {"$sort": {"last_updated": -1}}
            ]
            
            sessions = []
            async for doc in collection.aggregate(pipeline):
                sessions.append({
                    "session_id": doc["_id"],
                    "message_count": doc["message_count"],
                    "created_at": doc["created_at"],
                    "last_updated": doc["last_updated"]
                })
            
            logger.debug(f"📊 Found {len(sessions)} unique sessions in MongoDB")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Failed to get all sessions: {str(e)}")
            return []
    
    async def clear_conversation(self, session_id: str) -> bool:
        """
        Clear all messages for a conversation session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            return False
            
        try:
            collection = db[self.CONVERSATIONS_COLLECTION]
            result = await collection.delete_many({"session_id": session_id})
            
            logger.info(f"🗑️ Cleared {result.deleted_count} messages for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear conversation: {str(e)}")
            return False
    
    # ==================== DOCUMENT RAG METHODS ====================
    
    async def save_document_chunks(
        self,
        session_id: str,
        document_id: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Save document chunks for RAG with embeddings
        
        Args:
            session_id: Session identifier
            document_id: Unique document identifier
            chunks: List of text chunks
            metadatas: List of metadata for each chunk
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            logger.warning("MongoDB not available for saving document chunks")
            return False
            
        try:
            user_id = self._extract_user_id_from_session(session_id)
            collection = db[self.DOCUMENTS_COLLECTION]
            
            documents = []
            for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
                # Compute embedding
                embedding = None
                if self._embeddings_model is not None:
                    embedding = compute_embedding(chunk, self._embeddings_model)
                
                doc = {
                    "_id": f"{session_id}_{document_id}_chunk_{i}",
                    "session_id": session_id,
                    "user_id": user_id,
                    "document_id": document_id,
                    "chunk_index": i,
                    "content": chunk,
                    "embedding": embedding,
                    "created_at": datetime.now().isoformat(),
                    **metadata
                }
                documents.append(doc)
            
            # Bulk upsert
            from pymongo import UpdateOne
            operations = [
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": doc},
                    upsert=True
                )
                for doc in documents
            ]
            
            if operations:
                await collection.bulk_write(operations)
            
            logger.info(f"📄 Saved {len(chunks)} document chunks for {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save document chunks: {str(e)}")
            return False
    
    async def search_documents(
        self,
        session_id: str,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search uploaded documents for relevant content (RAG)
        
        Args:
            session_id: Session identifier
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            return []
        
        if self._embeddings_model is None:
            # Fallback to text search
            return await self._simple_document_search(session_id, query, n_results)
            
        try:
            query_embedding = compute_embedding(query, self._embeddings_model)
            if query_embedding is None:
                return await self._simple_document_search(session_id, query, n_results)
            
            collection = db[self.DOCUMENTS_COLLECTION]
            
            # Get all chunks with embeddings for this session
            cursor = collection.find(
                {"session_id": session_id, "embedding": {"$ne": None}}
            )
            
            # Calculate similarities
            scored_chunks = []
            async for doc in cursor:
                if doc.get("embedding"):
                    similarity = cosine_similarity(query_embedding, doc["embedding"])
                    scored_chunks.append({
                        "content": doc.get("content"),
                        "metadata": {
                            k: v for k, v in doc.items()
                            if k not in ["content", "embedding", "_id"]
                        },
                        "relevance_score": similarity
                    })
            
            # Sort by relevance and take top N
            scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = scored_chunks[:n_results]
            
            logger.debug(f"🔍 Found {len(results)} relevant document chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search documents: {str(e)}")
            return []
    
    async def _simple_document_search(
        self,
        session_id: str,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Simple text-based document search fallback"""
        try:
            db = await self._get_db()
            if db is None:
                return []
            collection = db[self.DOCUMENTS_COLLECTION]
            
            cursor = collection.find({
                "session_id": session_id,
                "content": {"$regex": query, "$options": "i"}
            }).limit(n_results)
            
            results = []
            async for doc in cursor:
                results.append({
                    "content": doc.get("content"),
                    "metadata": {
                        k: v for k, v in doc.items()
                        if k not in ["content", "embedding", "_id"]
                    },
                    "relevance_score": 0.5
                })
            
            return results
        except Exception as e:
            logger.error(f"Simple document search failed: {e}")
            return []
    
    async def delete_document(self, session_id: str, document_id: str) -> bool:
        """
        Delete a document and all its chunks
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            return False
            
        try:
            collection = db[self.DOCUMENTS_COLLECTION]
            result = await collection.delete_many({
                "session_id": session_id,
                "document_id": document_id
            })
            
            logger.info(f"🗑️ Deleted document {document_id} ({result.deleted_count} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {str(e)}")
            return False
    
    # ==================== UTILITY METHODS ====================
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session statistics
        """
        await self._ensure_initialized()
        
        db = await self._get_db()
        if db is None:
            return {}
            
        try:
            conv_collection = db[self.CONVERSATIONS_COLLECTION]
            doc_collection = db[self.DOCUMENTS_COLLECTION]
            
            message_count = await conv_collection.count_documents({"session_id": session_id})
            chunk_count = await doc_collection.count_documents({"session_id": session_id})
            
            return {
                "session_id": session_id,
                "total_messages": message_count,
                "total_document_chunks": chunk_count,
                "has_documents": chunk_count > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get session stats: {str(e)}")
            return {}
    
    async def clear_session_data(self, session_id: str) -> bool:
        """
        Clear all data (conversations + documents) for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        await self._ensure_initialized()
        
        try:
            # Clear conversations
            conv_success = await self.clear_conversation(session_id)
            
            # Clear documents
            db = await self._get_db()
            if db:
                doc_collection = db[self.DOCUMENTS_COLLECTION]
                await doc_collection.delete_many({"session_id": session_id})
            
            logger.info(f"🧹 Cleared all data for session {session_id}")
            return conv_success
            
        except Exception as e:
            logger.error(f"❌ Failed to clear session data: {str(e)}")
            return False


# Global instance
_chat_mongodb_client = None
_client_lock = asyncio.Lock()


async def get_chat_mongodb_client(db_manager=None) -> ChatMongoDBClient:
    """Get or create Chat MongoDB client instance (singleton)"""
    global _chat_mongodb_client
    
    async with _client_lock:
        if _chat_mongodb_client is None:
            _chat_mongodb_client = ChatMongoDBClient(db_manager)
            await _chat_mongodb_client._ensure_initialized()
        return _chat_mongodb_client
