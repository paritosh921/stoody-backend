"""
Chat ChromaDB Client - Separate ChromaDB instance for chat conversations
This is independent from the questions ChromaDB to avoid conflicts
"""

import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatChromaDBClient:
    """
    Separate ChromaDB client for managing chat conversations and RAG documents
    This is completely independent from the questions ChromaDB
    """

    # Separate path and collection names for chat
    CHAT_CHROMA_PATH = Path("data/chromadb_chat")
    CONVERSATIONS_COLLECTION = "chat_conversations"
    DOCUMENTS_COLLECTION = "chat_documents"

    def __init__(self):
        """Initialize chat ChromaDB client"""
        self.client = None
        self.conversations_collection = None
        self.documents_collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client and collections for chat"""
        try:
            # Ensure chat ChromaDB directory exists
            self.CHAT_CHROMA_PATH.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(self.CHAT_CHROMA_PATH),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection for conversations (chat history)
            self.conversations_collection = self.client.get_or_create_collection(
                name=self.CONVERSATIONS_COLLECTION,
                metadata={"description": "Chat conversation history with semantic search"}
            )

            # Get or create collection for uploaded documents (RAG)
            self.documents_collection = self.client.get_or_create_collection(
                name=self.DOCUMENTS_COLLECTION,
                metadata={"description": "User uploaded documents for RAG"}
            )

            logger.info(f"✅ Chat ChromaDB initialized successfully at {self.CHAT_CHROMA_PATH}")
            logger.info(f"   - Conversations collection: {self.CONVERSATIONS_COLLECTION}")
            logger.info(f"   - Documents collection: {self.DOCUMENTS_COLLECTION}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Chat ChromaDB: {str(e)}")
            raise

    # ==================== CONVERSATION HISTORY METHODS ====================

    def _extract_user_id_from_session(self, session_id: str) -> Optional[str]:
        """
        Extract user_id from session_id format: user_{userId}_{timestamp}_{random}

        Args:
            session_id: Session identifier

        Returns:
            user_id if valid format, None otherwise
        """
        try:
            if session_id.startswith("user_"):
                parts = session_id.split("_")
                if len(parts) >= 2:
                    return parts[1]  # user_id is the second part
        except Exception as e:
            logger.warning(f"Failed to extract user_id from session_id: {session_id}, error: {e}")
        return None

    def save_conversation_message(
        self,
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save a single conversation message to ChromaDB with user isolation

        Args:
            session_id: Session identifier (format: user_{userId}_{timestamp}_{random})
            message_id: Unique message ID
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (timestamp, attachments, etc.)

        Returns:
            Success status
        """
        try:
            # Extract and store user_id for additional security layer
            user_id = self._extract_user_id_from_session(session_id)

            full_metadata = {
                "session_id": session_id,
                "user_id": user_id,  # Store user_id for additional filtering
                "role": role,
                "timestamp": metadata.get("timestamp", datetime.now().isoformat()) if metadata else datetime.now().isoformat(),
                **(metadata or {})
            }

            # Use session_id + message_id as unique identifier
            doc_id = f"{session_id}_{message_id}"

            self.conversations_collection.add(
                documents=[content],
                metadatas=[full_metadata],
                ids=[doc_id]
            )

            logger.debug(f"💾 Saved message to ChromaDB: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save conversation message: {str(e)}")
            return False

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier
            limit: Optional limit on number of messages

        Returns:
            List of messages with content and metadata
        """
        try:
            # Query by session_id
            results = self.conversations_collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )

            if not results['ids']:
                return []

            # Combine results into message format
            messages = []
            for i in range(len(results['ids'])):
                messages.append({
                    "id": results['ids'][i],
                    "role": results['metadatas'][i]['role'],
                    "content": results['documents'][i],
                    "timestamp": results['metadatas'][i].get('timestamp'),
                    "metadata": results['metadatas'][i]
                })

            # Sort by timestamp
            messages.sort(key=lambda x: x['timestamp'])

            # Apply limit if specified
            if limit:
                messages = messages[-limit:]

            logger.debug(f"📥 Retrieved {len(messages)} messages for session {session_id}")
            return messages

        except Exception as e:
            logger.error(f"❌ Failed to get conversation history: {str(e)}")
            return []

    def search_conversation_context(
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
            List of relevant messages
        """
        try:
            results = self.conversations_collection.query(
                query_texts=[query],
                where={"session_id": session_id},
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            if not results['ids'] or not results['ids'][0]:
                return []

            # Format results
            relevant_messages = []
            for i in range(len(results['ids'][0])):
                relevant_messages.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "relevance_score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })

            logger.debug(f"🔍 Found {len(relevant_messages)} relevant messages for query")
            return relevant_messages

        except Exception as e:
            logger.error(f"❌ Failed to search conversation context: {str(e)}")
            return []

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all unique sessions that have messages in ChromaDB

        Returns:
            List of session info dicts with session_id, message_count, created_at, last_updated
        """
        try:
            # Get ALL conversation messages
            all_results = self.conversations_collection.get(
                include=["metadatas"]
            )

            if not all_results['ids']:
                return []

            # Group by session_id
            sessions_map = {}
            for i, metadata in enumerate(all_results['metadatas']):
                session_id = metadata.get('session_id')
                if not session_id:
                    continue

                timestamp = metadata.get('timestamp')
                
                if session_id not in sessions_map:
                    sessions_map[session_id] = {
                        'session_id': session_id,
                        'message_count': 0,
                        'created_at': timestamp,
                        'last_updated': timestamp
                    }
                
                sessions_map[session_id]['message_count'] += 1
                
                # Update last_updated if this message is newer
                if timestamp and timestamp > sessions_map[session_id]['last_updated']:
                    sessions_map[session_id]['last_updated'] = timestamp

            sessions_list = list(sessions_map.values())
            logger.debug(f"📊 Found {len(sessions_list)} unique sessions in ChromaDB")
            return sessions_list

        except Exception as e:
            logger.error(f"❌ Failed to get all sessions: {str(e)}")
            return []

    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear all messages for a conversation session

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            # Get all message IDs for this session
            results = self.conversations_collection.get(
                where={"session_id": session_id},
                include=["documents"]
            )

            if results['ids']:
                self.conversations_collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Cleared {len(results['ids'])} messages for session {session_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to clear conversation: {str(e)}")
            return False

    # ==================== DOCUMENT RAG METHODS ====================

    def save_document_chunks(
        self,
        session_id: str,
        document_id: str,
        chunks: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """
        Save document chunks for RAG with user isolation

        Args:
            session_id: Session identifier (format: user_{userId}_{timestamp}_{random})
            document_id: Unique document identifier
            chunks: List of text chunks
            metadatas: List of metadata for each chunk

        Returns:
            Success status
        """
        try:
            # Extract user_id for additional security
            user_id = self._extract_user_id_from_session(session_id)

            # Prepare IDs for chunks
            chunk_ids = [f"{session_id}_{document_id}_chunk_{i}" for i in range(len(chunks))]

            # Add session_id and user_id to all metadatas for double-layer security
            for metadata in metadatas:
                metadata['session_id'] = session_id
                metadata['user_id'] = user_id  # Additional security layer
                metadata['document_id'] = document_id

            self.documents_collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )

            logger.info(f"📄 Saved {len(chunks)} document chunks for {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save document chunks: {str(e)}")
            return False

    def search_documents(
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
        try:
            results = self.documents_collection.query(
                query_texts=[query],
                where={"session_id": session_id},
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            if not results['ids'] or not results['ids'][0]:
                return []

            # Format results
            relevant_chunks = []
            for i in range(len(results['ids'][0])):
                relevant_chunks.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "relevance_score": 1 - results['distances'][0][i]
                })

            logger.debug(f"🔍 Found {len(relevant_chunks)} relevant document chunks")
            return relevant_chunks

        except Exception as e:
            logger.error(f"❌ Failed to search documents: {str(e)}")
            return []

    def delete_document(self, session_id: str, document_id: str) -> bool:
        """
        Delete a document and all its chunks

        Args:
            session_id: Session identifier
            document_id: Document identifier

        Returns:
            Success status
        """
        try:
            results = self.documents_collection.get(
                where={
                    "session_id": session_id,
                    "document_id": document_id
                }
            )

            if results['ids']:
                self.documents_collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted document {document_id} and its chunks")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete document: {str(e)}")
            return False

    # ==================== UTILITY METHODS ====================

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        try:
            # Count conversation messages
            conv_results = self.conversations_collection.get(
                where={"session_id": session_id}
            )

            # Count document chunks
            doc_results = self.documents_collection.get(
                where={"session_id": session_id}
            )

            return {
                "session_id": session_id,
                "total_messages": len(conv_results['ids']),
                "total_document_chunks": len(doc_results['ids']),
                "has_documents": len(doc_results['ids']) > 0
            }

        except Exception as e:
            logger.error(f"❌ Failed to get session stats: {str(e)}")
            return {}

    def clear_session_data(self, session_id: str) -> bool:
        """
        Clear all data (conversations + documents) for a session

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            conv_success = self.clear_conversation(session_id)

            # Clear documents
            doc_results = self.documents_collection.get(
                where={"session_id": session_id}
            )
            if doc_results['ids']:
                self.documents_collection.delete(ids=doc_results['ids'])

            logger.info(f"🧹 Cleared all data for session {session_id}")
            return conv_success

        except Exception as e:
            logger.error(f"❌ Failed to clear session data: {str(e)}")
            return False


# Global instance
_chat_chromadb_client = None


def get_chat_chromadb_client() -> ChatChromaDBClient:
    """Get or create Chat ChromaDB client instance (singleton)"""
    global _chat_chromadb_client
    if _chat_chromadb_client is None:
        _chat_chromadb_client = ChatChromaDBClient()
    return _chat_chromadb_client
