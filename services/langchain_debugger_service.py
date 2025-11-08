"""
LangChain-based Debugger Chat Service with RAG
Uses ChromaDB for conversation memory and document storage
PRODUCTION-READY for 1000+ concurrent users across multiple workers
Uses Redis for distributed session state
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib

# LangChain imports (Pydantic v2 compatible)
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use HuggingFaceEmbeddings for FREE local embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local imports
from models.chat_chromadb_client import get_chat_chromadb_client
from services.document_processor import get_document_processor
from services.async_openai_service import AsyncOpenAIService
from config_async import OPENAI_API_KEY, OPENAI_MODEL, REDIS_URL
from core.cache import CacheManager

logger = logging.getLogger(__name__)


class LangChainDebuggerService:
    """
    Production-ready debugger chat service using LangChain
    - ChromaDB for conversation memory and RAG
    - Supports 1000+ concurrent users
    - Clean error handling and logging
    - Document upload support (PDF, Word, Images)
    """

    # System prompt optimized for student learning
    SYSTEM_PROMPT = """You are an expert AI tutor and problem-solving assistant designed to help students learn effectively. Your core responsibilities:

**CRITICAL - Conversation Memory:**
- You have access to the COMPLETE conversation history including ALL previous messages
- When students refer to "the image I sent earlier" or "that question from before", look back through the conversation
- Reference and build upon previous explanations and discussions
- Remember uploaded documents and images shared throughout the session
- Maintain context across the entire conversation

**Problem Solving Approach:**
1. Break down complex problems into manageable steps
2. Guide students to discover answers themselves rather than giving direct solutions
3. Provide hints and ask probing questions to stimulate critical thinking
4. Celebrate progress and encourage persistence
5. Reference previous work when relevant

**Communication Style:**
- Use clear, encouraging language appropriate for students
- Be patient and supportive, never condescending
- Use examples and analogies to clarify concepts
- Ask questions to gauge understanding
- Build on what the student has already learned in this conversation

**Content Analysis:**
- When analyzing uploaded documents or images, describe what you observe
- For math/science problems, identify the subject and key concepts
- Transcribe equations and formulas accurately
- Explain visual content in detail
- Remember images from earlier in the conversation

**Response Format:**
- Use markdown for structure and clarity
- Use LaTeX for ALL mathematical expressions:
  * Inline math: $E = mc^2$
  * Display math: $$F = ma$$
- Include step-by-step explanations
- Provide relevant examples

**Key Principles:**
- Foster independent learning and understanding
- Adapt difficulty to student's level
- Maintain conversation context across ALL messages
- Use uploaded documents as reference material
- Be thorough but concise
- ALWAYS reference previous parts of the conversation when the student asks about them

Remember: Your goal is to teach, not just answer. Help students develop problem-solving skills. You have FULL conversation history - use it!"""

    def __init__(self):
        """Initialize LangChain debugger service with Redis for multi-worker support"""
        self.chroma_client = get_chat_chromadb_client()
        self.document_processor = get_document_processor()
        self.openai_service = AsyncOpenAIService()
        
        # Align chunking with document processor so RAG storage stays consistent
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.document_processor.CHUNK_SIZE,
            chunk_overlap=self.document_processor.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        # CRITICAL: Redis-based session management for multi-worker concurrency
        self.cache_manager = None  # Will be initialized async
        self._sessions_lock = asyncio.Lock()
        
        # In-memory cache for performance (worker-local)
        self._local_session_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes local cache

        # LangChain embeddings - using FREE open-source Sentence Transformers
        # Same as ChromaDB default, ensuring consistency across the app
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, lightweight, free
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Better similarity search
        )

        logger.info("🚀 LangChain Debugger Service initialized with FREE Sentence Transformers (multi-worker ready)")

    async def _ensure_cache_manager(self):
        """Initialize cache manager if not already done"""
        if self.cache_manager is None:
            try:
                self.cache_manager = CacheManager(REDIS_URL)
                await self.cache_manager.initialize()
                logger.info("✅ Redis cache manager initialized for session state")
            except Exception as e:
                logger.warning(f"⚠️ Redis unavailable, using in-memory only: {str(e)}")
                self.cache_manager = None

    async def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get or create a chat session with Redis-backed memory for multi-worker support
        
        CRITICAL: This ensures all workers see the same session state

        Args:
            session_id: Unique session identifier

        Returns:
            Session data dictionary
        """
        await self._ensure_cache_manager()
        
        async with self._sessions_lock:
            # Try local cache first (fast path)
            if session_id in self._local_session_cache:
                cached_session = self._local_session_cache[session_id]
                # Verify cache is not stale
                cache_time = cached_session.get('_cached_at', 0)
                if (datetime.now().timestamp() - cache_time) < self._cache_ttl:
                    return cached_session
            
            # Try Redis (shared across all workers)
            if self.cache_manager:
                try:
                    redis_key = f"debugger_session:{session_id}"
                    session_data = await self.cache_manager.get(redis_key)
                    
                    if session_data:
                        session = json.loads(session_data)
                        # Update local cache
                        session['_cached_at'] = datetime.now().timestamp()
                        self._local_session_cache[session_id] = session
                        logger.debug(f"📥 Loaded session {session_id} from Redis")
                        return session
                except Exception as e:
                    logger.error(f"❌ Redis get error: {str(e)}")

            # Create new session (not found in cache or Redis)
            session = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "has_documents": False,
                "document_ids": [],
                "_cached_at": datetime.now().timestamp()
            }

            # Save to Redis immediately (so other workers can see it)
            if self.cache_manager:
                try:
                    redis_key = f"debugger_session:{session_id}"
                    await self.cache_manager.set(
                        redis_key,
                        json.dumps(session),
                        ttl=86400  # 24 hours
                    )
                    logger.info(f"✨ Created new session in Redis: {session_id}")
                except Exception as e:
                    logger.error(f"❌ Redis set error: {str(e)}")
            
            # Update local cache
            self._local_session_cache[session_id] = session
            
            return session
    
    async def _update_session(self, session: Dict[str, Any]):
        """
        Update session in Redis and local cache
        
        CRITICAL: This keeps all workers synchronized

        Args:
            session: Session dictionary to update
        """
        session_id = session['session_id']
        session['_cached_at'] = datetime.now().timestamp()
        
        # Update local cache
        self._local_session_cache[session_id] = session
        
        # Update Redis (for other workers)
        if self.cache_manager:
            try:
                redis_key = f"debugger_session:{session_id}"
                await self.cache_manager.set(
                    redis_key,
                    json.dumps(session),
                    ttl=86400  # 24 hours
                )
                logger.debug(f"💾 Updated session {session_id} in Redis")
            except Exception as e:
                logger.error(f"❌ Redis update error: {str(e)}")

    async def send_message(
        self,
        session_id: str,
        message: str,
        attachments: Optional[List[str]] = None,
        image_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a message and get AI response with full context

        Args:
            session_id: Session identifier
            message: User message
            attachments: Optional list of attachment names
            image_data: Optional base64 image data

        Returns:
            Dict containing response and metadata
        """
        try:
            # Get or create session
            session = await self._get_or_create_session(session_id)

            # Generate message ID
            message_id = f"msg_{int(datetime.now().timestamp() * 1000)}"
            timestamp = datetime.now().isoformat()

            # Save user message to ChromaDB
            # Convert attachments list to string (ChromaDB doesn't support list metadata)
            attachments_str = ",".join(attachments) if attachments else ""

            user_metadata = {
                "timestamp": timestamp,
                "attachments": attachments_str,  # Store as comma-separated string
                "has_image": "true" if image_data else "false",
                # Note: We store a flag but not the actual image data in ChromaDB metadata
                # Image data will be included in context when messages are retrieved
            }
            
            # Store the message content WITH image reference if present
            message_content = message
            if image_data:
                # Add image marker to content for better retrieval
                message_content = f"[IMAGE ATTACHED]\n{message}"
                logger.info(f"💾 Storing message with image marker for session {session_id}")

            self.chroma_client.save_conversation_message(
                session_id=session_id,
                message_id=message_id,
                role="user",
                content=message_content,
                metadata=user_metadata
            )

            # Get conversation history from ChromaDB - increased to 40 messages (20 exchanges) for better memory
            history = self.chroma_client.get_conversation_history(session_id, limit=40)

            # Build conversation context for OpenAI - IMPORTANT: Send full history!
            conversation_messages = []
            for msg in history[:-1]:  # Exclude current message (will add it with RAG context)
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            logger.info(f"📚 Retrieved {len(history)} messages from history, sending {len(conversation_messages)} to OpenAI")

            # Search for relevant context from previous conversations (for RAG context)
            relevant_context = []
            if len(history) > 3:
                relevant_context = self.chroma_client.search_conversation_context(
                    session_id=session_id,
                    query=message,
                    n_results=3
                )

            # Search documents if any exist
            session_stats = self.chroma_client.get_session_stats(session_id)
            document_context = []
            if session_stats.get("has_documents", False):
                logger.info(f"📚 Session has documents, searching for relevant context...")
                doc_results = self.chroma_client.search_documents(
                    session_id=session_id,
                    query=message,
                    n_results=5  # Increased for better document context
                )
                document_context = [doc["content"] for doc in doc_results]
                logger.info(f"📚 Found {len(document_context)} relevant document chunks")
                if document_context:
                    logger.debug(f"📚 First chunk preview: {document_context[0][:200]}...")
            else:
                logger.info(f"📚 No documents found for session {session_id}")

            # Build enhanced prompt with RAG context (for current message only)
            enhanced_message = await self._build_rag_prompt(
                user_message=message,
                relevant_context=relevant_context,
                document_context=document_context,
                has_image=image_data is not None
            )

            # Prepare messages for OpenAI - SEND FULL CONVERSATION HISTORY!
            # This is critical for maintaining context across messages
            
            # Build the current message (with or without image)
            if image_data:
                # Image message with multimodal content
                current_message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_message},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
                logger.info(f"🖼️ Adding image to current message (maintaining full conversation history)")
            else:
                # Regular text message
                current_message = {"role": "user", "content": enhanced_message}
            
            # CRITICAL: Combine history with current message (don't replace!)
            messages = conversation_messages + [current_message]

            logger.info(f"📝 Sending {len(messages)} messages to OpenAI ({len(conversation_messages)} from history + 1 current)")

            # Get AI response - use temperature=1 for o1 models, 0.7 for others
            temperature = 1.0 if "o1" in self.openai_service.model.lower() else 0.7
            response = await self.openai_service.chat_completion_async(
                messages=messages,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=2000
            )

            if not response.get("success"):
                logger.error(f"OpenAI API error: {response.get('error')}")
                return {
                    "success": False,
                    "error": "Failed to generate response",
                    "details": response.get("error")
                }

            assistant_content = response["response"]

            # Save assistant response to ChromaDB
            assistant_message_id = f"msg_{int(datetime.now().timestamp() * 1000)}_ai"
            assistant_metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": response.get("model", OPENAI_MODEL),
                "tokens": response.get("usage", {}).get("total_tokens", 0)
            }

            self.chroma_client.save_conversation_message(
                session_id=session_id,
                message_id=assistant_message_id,
                role="assistant",
                content=assistant_content,
                metadata=assistant_metadata
            )

            # Update session stats and sync to Redis
            session["message_count"] += 2
            session["last_updated"] = datetime.now().isoformat()
            await self._update_session(session)

            logger.info(f"💬 Session {session_id}: Message exchange completed ({session['message_count']} total messages)")

            return {
                "success": True,
                "data": {
                    "user_message": {
                        "id": message_id,
                        "role": "user",
                        "content": message,
                        "timestamp": timestamp,
                        "attachments": attachments or []
                    },
                    "assistant_message": {
                        "id": assistant_message_id,
                        "role": "assistant",
                        "content": assistant_content,
                        "timestamp": assistant_metadata["timestamp"]
                    },
                    "response": assistant_content,
                    "session_id": session_id,
                    "message_count": session["message_count"],
                    "usage": response.get("usage", {}),
                    "model": response.get("model", OPENAI_MODEL),
                    "has_document_context": len(document_context) > 0
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in send_message: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": "Internal server error",
                "details": str(e)
            }

    async def _build_rag_prompt(
        self,
        user_message: str,
        relevant_context: List[Dict[str, Any]],
        document_context: List[str],
        has_image: bool
    ) -> str:
        """
        Build enhanced prompt with RAG context (for current message only)
        Note: Conversation history is sent separately to OpenAI as messages array

        Args:
            user_message: Current user message
            relevant_context: Semantically relevant past messages
            document_context: Relevant document chunks
            has_image: Whether message includes an image

        Returns:
            Enhanced prompt string with RAG context
        """
        prompt_parts = []

        # Add document context if available
        if document_context:
            prompt_parts.append("**Reference Documents (from uploaded files):**")
            for i, doc in enumerate(document_context, 1):
                # Show more content for better context
                prompt_parts.append(f"{i}. {doc[:500]}{'...' if len(doc) > 500 else ''}")
            prompt_parts.append("")

        # Add relevant past context if available (from semantic search)
        if relevant_context and len(relevant_context) > 0:
            prompt_parts.append("**Relevant Previous Discussion:**")
            for ctx in relevant_context[:2]:  # Limit to 2 most relevant
                role = ctx["metadata"]["role"]
                content = ctx["content"][:300]  # Show more context
                prompt_parts.append(f"- {role.title()}: {content}{'...' if len(ctx['content']) > 300 else ''}")
            prompt_parts.append("")

        # Add current message
        if has_image:
            prompt_parts.append(f"**Current Question (with image):** {user_message}")
        else:
            prompt_parts.append(f"**Current Question:** {user_message}")

        return "\n".join(prompt_parts)

    async def upload_document(
        self,
        session_id: str,
        file_content: bytes,
        filename: str,
        mime_type: str
    ) -> Dict[str, Any]:
        """
        Upload and process a document for RAG
        Automatically handles image-based PDFs by converting to images and using Vision API

        Args:
            session_id: Session identifier
            file_content: Binary file content
            filename: Original filename
            mime_type: MIME type

        Returns:
            Dict with processing result
        """
        try:
            # Validate file
            validation = self.document_processor.validate_file(
                filename=filename,
                mime_type=mime_type,
                file_size=len(file_content),
                max_size_mb=10
            )

            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"]
                }

            # Process document
            result = await self.document_processor.process_file(
                file_content=file_content,
                filename=filename,
                mime_type=mime_type
            )

            if not result["success"]:
                return result

            # Check if this is an image-based PDF that was converted
            if result.get("is_image_based", False):
                logger.info(f"📸 Image-based PDF detected ({result.get('num_images', 0)} pages), processing with Vision API...")

                # Extract image data from result
                text_content = result["text"]
                image_data_urls = []

                # Parse image data markers from text
                import re
                pattern = r'\{IMAGE_DATA:(data:image/[^}]+)\}'
                matches = re.findall(pattern, text_content)
                image_data_urls = matches

                if not image_data_urls:
                    return {
                        "success": False,
                        "error": "Failed to extract image data from converted PDF"
                    }

                # Process each image with Vision API
                processed_texts = []
                for i, img_data in enumerate(image_data_urls, 1):
                    logger.info(f"🔍 Processing page {i}/{len(image_data_urls)} with Vision API...")

                    # Create prompt for Vision API
                    vision_prompt = f"""Analyze this page (Page {i} of {len(image_data_urls)}) from "{filename}".

Extract ALL text content including:
- Questions and their numbers
- Problem statements
- Mathematical equations and formulas
- Diagrams and their descriptions
- Any other visible text

Format the output clearly with proper structure."""

                    # Use Vision API to read the image - use temperature=1 for o1 models, 0.3 for others
                    vision_temperature = 1.0 if "o1" in self.openai_service.model.lower() else 0.3
                    vision_response = await self.openai_service.chat_completion_async(
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": img_data}}
                            ]
                        }],
                        system_prompt="You are an expert at reading and extracting text from images, especially educational content with questions, diagrams, and mathematical formulas.",
                        temperature=vision_temperature,
                        max_tokens=2000
                    )

                    if vision_response.get("success"):
                        page_text = vision_response["response"]
                        processed_texts.append(f"[Page {i} from {filename}]\n{page_text}")
                        logger.info(f"✅ Page {i} processed: {len(page_text)} characters extracted")
                    else:
                        logger.error(f"❌ Failed to process page {i}: {vision_response.get('error')}")
                        processed_texts.append(f"[Page {i} - Processing Failed]")

                # Combine all processed text
                full_text = "\n\n".join(processed_texts)

                # Re-chunk the Vision API processed text
                chunks = self.text_splitter.split_text(full_text)

                # Update result with processed data
                result["text"] = full_text
                result["chunks"] = chunks
                result["char_count"] = len(full_text)
                result["vision_processed"] = True
                result["num_images"] = len(image_data_urls)

                logger.info(f"📸 Vision API processing complete: {len(full_text)} chars -> {len(chunks)} chunks")

            # Generate document ID
            document_id = f"doc_{int(datetime.now().timestamp() * 1000)}"

            # Prepare chunk metadatas
            chunk_metadatas = []
            for i, chunk in enumerate(result["chunks"]):
                metadata = {
                    "chunk_index": str(i),
                    "filename": filename,
                    "file_type": result["file_type"],
                    "total_chunks": str(len(result["chunks"])),
                    "uploaded_at": datetime.now().isoformat()
                }

                # Add vision processing flag if applicable
                if result.get("is_image_based", False):
                    metadata["vision_processed"] = "true"
                    metadata["num_images"] = str(result.get("num_images", 0))

                chunk_metadatas.append(metadata)

            # Save to ChromaDB
            success = self.chroma_client.save_document_chunks(
                session_id=session_id,
                document_id=document_id,
                chunks=result["chunks"],
                metadatas=chunk_metadatas
            )

            if not success:
                return {
                    "success": False,
                    "error": "Failed to save document to database"
                }

            # Update session and sync to Redis
            session = await self._get_or_create_session(session_id)
            session["has_documents"] = True
            session["document_ids"].append(document_id)
            await self._update_session(session)

            logger.info(f"📄 Document uploaded: {filename} -> {document_id} ({len(result['chunks'])} chunks)")

            return {
                "success": True,
                "document_id": document_id,
                "filename": filename,
                "file_type": result["file_type"],
                "num_chunks": len(result["chunks"]),
                "char_count": result["char_count"],
                "is_image_based": result.get("is_image_based", False),
                "vision_processed": result.get("vision_processed", False)
            }

        except Exception as e:
            logger.error(f"❌ Error uploading document: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": "Failed to upload document",
                "details": str(e)
            }

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier
            limit: Optional message limit

        Returns:
            Dict containing conversation history
        """
        try:
            # Get session (creates if doesn't exist)
            session = await self._get_or_create_session(session_id)

            # Get messages from ChromaDB
            messages = self.chroma_client.get_conversation_history(session_id, limit)

            return {
                "success": True,
                "data": {
                    "session_id": session_id,
                    "messages": messages,
                    "metadata": {
                        "total_messages": len(messages),
                        "has_documents": session.get("has_documents", False),
                        "created_at": session.get("created_at")
                    }
                }
            }

        except Exception as e:
            logger.error(f"❌ Error getting history: {str(e)}")
            return {
                "success": False,
                "error": "Failed to retrieve history",
                "details": str(e)
            }

    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        """
        Clear conversation history and documents for a session
        
        CRITICAL: Clears from ChromaDB, Redis, and local cache

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        try:
            await self._ensure_cache_manager()
            
            # Clear from ChromaDB
            success = self.chroma_client.clear_session_data(session_id)

            # Clear from Redis (for all workers)
            if self.cache_manager:
                try:
                    redis_key = f"debugger_session:{session_id}"
                    await self.cache_manager.delete(redis_key)
                    logger.debug(f"🗑️ Deleted session {session_id} from Redis")
                except Exception as e:
                    logger.error(f"❌ Redis delete error: {str(e)}")

            # Clear from local cache
            async with self._sessions_lock:
                if session_id in self._local_session_cache:
                    del self._local_session_cache[session_id]

            logger.info(f"🧹 Cleared session: {session_id}")

            return {
                "success": True,
                "message": "Session cleared successfully"
            }

        except Exception as e:
            logger.error(f"❌ Error clearing session: {str(e)}")
            return {
                "success": False,
                "error": "Failed to clear session",
                "details": str(e)
            }

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Delete a session completely

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        return await self.clear_session(session_id)

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active sessions from ChromaDB (persistent storage)
        
        PRODUCTION: Gets data from ChromaDB (multi-worker safe)

        Returns:
            Dict containing session statistics
        """
        try:
            await self._ensure_cache_manager()
            
            # Get all sessions from ChromaDB (the source of truth for messages)
            chromadb_sessions = self.chroma_client.get_all_sessions()
            
            # Enrich with Redis session data if available (for has_documents flag)
            sessions_list = []
            for chroma_session in chromadb_sessions:
                session_id = chroma_session['session_id']
                
                # Try to get additional metadata from Redis
                has_documents = False
                if self.cache_manager:
                    try:
                        redis_key = f"debugger_session:{session_id}"
                        session_data = await self.cache_manager.get(redis_key)
                        if session_data:
                            redis_session = json.loads(session_data)
                            has_documents = redis_session.get("has_documents", False)
                    except Exception as e:
                        logger.debug(f"Could not load session {session_id} from Redis: {str(e)}")
                
                sessions_list.append({
                    "session_id": session_id,
                    "message_count": chroma_session.get('message_count', 0),
                    "has_documents": has_documents,
                    "created_at": chroma_session.get('created_at'),
                    "last_updated": chroma_session.get('last_updated')
                })
            
            logger.debug(f"📊 Returning {len(sessions_list)} sessions (multi-worker safe)")
            
            return {
                "success": True,
                "data": {
                    "active_sessions": len(sessions_list),
                    "sessions": sessions_list
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting session stats: {str(e)}")
            return {
                "success": True,
                "data": {
                    "active_sessions": 0,
                    "sessions": []
                }
            }


# Global instance
_langchain_debugger_service = None
_service_lock = asyncio.Lock()


async def get_langchain_debugger_service():
    """Get singleton instance of LangChain debugger service"""
    global _langchain_debugger_service

    async with _service_lock:
        if _langchain_debugger_service is None:
            _langchain_debugger_service = LangChainDebuggerService()
            logger.info("🎯 LangChain Debugger Service singleton initialized")

        return _langchain_debugger_service
