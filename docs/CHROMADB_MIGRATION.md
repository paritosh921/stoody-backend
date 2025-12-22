# ChromaDB to MongoDB Migration - Conversation Memory

## Summary

This document describes the migration of the **conversation memory system** from ChromaDB to MongoDB.

**IMPORTANT**: This migration only affects the **Debugger/Chat functionality**. ChromaDB is still used for the **Question Search** system in `question_service.py` and `core/database.py`.

## What Was Changed

### Files Created

1. **`models/chat_mongodb_client.py`**
   - New MongoDB-based chat memory client
   - Uses sentence-transformers for embeddings (same model as ChromaDB default)
   - Stores conversation messages and document chunks in MongoDB collections:
     - `chat_conversations`: Stores chat messages with embeddings
     - `chat_documents`: Stores document chunks for RAG
   - Provides semantic search using cosine similarity on stored embeddings
   - Fully async implementation compatible with the existing async architecture

### Files Modified

1. **`services/langchain_debugger_service.py`**
   - Changed from synchronous ChromaDB calls to async MongoDB calls
   - Added `_ensure_mongo_client()` method for lazy initialization
   - All `self.chroma_client.*` calls replaced with `await self.mongo_client.*`

### Files Removed

1. **`models/chat_chromadb_client.py`** - Obsolete chat ChromaDB client
2. **`services/langchain_debugger_service_v2.py`** - Temporary V2 file
3. **`scripts/database/check_chromadb_questions.py`** - Obsolete script
4. **`scripts/database/reset_chromadb.py`** - Obsolete script
5. **`scripts/database/sync_chromadb.py`** - Obsolete script
6. **`scripts/database/sync_questions_to_chromadb.py`** - Obsolete script

### Frontend References (Informational Only)
- `skiller-bot/src/services/backendService.ts` - Has ChromaDB status check methods
- `skiller-bot/src/components/admin/DocumentManagement.tsx` - ChromaDB integration UI

## Migration Path to Fully Remove ChromaDB

### Phase 1: Conversation Memory ✅ DONE
- MongoDB-based conversation storage is now in place
- The debugger service now uses MongoDB instead of ChromaDB

### Phase 2: Questions Storage (Future)
To remove ChromaDB from questions:
1. Update `core/database.py` to remove ChromaDB initialization
2. Update `services/question_service.py` to use MongoDB with text search
3. Update or remove the sync scripts
4. Update frontend status checks

### Phase 3: Cleanup (Future)
1. Remove `models/chat_chromadb_client.py`
2. Remove `chromadb` and `langchain-chroma` from `requirements.txt`
3. Delete the `chromadb_data/` and `data/chromadb_chat/` directories
4. Update any remaining documentation

## Benefits of MongoDB-Based Approach

1. **Single Database Dependency**: Only MongoDB needed (already in use)
2. **Production Ready**: Works well with MongoDB Atlas in cloud deployments
3. **Cost Savings**: No need to manage a separate vector database
4. **Simpler Deployment**: One less service to deploy and maintain
5. **Semantic Search Still Works**: Uses sentence-transformers for embeddings

## Performance Considerations

- Embeddings are computed on message save (slight overhead)
- Semantic search does in-memory cosine similarity (suitable for conversation-sized data)
- For very large document sets, consider MongoDB Atlas Vector Search in the future

## Testing Notes

The service maintains the same API interface, so existing frontend code should work without changes. Test the debugger chat functionality to ensure messages are being stored and retrieved correctly.
