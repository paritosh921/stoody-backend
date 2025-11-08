"""
Sync ChromaDB with MongoDB - remove orphaned ChromaDB entries and add missing questions
"""
import asyncio
import logging
from core.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def sync_chromadb():
    db = DatabaseManager()
    await db.initialize()

    # Get all questions from MongoDB
    mongo_questions = await db.mongo_find('questions', {})
    logger.info(f"MongoDB has {len(mongo_questions)} questions")

    # Get all questions from ChromaDB
    chroma_results = await db.chroma_get(where={}, limit=10000)
    chroma_ids = set(chroma_results.get('ids', []))
    logger.info(f"ChromaDB has {len(chroma_ids)} questions")

    # Find orphaned ChromaDB entries (in ChromaDB but not in MongoDB)
    mongo_ids = set(q.get('id') for q in mongo_questions if q.get('id'))
    orphaned = chroma_ids - mongo_ids
    logger.info(f"Found {len(orphaned)} orphaned ChromaDB entries")

    if orphaned:
        logger.info(f"Deleting {len(orphaned)} orphaned entries from ChromaDB...")
        for orphan_id in orphaned:
            await db.chroma_delete(ids=[orphan_id])
        logger.info("✅ Deleted orphaned entries")

    # Find missing ChromaDB entries (in MongoDB but not in ChromaDB)
    missing = mongo_ids - chroma_ids
    logger.info(f"Found {len(missing)} questions in MongoDB not in ChromaDB")

    if missing:
        logger.info(f"Adding {len(missing)} questions to ChromaDB...")
        added = 0
        for q in mongo_questions:
            if q.get('id') in missing and q.get('enhanced_options'):
                # Only add if it has valid options
                try:
                    await db.chroma_add(
                        ids=[q['id']],
                        documents=[q.get('text', '')],
                        metadatas=[{
                            'subject': q.get('subject', 'General'),
                            'difficulty': q.get('difficulty', 'medium'),
                            'document_type': q.get('metadata', {}).get('document_type', 'Unknown'),
                            'pdf_source': q.get('pdf_source', '')
                        }]
                    )
                    added += 1
                    logger.info(f"  Added {q['id']}")
                except Exception as e:
                    logger.error(f"  Failed to add {q['id']}: {e}")

        logger.info(f"✅ Added {added} questions to ChromaDB")

    logger.info("\n✅ Sync complete!")

if __name__ == "__main__":
    asyncio.run(sync_chromadb())
