"""
Clear and rebuild ChromaDB from MongoDB
"""
import asyncio
import logging
from core.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_chromadb():
    db = DatabaseManager()
    await db.initialize()

    # Clear ChromaDB
    logger.info("Clearing ChromaDB collection...")
    try:
        db.chroma_client.delete_collection('questions')
        logger.info("✅ Deleted old collection")
    except:
        logger.info("Collection doesn't exist, creating new one")

    # Recreate collection
    db.chroma_collection = db.chroma_client.create_collection('questions')
    logger.info("✅ Created new collection")

    # Get all questions from MongoDB with enhanced_options
    mongo_questions = await db.mongo_find('questions', {})
    valid_questions = [q for q in mongo_questions if q.get('enhanced_options')]
    logger.info(f"Found {len(valid_questions)} valid questions in MongoDB to add")

    # Add to ChromaDB
    added = 0
    for q in valid_questions:
        try:
            metadata = {
                'subject': q.get('subject', 'General'),
                'difficulty': q.get('difficulty', 'medium'),
                'document_type': q.get('metadata', {}).get('document_type', 'Unknown'),
                'pdf_source': q.get('pdf_source', '')
            }

            await db.chroma_add(
                ids=[q['id']],
                documents=[q.get('text', '')],
                metadatas=[metadata]
            )
            added += 1
            if added % 5 == 0:
                logger.info(f"Added {added}/{len(valid_questions)}...")
        except Exception as e:
            logger.error(f"Failed to add {q['id']}: {e}")

    logger.info(f"✅ Added {added} questions to ChromaDB")

if __name__ == "__main__":
    asyncio.run(reset_chromadb())
