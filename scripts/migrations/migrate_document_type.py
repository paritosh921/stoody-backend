"""
Migration Script: Add document_type to ChromaDB metadata for all existing questions

This script fixes the issue where questions were stored without document_type
in ChromaDB metadata, making them unsearchable by document type.

The script:
1. Fetches all questions from MongoDB (source of truth for document_type)
2. Updates ChromaDB metadata to include document_type field
3. Logs progress and results

Run this after updating the Question model to include document_type.
"""

import asyncio
import logging
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import DatabaseManager
from config_async import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def migrate_document_types():
    """Migrate all questions to include document_type in ChromaDB metadata"""

    db = DatabaseManager()
    await db.initialize()

    try:
        logger.info("=" * 80)
        logger.info("Starting document_type migration for ChromaDB")
        logger.info("=" * 80)

        # Get all questions from MongoDB (source of truth)
        logger.info("📥 Fetching all questions from MongoDB...")
        all_questions = await db.mongo_find("questions", {})
        total_questions = len(all_questions)

        logger.info(f"✅ Found {total_questions} questions in MongoDB")

        if total_questions == 0:
            logger.warning("⚠️ No questions found in MongoDB. Nothing to migrate.")
            return

        # Track statistics
        updated = 0
        skipped = 0
        errors = 0

        # Process each question
        logger.info("🔄 Processing questions...")
        for idx, question in enumerate(all_questions, 1):
            question_id = question.get("id")
            if not question_id:
                logger.warning(f"⚠️ Question #{idx} has no ID, skipping")
                skipped += 1
                continue

            try:
                # Get document_type from MongoDB question
                document_type = question.get("document_type")

                # If no document_type, try to get from metadata
                if not document_type and question.get("metadata"):
                    document_type = question.get("metadata", {}).get("document_type")

                # If still no document_type, try to infer from document_id
                if not document_type:
                    document_id = question.get("document_id")
                    if document_id:
                        # Look up the document to get its type
                        doc = await db.mongo_find_one("documents", {"document_id": document_id})
                        if doc:
                            document_type = doc.get("document_type")

                # Default to "Chapter Notes" if still no type found
                if not document_type:
                    document_type = "Chapter Notes"
                    logger.warning(f"⚠️ Question {question_id} has no document_type, defaulting to 'Chapter Notes'")

                # Get existing ChromaDB entry
                chroma_result = await db.chroma_get(ids=[question_id])

                if not chroma_result.get("ids"):
                    logger.warning(f"⚠️ Question {question_id} not found in ChromaDB, skipping")
                    skipped += 1
                    continue

                # Get existing metadata
                existing_metadata = chroma_result["metadatas"][0] if chroma_result.get("metadatas") else {}
                existing_document = chroma_result["documents"][0] if chroma_result.get("documents") else question.get("text", "")

                # Check if already has document_type
                if existing_metadata.get("document_type"):
                    logger.debug(f"✓ Question {question_id} already has document_type={existing_metadata.get('document_type')}")
                    skipped += 1
                    continue

                # Update metadata with document_type
                updated_metadata = {**existing_metadata, "document_type": document_type}

                # Also update fullData if it exists
                if "fullData" in updated_metadata:
                    import json
                    try:
                        full_data = json.loads(updated_metadata["fullData"])
                        full_data["document_type"] = document_type
                        updated_metadata["fullData"] = json.dumps(full_data)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not update fullData for {question_id}: {e}")

                # Update in ChromaDB
                await db.chroma_update(
                    ids=[question_id],
                    documents=[existing_document],
                    metadatas=[updated_metadata]
                )

                updated += 1

                if idx % 50 == 0:
                    logger.info(f"📊 Progress: {idx}/{total_questions} questions processed ({updated} updated, {skipped} skipped, {errors} errors)")

            except Exception as e:
                logger.error(f"❌ Error processing question {question_id}: {str(e)}")
                errors += 1

        # Final statistics
        logger.info("=" * 80)
        logger.info("✅ Migration completed!")
        logger.info(f"📊 Results:")
        logger.info(f"   - Total questions: {total_questions}")
        logger.info(f"   - Updated: {updated}")
        logger.info(f"   - Skipped (already migrated): {skipped}")
        logger.info(f"   - Errors: {errors}")
        logger.info("=" * 80)

        # Verify document type distribution
        logger.info("📈 Verifying document type distribution in ChromaDB...")

        practice_sets = await db.chroma_get(where={"document_type": "Practice Sets"})
        test_series = await db.chroma_get(where={"document_type": "Test Series"})
        chapter_notes = await db.chroma_get(where={"document_type": "Chapter Notes"})

        logger.info(f"   - Practice Sets: {len(practice_sets.get('ids', []))} questions")
        logger.info(f"   - Test Series: {len(test_series.get('ids', []))} questions")
        logger.info(f"   - Chapter Notes: {len(chapter_notes.get('ids', []))} questions")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Migration failed: {str(e)}")
        raise
    finally:
        # Cleanup (if needed)
        pass


if __name__ == "__main__":
    logger.info("🚀 Starting document_type migration script...")
    asyncio.run(migrate_document_types())
    logger.info("✅ Migration script completed!")
