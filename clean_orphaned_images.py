#!/usr/bin/env python3
"""
Script to clean orphaned image references from questions
Run this to fix questions with broken image references
"""

import asyncio
import sys
from core.database import DatabaseManager
from utils.image_validator import (
    get_orphaned_images_in_document,
    clean_question_images,
    get_orphaned_images_in_question
)
from datetime import datetime


async def clean_document(document_id: str):
    """Clean orphaned images from a specific document"""
    db = DatabaseManager()
    await db.initialize()

    try:
        # Verify document exists
        document = await db.mongo_find_one("documents", {"document_id": document_id})
        if not document:
            print(f"❌ Document {document_id} not found")
            return

        print(f"\n📄 Document: {document.get('title', document_id)}")
        print(f"🔍 Scanning for orphaned images...\n")

        # Find all orphaned images
        orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

        if not orphaned_by_question:
            print("✅ No orphaned images found!")
            return

        print(f"⚠️  Found orphaned images in {len(orphaned_by_question)} questions:\n")

        # Clean each affected question
        questions_cleaned = 0
        total_images_removed = 0

        for question_id, orphaned_ids in orphaned_by_question.items():
            print(f"  Question: {question_id}")
            print(f"    Orphaned images: {', '.join(orphaned_ids)}")

            # Get question
            question = await db.mongo_find_one("questions", {"id": question_id})
            if not question:
                print(f"    ❌ Question not found in database")
                continue

            # Clean orphaned references
            cleaned_question, removed_count = await clean_question_images(question, db)

            if removed_count > 0:
                # Update question in database
                await db.mongo_update_one(
                    "questions",
                    {"id": question_id},
                    {"$set": {
                        "images": cleaned_question.get("images", []),
                        "question_figures": cleaned_question.get("question_figures", []),
                        "cleaned_at": datetime.utcnow(),
                        "cleaned_by": "cleanup_script"
                    }}
                )

                questions_cleaned += 1
                total_images_removed += removed_count
                print(f"    ✅ Removed {removed_count} orphaned references\n")

        print(f"\n🎉 Cleanup complete!")
        print(f"   Questions cleaned: {questions_cleaned}")
        print(f"   Total images removed: {total_images_removed}")

    finally:
        await db.close()


async def clean_question(question_id: str):
    """Clean orphaned images from a specific question"""
    db = DatabaseManager()
    await db.initialize()

    try:
        # Get question
        question = await db.mongo_find_one("questions", {"id": question_id})
        if not question:
            print(f"❌ Question {question_id} not found")
            return

        print(f"\n❓ Question: {question_id}")
        print(f"🔍 Scanning for orphaned images...\n")

        # Get orphaned images
        orphaned_ids = await get_orphaned_images_in_question(question_id, db)

        if not orphaned_ids:
            print("✅ No orphaned images found!")
            return

        print(f"⚠️  Found {len(orphaned_ids)} orphaned images:")
        for img_id in orphaned_ids:
            print(f"    - {img_id}")

        # Clean orphaned references
        cleaned_question, removed_count = await clean_question_images(question, db)

        if removed_count > 0:
            # Update question in database
            await db.mongo_update_one(
                "questions",
                {"id": question_id},
                {"$set": {
                    "images": cleaned_question.get("images", []),
                    "question_figures": cleaned_question.get("question_figures", []),
                    "cleaned_at": datetime.utcnow(),
                    "cleaned_by": "cleanup_script"
                }}
            )

            print(f"\n🎉 Successfully removed {removed_count} orphaned image references!")

    finally:
        await db.close()


async def scan_all_documents():
    """Scan all documents and report orphaned images"""
    db = DatabaseManager()
    await db.initialize()

    try:
        print("\n🔍 Scanning all documents for orphaned images...\n")

        documents = await db.mongo_find("documents", {})
        total_documents_affected = 0
        total_orphaned_images = 0

        for doc in documents:
            document_id = doc.get("document_id")
            if not document_id:
                continue

            orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

            if orphaned_by_question:
                total_documents_affected += 1
                orphaned_count = sum(len(ids) for ids in orphaned_by_question.values())
                total_orphaned_images += orphaned_count

                print(f"📄 {doc.get('title', document_id)}")
                print(f"   Document ID: {document_id}")
                print(f"   Affected questions: {len(orphaned_by_question)}")
                print(f"   Orphaned images: {orphaned_count}\n")

        if total_documents_affected == 0:
            print("✅ No orphaned images found in any document!")
        else:
            print(f"\n📊 Summary:")
            print(f"   Documents affected: {total_documents_affected}")
            print(f"   Total orphaned images: {total_orphaned_images}")
            print(f"\n💡 Run with --clean-all to remove all orphaned references")

    finally:
        await db.close()


async def clean_all_documents():
    """Clean orphaned images from all documents"""
    db = DatabaseManager()
    await db.initialize()

    try:
        print("\n🧹 Cleaning orphaned images from all documents...\n")

        documents = await db.mongo_find("documents", {})
        total_questions_cleaned = 0
        total_images_removed = 0

        for doc in documents:
            document_id = doc.get("document_id")
            if not document_id:
                continue

            orphaned_by_question = await get_orphaned_images_in_document(document_id, db)

            if orphaned_by_question:
                print(f"📄 Cleaning {doc.get('title', document_id)}...")

                for question_id, orphaned_ids in orphaned_by_question.items():
                    question = await db.mongo_find_one("questions", {"id": question_id})
                    if not question:
                        continue

                    cleaned_question, removed_count = await clean_question_images(question, db)

                    if removed_count > 0:
                        await db.mongo_update_one(
                            "questions",
                            {"id": question_id},
                            {"$set": {
                                "images": cleaned_question.get("images", []),
                                "question_figures": cleaned_question.get("question_figures", []),
                                "cleaned_at": datetime.utcnow(),
                                "cleaned_by": "cleanup_script"
                            }}
                        )

                        total_questions_cleaned += 1
                        total_images_removed += removed_count

                print(f"   ✅ Cleaned {len(orphaned_by_question)} questions\n")

        print(f"\n🎉 Cleanup complete!")
        print(f"   Questions cleaned: {total_questions_cleaned}")
        print(f"   Total images removed: {total_images_removed}")

    finally:
        await db.close()


def print_usage():
    """Print usage instructions"""
    print("""
Usage:
    python clean_orphaned_images.py [COMMAND] [ARGS]

Commands:
    --scan                      Scan all documents and report orphaned images
    --clean-all                 Clean orphaned images from all documents
    --document <document_id>    Clean orphaned images from specific document
    --question <question_id>    Clean orphaned images from specific question

Examples:
    # Scan all documents
    python clean_orphaned_images.py --scan

    # Clean specific document
    python clean_orphaned_images.py --document phy006

    # Clean specific question
    python clean_orphaned_images.py --question 690603d5edeb5d77f93b20b3

    # Clean all documents
    python clean_orphaned_images.py --clean-all
""")


async def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1]

    if command == "--scan":
        await scan_all_documents()
    elif command == "--clean-all":
        await clean_all_documents()
    elif command == "--document":
        if len(sys.argv) < 3:
            print("❌ Error: Document ID required")
            print_usage()
            return
        await clean_document(sys.argv[2])
    elif command == "--question":
        if len(sys.argv) < 3:
            print("❌ Error: Question ID required")
            print_usage()
            return
        await clean_question(sys.argv[2])
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    asyncio.run(main())
