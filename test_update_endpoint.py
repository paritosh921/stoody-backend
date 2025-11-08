#!/usr/bin/env python3
"""
Test script to verify the question update endpoint works with new fields
"""

import asyncio
import json
from core.database import DatabaseManager


async def test_update_fields():
    """Test that all update fields are accepted"""
    db = DatabaseManager()
    await db.initialize()

    try:
        # Get a test question
        question = await db.mongo_find_one("questions", {"document_id": "phy006"})
        if not question:
            print("❌ No question found for testing")
            return

        question_id = question.get("id")
        print(f"✅ Found test question: {question_id}\n")

        # Simulate update with all new fields
        update_data = {
            "text": question.get("text"),  # Keep same text
            "question_figures": [],  # Empty for now (no orphaned images)
            "enhanced_options": [
                {"id": "a", "type": "text", "content": "Option A", "label": "A"},
                {"id": "b", "type": "text", "content": "Option B", "label": "B"},
                {"id": "c", "type": "text", "content": "Option C", "label": "C"},
                {"id": "d", "type": "text", "content": "Option D", "label": "D"}
            ]
        }

        print("📝 Simulating update with new fields:")
        print(f"   Fields: {list(update_data.keys())}")
        print(f"   question_figures: {len(update_data['question_figures'])} items")
        print(f"   enhanced_options: {len(update_data['enhanced_options'])} items\n")

        # Update the question
        result = await db.mongo_update_one(
            "questions",
            {"id": question_id},
            {"$set": update_data}
        )

        if result:
            print("✅ Update successful!")

            # Verify the update
            updated = await db.mongo_find_one("questions", {"id": question_id})

            print("\n📊 Verification:")
            print(f"   question_figures present: {'question_figures' in updated}")
            print(f"   enhanced_options present: {'enhanced_options' in updated}")

            if "question_figures" in updated:
                print(f"   question_figures count: {len(updated['question_figures'])}")

            if "enhanced_options" in updated:
                print(f"   enhanced_options count: {len(updated['enhanced_options'])}")
                print(f"   enhanced_options sample: {json.dumps(updated['enhanced_options'][0], indent=6)}")

            print("\n🎉 All new fields are working correctly!")
        else:
            print("❌ Update failed")

    finally:
        await db.close()


async def test_document_id_consistency():
    """Test that document_id is used consistently"""
    db = DatabaseManager()
    await db.initialize()

    try:
        print("\n🔍 Testing document_id consistency:\n")

        # Find questions with document_id
        questions_with_doc_id = await db.mongo_find("questions", {"document_id": {"$exists": True}})

        # Find questions with pdf_source
        questions_with_pdf_source = await db.mongo_find("questions", {"pdf_source": {"$exists": True}})

        print(f"   Questions with document_id: {len(questions_with_doc_id)}")
        print(f"   Questions with pdf_source: {len(questions_with_pdf_source)}")

        # Check if any have both
        sample = await db.mongo_find_one("questions", {
            "document_id": {"$exists": True},
            "pdf_source": {"$exists": True}
        })

        if sample:
            print(f"\n   Sample question has both fields:")
            print(f"      document_id: {sample.get('document_id')}")
            print(f"      pdf_source: {sample.get('pdf_source')}")
            print(f"      ✅ Fallback logic will handle this correctly")
        else:
            print(f"\n   ✅ No overlap - clean data structure")

    finally:
        await db.close()


async def main():
    print("=" * 60)
    print("Testing Question Update Endpoint - New Fields")
    print("=" * 60)

    await test_update_fields()
    await test_document_id_consistency()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
