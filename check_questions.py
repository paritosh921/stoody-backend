import asyncio
from core.database import DatabaseManager

async def check():
    db = DatabaseManager()
    await db.initialize()

    # Get the test series document
    doc = await db.mongo_find_one('documents', {'document_id': 'tst05'})
    if doc:
        print(f"Test series: {doc.get('title')}")
        print(f"  document_id: {doc.get('document_id')}")
        print(f"  extracted_questions_count: {doc.get('extracted_questions_count')}")

    # Get questions for this test series
    questions = await db.mongo_find('questions', {'document_id': 'tst05'})
    print(f"\nFound {len(questions)} questions in database")

    if questions:
        print("\nFirst question:")
        q = questions[0]
        for key, value in q.items():
            if key != 'images':
                print(f"  {key}: {value}")

    await db.close()

asyncio.run(check())
