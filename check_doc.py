import asyncio
from core.database import DatabaseManager

async def check():
    db = DatabaseManager()
    await db.initialize()
    doc = await db.mongo_find_one('documents', {'document_type': 'Test Series'})
    if doc:
        print('Keys:', list(doc.keys()))
        print('\nDocument structure:')
        for key, value in doc.items():
            print(f"  {key}: {value}")
    else:
        print('No test series document found')
    await db.close()

asyncio.run(check())
