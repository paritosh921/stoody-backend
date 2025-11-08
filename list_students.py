import asyncio
from core.database import DatabaseManager

async def check():
    db = DatabaseManager()
    await db.initialize()

    students = await db.mongo_find('students', {}, limit=10)
    print(f'Found {len(students)} students:')
    for s in students:
        print(f"\n  _id: {s.get('_id')}")
        print(f"  student_id: {s.get('student_id')}")
        print(f"  name: {s.get('name')}")
        print(f"  subdomain: {s.get('subdomain')}")
        print(f"  grade: {s.get('grade')} (type: {type(s.get('grade')).__name__})")

    await db.close()

asyncio.run(check())
