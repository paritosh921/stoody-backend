import asyncio
from core.database import DatabaseManager

async def check():
    db = DatabaseManager()
    await db.initialize()

    student = await db.mongo_find_one('students', {'name': 'Paritosh'})
    if student:
        print('Student Paritosh:')
        print(f"  _id: {student.get('_id')}")
        print(f"  student_id: {student.get('student_id')}")
        print(f"  grade: {student.get('grade')}")
        print(f"  subjects: {student.get('subjects')}")
        print(f"  plan_types: {student.get('plan_types')}")
    else:
        print('Student not found')

    await db.close()

asyncio.run(check())
