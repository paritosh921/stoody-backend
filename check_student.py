import asyncio
from core.database import DatabaseManager

async def check():
    db = DatabaseManager()
    await db.initialize()

    # Find student with subdomain paritosh
    student = await db.mongo_find_one('students', {'student_id': 'paritosh'})
    if student:
        print('Student found:')
        print(f"  Grade: {student.get('grade')} (type: {type(student.get('grade')).__name__})")
        print(f"  Subjects: {student.get('subjects')}")
        print(f"  Plan types: {student.get('plan_types')}")
    else:
        print('Student not found')

    # Check test series document
    doc = await db.mongo_find_one('documents', {'document_type': 'Test Series'})
    if doc:
        print('\nTest Series document:')
        print(f"  Standard: {doc.get('standard')} (type: {type(doc.get('standard')).__name__})")
        print(f"  Subject: {doc.get('subject')}")
        print(f"  Course plan: {doc.get('course_plan')}")

    await db.close()

asyncio.run(check())
