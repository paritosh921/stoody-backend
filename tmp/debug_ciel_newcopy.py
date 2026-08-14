"""Inspect new Goodley/Maths papers vs rohan21 visibility."""
import json
import os
from pathlib import Path

from pymongo import MongoClient

env_path = Path(__file__).resolve().parents[1] / ".env"
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    os.environ.setdefault(key.strip(), value.strip())

client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=20000)
db = client["skb_ciel-1008"]

keep_doc = [
    "document_id",
    "title",
    "document_type",
    "subject",
    "standard",
    "section",
    "course_plan",
    "question_type",
    "exam_mode",
    "is_active",
    "is_validated",
    "ocr_status",
    "exam_finalized",
    "exam_finalized_at",
    "extracted_questions_count",
    "total_minutes",
    "created_at",
    "uploaded_at",
    "updated_at",
    "admin_id",
]

print("=== RECENT / GOODLEY / MATHS / COPY DOCUMENTS ===")
docs = list(
    db["documents"].find(
        {
            "$or": [
                {"title": {"$regex": "Goodley|Maths|UT|copy", "$options": "i"}},
                {"document_id": {"$regex": "MathsUT|Goodley|copy", "$options": "i"}},
                {"exam_mode": "pcr"},
            ]
        }
    ).sort("created_at", -1)
)
print("count", len(docs))
for doc in docs:
    print(json.dumps({k: doc.get(k) for k in keep_doc}, default=str))
    print("---")

print("\n=== ALL EXAMPEN EXAMS ===")
keep_exam = [
    "exam_id",
    "title",
    "exam_type",
    "lifecycle_state",
    "prepared_document_id",
    "session_request_id",
    "student_self_submission_enabled",
    "capture_mode",
    "created_at",
    "updated_at",
    "started_at",
]
for exam in db["exampen_exams"].find().sort("created_at", -1):
    roster = exam.get("roster") or []
    payload = {k: exam.get(k) for k in keep_exam}
    payload["roster_count"] = len(roster)
    payload["rohan21_on_roster"] = "rohan21" in roster
    payload["roster_tail"] = roster[-8:]
    print(json.dumps(payload, default=str))
    print("---")

print("\n=== ROHAN21 ===")
rohan = db["students"].find_one({"username": "rohan21"})
print(
    json.dumps(
        {
            "username": rohan.get("username") if rohan else None,
            "student_id": rohan.get("student_id") if rohan else None,
            "grade": rohan.get("grade") if rohan else None,
            "section": rohan.get("section") if rohan else None,
            "is_active": rohan.get("is_active") if rohan else None,
            "admin_id": str(rohan.get("admin_id")) if rohan else None,
            "created_at": rohan.get("created_at") if rohan else None,
        },
        default=str,
        indent=2,
    )
)

print("\n=== CLASS 9 ACTIVE COUNT ===")
print(db["students"].count_documents({"grade": "9", "is_active": True}))
print("grade 9 any active field missing", db["students"].count_documents({"grade": "9"}))
