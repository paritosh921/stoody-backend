"""Compare class 9 students vs exam roster and timestamps."""
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

exam = db["exampen_exams"].find_one({"prepared_document_id": "MathsUT1"})
roster = set(exam.get("roster") or [])
print("exam created_at", exam.get("created_at"))
print("exam updated_at", exam.get("updated_at"))
print("exam lifecycle", exam.get("lifecycle_state"))
print("roster_count", len(roster))

class9 = list(db["students"].find({"grade": {"$in": ["9", 9]}}))
print("class9_count", len(class9))

missing = []
for student in class9:
    sid = student.get("student_id")
    username = student.get("username")
    on_roster = sid in roster or username in roster
    if not on_roster:
        missing.append(
            {
                "username": username,
                "student_id": sid,
                "name": student.get("name"),
                "grade": student.get("grade"),
                "section": student.get("section"),
                "is_active": student.get("is_active"),
                "created_at": student.get("created_at"),
                "updated_at": student.get("updated_at"),
            }
        )

print("\n=== CLASS 9 NOT ON ROSTER ===")
print(json.dumps(missing, default=str, indent=2))

roster_not_in_class9 = []
class9_ids = {s.get("student_id") for s in class9} | {s.get("username") for s in class9}
for sid in sorted(roster):
    if sid not in class9_ids:
        roster_not_in_class9.append(sid)
print("\n=== ROSTER NOT IN CLASS 9 ===")
print(roster_not_in_class9)

rohan = db["students"].find_one({"username": "rohan21"})
print("\n=== ROHAN21 TIMESTAMPS ===")
print(
    json.dumps(
        {
            "created_at": rohan.get("created_at"),
            "updated_at": rohan.get("updated_at"),
            "student_id": rohan.get("student_id"),
            "grade": rohan.get("grade"),
            "section": rohan.get("section"),
            "is_active": rohan.get("is_active"),
            "on_roster": rohan.get("student_id") in roster,
        },
        default=str,
        indent=2,
    )
)

print("\n=== ROHAN21 SUBMISSIONS ===")
subs = list(
    db["evalpen_submissions"].find(
        {
            "$or": [
                {"student_id": "rohan21"},
                {"student_id": str(rohan["_id"])},
            ]
        }
    )
)
print("submissions", len(subs))
for sub in subs:
    print(
        {
            "exam_id": sub.get("exam_id"),
            "student_id": sub.get("student_id"),
            "publication_status": sub.get("publication_status"),
            "submitted_at": sub.get("submitted_at"),
        }
    )
