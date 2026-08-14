"""Read-only investigation: why Maths Test Goodley UT is hidden for rohan21."""
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
master = client["skb_master"]

print("=== TENANTS matching CIEL ===")
tenants = list(
    master["tenants"].find(
        {
            "$or": [
                {"tenant_id": {"$regex": "CIEL", "$options": "i"}},
                {"institution_id": {"$regex": "CIEL", "$options": "i"}},
                {"name": {"$regex": "Ciel", "$options": "i"}},
                {"institution_name": {"$regex": "Ciel", "$options": "i"}},
                {"school_name": {"$regex": "Ciel", "$options": "i"}},
            ]
        }
    )
)
for tenant in tenants:
    print(
        {
            "tenant_id": tenant.get("tenant_id"),
            "institution_id": tenant.get("institution_id"),
            "db_name": tenant.get("db_name"),
            "status": tenant.get("status"),
            "name": tenant.get("name") or tenant.get("institution_name") or tenant.get("school_name"),
        }
    )

target_ids = ["CIEL-1008", "CIEL-1001", "CIEL-0001"]
chosen = None
for tenant in tenants:
    if tenant.get("tenant_id") in target_ids or tenant.get("institution_id") in target_ids:
        if tenant.get("tenant_id") == "CIEL-1008" or tenant.get("institution_id") == "CIEL-1008":
            chosen = tenant
            break
        if chosen is None:
            chosen = tenant

if chosen is None:
    print("No matching tenant found in regex search, trying exact lookups")
    for key in ("tenant_id", "institution_id"):
        for value in target_ids:
            doc = master["tenants"].find_one({key: value})
            if doc:
                print("exact", key, value, {"db_name": doc.get("db_name"), "status": doc.get("status")})
                if value == "CIEL-1008":
                    chosen = doc

if not chosen:
    raise SystemExit("Tenant CIEL-1008 not found")

db_name = chosen.get("db_name")
print("\n=== CHOSEN TENANT ===")
print(
    json.dumps(
        {
            "tenant_id": chosen.get("tenant_id"),
            "institution_id": chosen.get("institution_id"),
            "db_name": db_name,
            "status": chosen.get("status"),
        },
        default=str,
    )
)

tenant_db = client[db_name]
print("collections:", sorted(tenant_db.list_collection_names())[:40])

print("\n=== STUDENT rohan21 ===")
students = list(
    tenant_db["students"].find(
        {
            "$or": [
                {"username": {"$regex": "^rohan21$", "$options": "i"}},
                {"username_lower": "rohan21"},
                {"student_id": {"$regex": "rohan21", "$options": "i"}},
                {"name": {"$regex": "rohan21", "$options": "i"}},
            ]
        }
    )
)
if not students:
    students = list(tenant_db["students"].find({"username": {"$regex": "rohan", "$options": "i"}}))

keep_student = [
    "_id",
    "username",
    "username_lower",
    "student_id",
    "name",
    "grade",
    "standard",
    "section",
    "subjects",
    "plan_types",
    "course_plan",
    "teacher_ids",
    "admin_id",
    "is_active",
    "class_level",
]
for student in students:
    print(json.dumps({k: student.get(k) for k in keep_student}, default=str, indent=2))

print("\n=== DOCUMENT Maths Test Goodley / MathsUT1 ===")
docs = list(
    tenant_db["documents"].find(
        {
            "$or": [
                {"title": {"$regex": "Goodley", "$options": "i"}},
                {"title": {"$regex": "Maths Test", "$options": "i"}},
                {"code": {"$regex": "MathsUT1", "$options": "i"}},
                {"document_id": {"$regex": "MathsUT1", "$options": "i"}},
            ]
        }
    )
)
keep_doc = [
    "_id",
    "document_id",
    "title",
    "code",
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
    "admin_id",
    "teacher_ids",
    "exam_finalized",
    "extracted_questions_count",
    "total_points",
    "total_minutes",
    "uploaded_by",
    "uploaded_by_name",
]
for doc in docs:
    print(json.dumps({k: doc.get(k) for k in keep_doc}, default=str, indent=2))

print("\n=== EXAM SESSIONS for those documents ===")
doc_ids = [d.get("document_id") for d in docs if d.get("document_id")]
if "exampen_exams" in tenant_db.list_collection_names():
    exams = list(tenant_db["exampen_exams"].find({"prepared_document_id": {"$in": doc_ids}}))
    if not exams:
        exams = list(
            tenant_db["exampen_exams"].find(
                {"title": {"$regex": "Goodley|MathsUT1|Maths Test", "$options": "i"}}
            )
        )
    keep_exam = [
        "exam_id",
        "title",
        "exam_type",
        "lifecycle_state",
        "prepared_document_id",
        "roster",
        "student_self_submission_enabled",
        "capture_mode",
        "subject",
        "code",
    ]
    for exam in exams:
        payload = {k: exam.get(k) for k in keep_exam}
        roster = exam.get("roster") or []
        payload["roster_count"] = len(roster)
        payload["roster_sample"] = roster[:20]
        print(json.dumps(payload, default=str, indent=2))
else:
    print("no exampen_exams collection")

print("\n=== CLASS 9 students sample ===")
for student in tenant_db["students"].find({"grade": {"$in": ["9", 9, "Class 9", "IX", "9th"]}}).limit(8):
    print(
        {
            "username": student.get("username"),
            "student_id": student.get("student_id"),
            "grade": student.get("grade"),
            "section": student.get("section"),
            "subjects": student.get("subjects"),
            "plan_types": student.get("plan_types"),
            "is_active": student.get("is_active"),
        }
    )

print("\n=== grade value counts ===")
pipeline = [{"$group": {"_id": "$grade", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
print(list(tenant_db["students"].aggregate(pipeline)))

print("\n=== document standard value counts ===")
print(list(tenant_db["documents"].aggregate([{"$group": {"_id": "$standard", "count": {"$sum": 1}}}])))
