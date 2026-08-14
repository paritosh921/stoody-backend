"""Inspect MathsUT2 / rohan21 upload vs teacher missing state."""
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

exam_id = "exam-4e55932845e743208b02522013ca6035"

print("=== EXAM ===")
exam = db["exampen_exams"].find_one({"exam_id": exam_id})
if not exam:
    exam = db["exampen_exams"].find_one({"prepared_document_id": "MathsUT2"})
    print("looked up by document, found", exam.get("exam_id") if exam else None)

if exam:
    roster = exam.get("roster") or []
    print(
        json.dumps(
            {
                "exam_id": exam.get("exam_id"),
                "title": exam.get("title"),
                "prepared_document_id": exam.get("prepared_document_id"),
                "lifecycle_state": exam.get("lifecycle_state"),
                "capture_mode": exam.get("capture_mode"),
                "student_self_submission_enabled": exam.get("student_self_submission_enabled"),
                "session_request_id": exam.get("session_request_id"),
                "roster_count": len(roster),
                "rohan21_on_roster": "rohan21" in roster,
                "mongo_id_on_roster": any("6a7ef13b" in str(x) for x in roster),
                "created_at": exam.get("created_at"),
                "updated_at": exam.get("updated_at"),
            },
            default=str,
            indent=2,
        )
    )

print("\n=== DOCUMENT MathsUT2 ===")
doc = db["documents"].find_one({"document_id": "MathsUT2"})
if doc:
    print(
        json.dumps(
            {
                "is_active": doc.get("is_active"),
                "exam_finalized": doc.get("exam_finalized"),
                "file_path": doc.get("file_path"),
                "storage_path": doc.get("storage_path"),
                "source_storage_path": doc.get("source_storage_path"),
                "ocr_status": doc.get("ocr_status"),
            },
            default=str,
            indent=2,
        )
    )

ids = ["rohan21", "6a7ef13b78c58f9c977d93a5"]
print("\n=== evalpen_submissions ===")
for rec in db["evalpen_submissions"].find(
    {"$or": [{"exam_id": exam.get("exam_id") if exam else exam_id}, {"student_id": {"$in": ids}}]}
):
    print(
        json.dumps(
            {
                "exam_id": rec.get("exam_id"),
                "student_id": rec.get("student_id"),
                "submission_id": rec.get("submission_id"),
                "publication_status": rec.get("publication_status"),
                "segmentation_status": rec.get("segmentation_status"),
                "page_count": rec.get("page_count"),
                "submitted_at": rec.get("submitted_at"),
                "source": rec.get("source"),
            },
            default=str,
        )
    )

print("\n=== exampen_student_copy_uploads ===")
for rec in db["exampen_student_copy_uploads"].find(
    {"$or": [{"exam_id": exam.get("exam_id") if exam else exam_id}, {"student_id": {"$in": ids}}]}
):
    print(json.dumps({k: rec.get(k) for k in rec if k != "_id"}, default=str))

print("\n=== evalpen_answer_pages for these submissions ===")
subs = list(
    db["evalpen_submissions"].find(
        {"$or": [{"exam_id": exam.get("exam_id") if exam else exam_id}, {"student_id": {"$in": ids}}]},
        {"submission_id": 1},
    )
)
sub_ids = [s.get("submission_id") for s in subs if s.get("submission_id")]
print("submission_ids", sub_ids)
if sub_ids:
    pages = list(db["evalpen_answer_pages"].find({"submission_id": {"$in": sub_ids}}))
    print("page_count", len(pages))
    for p in pages[:8]:
        print(
            {
                "submission_id": p.get("submission_id"),
                "page_index": p.get("page_index"),
                "raw_image_ref": bool(p.get("raw_image_ref")),
                "student_id": p.get("student_id"),
            }
        )

print("\n=== processing jobs ===")
for rec in db["exampen_processing_jobs"].find(
    {"$or": [{"exam_id": exam.get("exam_id") if exam else exam_id}, {"student_id": {"$in": ids}}]}
):
    print(
        json.dumps(
            {
                "job_id": rec.get("job_id"),
                "exam_id": rec.get("exam_id"),
                "student_id": rec.get("student_id"),
                "submission_id": rec.get("submission_id"),
                "status": rec.get("status"),
                "last_error": rec.get("last_error"),
                "created_at": rec.get("created_at"),
                "updated_at": rec.get("updated_at"),
            },
            default=str,
        )
    )
