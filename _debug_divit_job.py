import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
EXAM = "exam-44bcab9707494bcc85ba272a99c4da86"
DB = "skb_sgtb-0001"


async def main():
    db = AsyncIOMotorClient(os.getenv("MONGODB_URI"))[DB]
    jobs = (
        await db["exampen_processing_jobs"]
        .find({"exam_id": EXAM})
        .sort("updated_at", -1)
        .to_list(20)
    )
    print("jobs", len(jobs))
    now = datetime.now(timezone.utc)
    for j in jobs:
        started = j.get("started_at") or j.get("updated_at")
        age = None
        if started is not None:
            if getattr(started, "tzinfo", None) is None:
                started = started.replace(tzinfo=timezone.utc)
            age = (now - started).total_seconds()
        print(
            json.dumps(
                {
                    "student_id": j.get("student_id"),
                    "status": j.get("status"),
                    "attempts": j.get("attempts"),
                    "processing_path": j.get("processing_path"),
                    "last_error": (j.get("last_error") or "")[:180],
                    "started_at": j.get("started_at"),
                    "updated_at": j.get("updated_at"),
                    "age_sec": age,
                    "progress": j.get("progress"),
                    "lease_expires_at": j.get("lease_expires_at"),
                    "evaluation": j.get("evaluation"),
                    "segmentation": j.get("segmentation"),
                },
                default=str,
            )
        )

    runs = (
        await db["evalpen_document_grading_runs"]
        .find({"exam_id": EXAM})
        .sort("updated_at", -1)
        .to_list(20)
    )
    print("\nruns", len(runs))
    for r in runs[:12]:
        grades = r.get("evidence_graph_question_grades") or {}
        usages = r.get("evidence_graph_question_grade_usages") or {}
        print(
            r.get("student_id"),
            r.get("status"),
            r.get("prompt_version"),
            "map",
            bool(r.get("evidence_graph_mapping")),
            "grades",
            len(grades) if isinstance(grades, dict) else 0,
            "usage_keys",
            list(usages.keys())[:8] if isinstance(usages, dict) else None,
            "err",
            (r.get("generation_error") or "")[:120],
            "tokens",
            (r.get("token_usage") or {}).get("total_tokens"),
            "updated",
            r.get("updated_at"),
        )

    exam = await db["exampen_exams"].find_one({"exam_id": EXAM})
    print("\ncontract", exam.get("pcr_grading_contract") if exam else None)


asyncio.run(main())
