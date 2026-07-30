import asyncio
import json
import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
EXAM_ID = "exam-ccb8b1c2da064083872dbeb84d1ef488"


async def main():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    for db_name in ["skb_indl-ciel-1001", "skb_ciel-1008", "skb_master"]:
        db = client[db_name]
        exam = await db["exampen_exams"].find_one({"exam_id": EXAM_ID})
        if not exam:
            exam = await db["exampen_exams"].find_one({"title": "jee13"})
        if not exam:
            continue
        print("=== DB", db_name, "===")
        exam_id = exam["exam_id"]
        print(
            json.dumps(
                {
                    k: exam.get(k)
                    for k in [
                        "exam_id",
                        "title",
                        "exam_type",
                        "paper_version_id",
                        "prepared_document_id",
                        "pcr_grading_contract",
                    ]
                },
                default=str,
                indent=2,
            )
        )
        subs = await db["evalpen_submissions"].find({"exam_id": exam_id}).to_list(20)
        print("subs", len(subs))
        for s in subs:
            print(
                "SUB",
                json.dumps(
                    {
                        k: s.get(k)
                        for k in [
                            "submission_id",
                            "student_id",
                            "processing_path",
                            "review_state",
                            "segmentation_status",
                            "document_grading_run_id",
                            "grading_input_hash",
                            "document_review",
                        ]
                    },
                    default=str,
                )[:2000],
            )
        jobs = (
            await db["exampen_processing_jobs"]
            .find({"exam_id": exam_id})
            .sort("updated_at", -1)
            .to_list(20)
        )
        for j in jobs:
            print(
                "JOB",
                json.dumps(
                    {
                        k: j.get(k)
                        for k in [
                            "job_id",
                            "student_id",
                            "status",
                            "last_error",
                            "processing_path",
                            "required_processing_path",
                            "attempts",
                            "reprocess_count",
                            "progress",
                            "evaluation",
                            "segmentation",
                            "review",
                        ]
                    },
                    default=str,
                )[:2500],
            )
        sub_ids = [s.get("submission_id") for s in subs]
        for col in [
            "evalpen_document_grading_runs",
            "evalpen_objective_grading_runs",
        ]:
            runs = (
                await db[col]
                .find(
                    {
                        "$or": [
                            {"exam_id": exam_id},
                            {"submission_id": {"$in": sub_ids}},
                        ]
                    }
                )
                .sort("updated_at", -1)
                .to_list(20)
            )
            print(col, len(runs))
            for r in runs[:8]:
                print(
                    json.dumps(
                        {
                            k: r.get(k)
                            for k in [
                                "run_id",
                                "submission_id",
                                "student_id",
                                "status",
                                "grading_revision",
                                "generation_revision",
                                "prompt_version",
                                "extraction_path",
                                "input_fingerprint",
                                "result",
                                "generation_error",
                                "generation_lease_token",
                                "updated_at",
                            ]
                        },
                        default=str,
                    )[:1800]
                )
        return
    print("exam not found in known dbs")


asyncio.run(main())
