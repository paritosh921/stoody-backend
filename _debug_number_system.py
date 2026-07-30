import asyncio
import json
import os
from collections import Counter

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
EXAM_ID = "exam-44bcab9707494bcc85ba272a99c4da86"


async def main():
    client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
    dbs = await client.list_database_names()
    tenant_dbs = [n for n in dbs if n.startswith("skb_")]
    for db_name in tenant_dbs:
        db = client[db_name]
        exam = await db["exampen_exams"].find_one({"exam_id": EXAM_ID})
        if not exam:
            continue
        print("=== DB", db_name, "===")
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
        paper = await db["exampen_paper_versions"].find_one(
            {"paper_version_id": exam.get("paper_version_id")}
        )
        print("paper_context", json.dumps((paper or {}).get("paper_context"), default=str, indent=2))
        qs = await db["evalpen_questions"].find({"exam_id": EXAM_ID}).to_list(100)
        print(
            "questions",
            len(qs),
            "modes",
            dict(Counter(str(q.get("grading_mode") or q.get("question_type")) for q in qs)),
        )
        if qs:
            print(
                "sample q",
                {
                    k: qs[0].get(k)
                    for k in [
                        "question_number",
                        "question_type",
                        "grading_mode",
                        "max_marks",
                        "marking_criteria",
                    ]
                },
            )
        jobs = (
            await db["exampen_processing_jobs"]
            .find({"exam_id": EXAM_ID})
            .sort("updated_at", -1)
            .to_list(30)
        )
        print("jobs", len(jobs))
        for j in jobs:
            print(
                j.get("student_id"),
                j.get("status"),
                j.get("processing_path"),
                "attempts",
                j.get("attempts"),
                "err",
                (j.get("last_error") or "")[:180],
                "eval",
                j.get("evaluation"),
                "seg",
                j.get("segmentation"),
            )
        runs = (
            await db["evalpen_document_grading_runs"]
            .find({"exam_id": EXAM_ID})
            .sort("updated_at", -1)
            .to_list(30)
        )
        print("runs", len(runs))
        for r in runs:
            usage = r.get("token_usage") or {}
            print(
                r.get("student_id"),
                r.get("status"),
                r.get("prompt_version"),
                "pages",
                r.get("page_count"),
                "err",
                (r.get("generation_error") or "")[:160],
                "result",
                r.get("result"),
                "tokens",
                usage.get("total_tokens"),
                "model",
                usage.get("model") or r.get("model_used"),
            )
            # checkpoint fields for evidence graph
            has_map = bool(r.get("evidence_graph_mapping"))
            grades = r.get("evidence_graph_question_grades") or {}
            print(
                "  map",
                has_map,
                "saved_grade_keys",
                list(grades.keys())[:20] if isinstance(grades, dict) else type(grades),
                "batch_usages",
                list((r.get("evidence_graph_question_grade_usages") or {}).keys())[:10],
            )
        return
    print("exam not found in any skb_ db")


asyncio.run(main())
