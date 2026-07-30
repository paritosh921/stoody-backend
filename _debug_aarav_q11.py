import asyncio
import json
import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
EXAM = "exam-44bcab9707494bcc85ba272a99c4da86"
DB = "skb_sgtb-0001"


async def main():
    db = AsyncIOMotorClient(os.getenv("MONGODB_URI"))[DB]
    subs = await db["evalpen_submissions"].find({"exam_id": EXAM}).to_list(20)
    target = None
    for s in subs:
        sid = str(s.get("student_id") or "")
        print(
            sid,
            s.get("review_state"),
            s.get("processing_path"),
            (s.get("document_review") or {}).get("warnings"),
        )
        if "aarav" in sid.lower() or "Aarav" in sid:
            target = s
    if not target:
        print("Aarav submission not found")
        return
    print("SUB", target.get("submission_id"))
    print(
        "document_review",
        json.dumps(target.get("document_review"), default=str, indent=2)[:2000],
    )
    resps = (
        await db["evalpen_detected_responses"]
        .find(
            {
                "submission_id": target["submission_id"],
                "superseded_at": {"$exists": False},
            }
        )
        .sort("question_number", 1)
        .to_list(50)
    )
    print("responses", len(resps))
    for r in resps:
        flags = r.get("flags") or []
        reasons = [f.get("reason") for f in flags]
        print(
            f"Q{r.get('question_number')} eval={r.get('eval_status')} "
            f"state={r.get('answer_state')} reason={r.get('manual_review_reason')!r} "
            f"flags={reasons}"
        )
    r11 = next((r for r in resps if r.get("question_number") == 11), None)
    if not r11:
        print("no Q11")
        return
    print("Q11 visual_evidence:")
    print(json.dumps(r11.get("visual_evidence"), default=str, indent=2)[:4000])
    print("Q11 source_pages:")
    print(json.dumps(r11.get("source_pages"), default=str, indent=2)[:2000])
    evals = await db["evalpen_evaluations"].find(
        {"response_id": r11.get("response_id")}
    ).to_list(5)
    print("evals for q11", len(evals))
    if evals:
        print(json.dumps({k: evals[0].get(k) for k in ["total_score", "eval_path", "manual_review_required", "overall_feedback"]}, default=str))


asyncio.run(main())
