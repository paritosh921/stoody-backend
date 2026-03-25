#!/usr/bin/env python3
"""
ExamPen E2E Pipeline Test Script

Usage:
    python scripts/test_exampen_pipeline.py --base-url http://localhost:5001 --token <JWT>

Exercises the full pipeline:
1. Register question metadata + answer keys
2. Upload reference solution
3. Create a submission (simulating hub upload)
4. Test practice evaluation (text)
5. Check gate usage
6. List submissions
7. Test flagged queue
8. Test review summary

This is a manual verification script using httpx to hit the actual API
endpoints.  It is NOT a pytest file -- no exam-conductor imports, no
direct DB access.  Only httpx (async) and standard library.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    name: str
    passed: bool
    detail: str = ""


@dataclass
class PipelineState:
    """Mutable state bag threaded through steps."""
    results: list[StepResult] = field(default_factory=list)
    submission_id: Optional[str] = None
    question_id: str = ""
    exam_id: str = ""
    student_id: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _record(state: PipelineState, step: int, name: str, passed: bool, detail: str = "") -> None:
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"{tag} Step {step}: {detail or name}")
    state.results.append(StepResult(step=step, name=name, passed=passed, detail=detail))


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------

async def step1_register_question(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 1: Register question metadata."""
    url = f"{base}/api/v1/evalpen/questions"
    payload = {
        "question_id": state.question_id,
        "exam_id": state.exam_id,
        "subject": "Mathematics",
        "question_type": "short_answer",
        "complexity": "L1",
        "eval_template": "stepwise_numerical",
        "max_marks": 5,
    }
    try:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code in (200, 201, 202):
            body = resp.json()
            status = body.get("status", "unknown")
            _record(state, 1, "Register question", True,
                    f"Question metadata registered ({state.question_id}, status={status})")
        else:
            _record(state, 1, "Register question", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 1, "Register question", False, f"Exception: {exc}")


async def step2_upload_solution(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 2: Upload reference solution."""
    url = f"{base}/api/v1/evalpen/solutions/{state.question_id}"
    payload = {
        "reference_solution": (
            "The answer is 42. This is derived from computing the product "
            "of 6 and 7, which yields 42."
        ),
        "solution_source": "teacher",
    }
    try:
        resp = await client.put(url, json=payload, headers=headers)
        if resp.status_code in (200, 201, 202):
            body = resp.json()
            version = body.get("version", "?")
            _record(state, 2, "Upload solution", True,
                    f"Reference solution uploaded (v{version})")
        else:
            _record(state, 2, "Upload solution", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 2, "Upload solution", False, f"Exception: {exc}")


async def step3_create_submission(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 3: Create a conducted-exam submission."""
    url = f"{base}/api/v1/evalpen/submissions"
    payload = {
        "exam_id": state.exam_id,
        "student_id": state.student_id,
        "source": "camera",
        "page_refs": [{"page_num": 1}],
    }
    try:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code in (200, 201, 202):
            body = resp.json()
            state.submission_id = body.get("submission_id")
            seg_status = body.get("segmentation_status", "?")
            _record(state, 3, "Create submission", True,
                    f"Submission created (submission_id: {state.submission_id}, "
                    f"segmentation: {seg_status})")
        else:
            _record(state, 3, "Create submission", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 3, "Create submission", False, f"Exception: {exc}")


async def step4_practice_evaluate(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 4: Test practice evaluation (text, stateless)."""
    url = f"{base}/api/v1/evalpen/practice/evaluate"
    payload = {
        "question_id": state.question_id,
        "source_type": "canvas",
        "text": "The answer is 42 because it is the product of 6 and 7.",
    }
    try:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            body = resp.json()
            score = body.get("total_score", "?")
            max_score = body.get("max_score", "?")
            _record(state, 4, "Practice evaluate", True,
                    f"Practice eval returned score={score}/{max_score}")
        else:
            _record(state, 4, "Practice evaluate", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 4, "Practice evaluate", False, f"Exception: {exc}")


async def step5_check_usage(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 5: Check gate usage."""
    url = f"{base}/api/v1/evalpen/usage/current"
    try:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            body = resp.json()
            # The response shape depends on CurrentUsage -- extract what we can
            if isinstance(body, dict):
                # Try to summarise daily tokens
                daily = body.get("daily", {})
                tokens_today = daily.get("total_tokens", body.get("total_tokens", "N/A"))
                _record(state, 5, "Gate usage", True,
                        f"Gate usage: {tokens_today} tokens today")
            else:
                _record(state, 5, "Gate usage", True,
                        f"Gate usage response received")
        else:
            _record(state, 5, "Gate usage", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 5, "Gate usage", False, f"Exception: {exc}")


async def step6_list_submissions(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 6: List submissions."""
    url = f"{base}/api/v1/evalpen/submissions"
    try:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            body = resp.json()
            items = body.get("items", [])
            _record(state, 6, "List submissions", True,
                    f"Submissions list: {len(items)} found")
        else:
            _record(state, 6, "List submissions", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 6, "List submissions", False, f"Exception: {exc}")


async def step7_flagged_queue(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 7: Test flagged queue."""
    url = f"{base}/api/v1/evalpen/flagged/queue"
    try:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            body = resp.json()
            items = body.get("items", [])
            total = body.get("total", len(items))
            _record(state, 7, "Flagged queue", True,
                    f"Flagged queue: {total} items")
        else:
            _record(state, 7, "Flagged queue", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 7, "Flagged queue", False, f"Exception: {exc}")


async def step8_review_summary(
    client: Any,
    base: str,
    headers: Dict[str, str],
    state: PipelineState,
) -> None:
    """Step 8: Test review summary (if submission exists)."""
    if not state.submission_id:
        _record(state, 8, "Review summary", False,
                "Skipped: no submission_id from step 3")
        return

    url = f"{base}/api/v1/evalpen/review/submissions/{state.submission_id}/summary"
    try:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            body = resp.json()
            evaluated = body.get("evaluated_count", 0)
            blocked = body.get("blocked_count", 0)
            pending = body.get("pending_count", 0)
            total_score = body.get("total_score", 0)
            total_max = body.get("total_max_score", 0)
            _record(state, 8, "Review summary", True,
                    f"Review summary: score={total_score}/{total_max}, "
                    f"evaluated={evaluated}, blocked={blocked}, pending={pending}")
        elif resp.status_code == 404:
            _record(state, 8, "Review summary", False,
                    f"404: submission or evaluations not found (expected if no eval ran)")
        else:
            _record(state, 8, "Review summary", False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _record(state, 8, "Review summary", False, f"Exception: {exc}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(base_url: str, token: str, exam_id: str, student_id: str) -> int:
    """Run all pipeline steps and return exit code (0 = all pass)."""
    import httpx

    state = PipelineState(
        question_id="q1",
        exam_id=exam_id,
        student_id=student_id,
    )
    headers = _auth_headers(token)

    print("=" * 60)
    print("ExamPen E2E Pipeline Test")
    print(f"  Base URL   : {base_url}")
    print(f"  Exam ID    : {exam_id}")
    print(f"  Student ID : {student_id}")
    print(f"  Question ID: {state.question_id}")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        await step1_register_question(client, base_url, headers, state)
        await step2_upload_solution(client, base_url, headers, state)
        await step3_create_submission(client, base_url, headers, state)
        await step4_practice_evaluate(client, base_url, headers, state)
        await step5_check_usage(client, base_url, headers, state)
        await step6_list_submissions(client, base_url, headers, state)
        await step7_flagged_queue(client, base_url, headers, state)
        await step8_review_summary(client, base_url, headers, state)

    # Summary
    print()
    print("=" * 60)
    passed = sum(1 for r in state.results if r.passed)
    total = len(state.results)
    print(f"Results: {passed}/{total} passed")
    print("=" * 60)

    return 0 if passed == total else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ExamPen E2E Pipeline Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/test_exampen_pipeline.py \\\n"
            "    --base-url http://localhost:5001 \\\n"
            "    --token eyJhbGciOiJIUzI1NiIs..."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:5001",
        help="Backend base URL (default: http://localhost:5001)",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Admin JWT for the test tenant (required)",
    )
    parser.add_argument(
        "--exam-id",
        default="e2e-test-exam-001",
        help="Exam ID to use (default: e2e-test-exam-001)",
    )
    parser.add_argument(
        "--student-id",
        default="e2e-test-student-001",
        help="Student ID to use (default: e2e-test-student-001)",
    )

    args = parser.parse_args()

    exit_code = asyncio.run(
        run_pipeline(
            base_url=args.base_url.rstrip("/"),
            token=args.token,
            exam_id=args.exam_id,
            student_id=args.student_id,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
