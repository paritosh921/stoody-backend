"""
ExamPen API TUI — Interactive Textual-based test client for all 18 ExamPen routers.

Makes real HTTP calls to the Stoody backend and validates actual API responses.
Not a mock/dummy tester — every endpoint hits the live server.

Usage:
    python exampen_api_tui.py [--base-url http://localhost:5001]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Header, Footer, Static, Label, Input, Button, Select,
        Tree, DataTable, Log, TextArea, Markdown,
        LoadingIndicator, ProgressBar, Checkbox,
    )
    from textual.screen import Screen, ModalScreen
    from textual.binding import Binding
    from textual import work
    from textual.events import Mount
    from textual.message import Message
    from textual.widget import Widget
    from rich.text import Text
    from rich.json import JSON
    from rich.panel import Panel
    from rich.table import Table as RichTable
except ImportError:
    print("Error: textual and httpx are required.")
    print("Install with: pip install textual httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Endpoint definitions — all 18 ExamPen routers
# ---------------------------------------------------------------------------

@dataclass
class EndpointParam:
    name: str
    type: str  # "path", "query", "body", "header"
    required: bool = True
    default: str = ""
    description: str = ""
    options: List[str] = field(default_factory=list)


@dataclass
class EndpointDef:
    id: str
    method: str
    path: str
    summary: str
    category: str
    params: List[EndpointParam] = field(default_factory=list)
    expected_statuses: List[int] = field(default_factory=lambda: [200, 201, 202])
    auth_required: bool = True
    content_type: str = "application/json"


ALL_ENDPOINTS: List[EndpointDef] = [
    # 1. Exam Orchestration
    EndpointDef(
        id="exam_list", method="GET", path="/api/v1/exams",
        summary="List exams", category="Exam Orchestration",
        params=[EndpointParam("lifecycle_filter", "query", required=False, description="Filter by lifecycle state")],
    ),
    EndpointDef(
        id="exam_create", method="POST", path="/api/v1/exams",
        summary="Create exam", category="Exam Orchestration",
        params=[
            EndpointParam("exam_id", "body", required=True, description="Unique exam ID"),
            EndpointParam("exam_type", "body", required=True, description="dcr or pcr", options=["dcr", "pcr"]),
            EndpointParam("roster", "body", required=False, default="[]", description="JSON array of student IDs"),
            EndpointParam("duration_minutes", "body", required=False, description="Exam duration in minutes"),
        ],
        expected_statuses=[201],
    ),
    EndpointDef(
        id="exam_detail", method="GET", path="/api/v1/exams/{exam_id}",
        summary="Get exam detail", category="Exam Orchestration",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),
    EndpointDef(
        id="exam_lifecycle", method="PATCH", path="/api/v1/exams/{exam_id}/lifecycle",
        summary="Transition exam lifecycle", category="Exam Orchestration",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("target_state", "body", required=True, description="Target state",
                          options=["armed", "in_progress", "collection_closed", "uploading", "ready_for_eval"]),
        ],
        expected_statuses=[202, 409],
    ),
    EndpointDef(
        id="exam_assign_hub", method="POST", path="/api/v1/exams/{exam_id}/hubs",
        summary="Assign hub to exam", category="Exam Orchestration",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("hub_id", "body", required=True, description="Hub ID"),
        ],
        expected_statuses=[200, 409],
    ),
    EndpointDef(
        id="exam_unassign_hub", method="DELETE", path="/api/v1/exams/{exam_id}/hubs",
        summary="Unassign hub from exam", category="Exam Orchestration",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("hub_id", "body", required=True, description="Hub ID"),
        ],
    ),
    EndpointDef(
        id="exam_progress", method="GET", path="/api/v1/exams/{exam_id}/progress",
        summary="Get exam upload progress", category="Exam Orchestration",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),

    # 2. Stroke Ingest
    EndpointDef(
        id="stroke_upload", method="POST", path="/api/v1/ingest/strokes/{exam_id}/{pen_mac}",
        summary="Upload stroke chunk", category="Stroke Ingest",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("pen_mac", "path", required=True, description="BLE pen MAC address"),
            EndpointParam("exam_type", "body", required=True, options=["dcr", "pcr"]),
            EndpointParam("student_id", "body", required=True, description="Student ID"),
            EndpointParam("chunk_index", "body", required=True, default="0", description="Chunk index"),
            EndpointParam("total_chunks", "body", required=True, default="1", description="Total chunks"),
            EndpointParam("payload_base64", "body", required=True, default="dGVzdA==", description="Base64 payload"),
        ],
        expected_statuses=[202, 400, 404],
    ),
    EndpointDef(
        id="stroke_complete", method="POST", path="/api/v1/ingest/strokes/{exam_id}/{pen_mac}/complete",
        summary="Finalize pen upload", category="Stroke Ingest",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("pen_mac", "path", required=True, description="BLE pen MAC address"),
            EndpointParam("student_id", "body", required=True, description="Student ID"),
            EndpointParam("expected_checksum", "body", required=True, description="SHA-256 hex checksum"),
            EndpointParam("total_chunks", "body", required=True, default="1", description="Total chunks"),
        ],
        expected_statuses=[200, 400],
    ),
    EndpointDef(
        id="stroke_status", method="GET", path="/api/v1/ingest/strokes/{exam_id}/{pen_mac}/status",
        summary="Get pen upload status", category="Stroke Ingest",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("pen_mac", "path", required=True, description="BLE pen MAC address"),
        ],
    ),
    EndpointDef(
        id="stroke_dedup", method="POST", path="/api/v1/ingest/strokes/{exam_id}/{pen_mac}/dedup",
        summary="Check chunk deduplication", category="Stroke Ingest",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
            EndpointParam("pen_mac", "path", required=True, description="BLE pen MAC address"),
            EndpointParam("chunk_index", "body", required=True, default="0", description="Chunk index"),
            EndpointParam("payload_hash", "body", required=True, description="SHA-256 hash of payload"),
        ],
    ),

    # 3. Camera Upload
    EndpointDef(
        id="camera_upload", method="POST", path="/api/v1/ingest/camera/{exam_id}/copies/upload",
        summary="Upload camera image", category="Camera Upload",
        params=[
            EndpointParam("exam_id", "path", required=True, description="Exam ID"),
        ],
        expected_statuses=[202],
        content_type="multipart/form-data",
    ),

    # 4. DCR Evaluation
    EndpointDef(
        id="dcr_evaluate", method="POST", path="/api/v1/evalpen/dcr/evaluate",
        summary="Evaluate DCR response", category="DCR Evaluation",
        params=[
            EndpointParam("submission_id", "body", required=True, description="Submission ID"),
            EndpointParam("artifact_id", "body", required=True, description="Artifact ID"),
        ],
        expected_statuses=[202],
    ),
    EndpointDef(
        id="dcr_results", method="GET", path="/api/v1/evalpen/dcr/results/{submission_id}",
        summary="Get DCR results", category="DCR Evaluation",
        params=[EndpointParam("submission_id", "path", required=True, description="Submission ID")],
    ),

    # 5. PCR Submissions
    EndpointDef(
        id="submission_create", method="POST", path="/api/v1/evalpen/submissions",
        summary="Create PCR submission", category="PCR Submissions",
        params=[
            EndpointParam("exam_id", "body", required=True, description="Exam ID"),
            EndpointParam("student_id", "body", required=True, description="Student ID"),
            EndpointParam("source", "body", required=True, options=["ble_pen", "camera"]),
            EndpointParam("page_count", "body", required=False, default="1", description="Number of pages"),
        ],
        expected_statuses=[202],
    ),
    EndpointDef(
        id="submission_list", method="GET", path="/api/v1/evalpen/submissions",
        summary="List PCR submissions", category="PCR Submissions",
    ),
    EndpointDef(
        id="submission_responses", method="GET", path="/api/v1/evalpen/submissions/{submission_id}/responses",
        summary="Get submission responses", category="PCR Submissions",
        params=[EndpointParam("submission_id", "path", required=True, description="Submission ID")],
    ),
    EndpointDef(
        id="flag_resolve", method="PATCH", path="/api/v1/evalpen/flags/{flag_id}/resolve",
        summary="Resolve PCR flag", category="PCR Submissions",
        params=[
            EndpointParam("flag_id", "path", required=True, description="Flag ID"),
            EndpointParam("resolution", "body", required=True, description="Resolution text"),
            EndpointParam("note", "body", required=False, description="Optional note"),
        ],
    ),

    # 6. PCR Evaluation
    EndpointDef(
        id="evaluate_single", method="POST", path="/api/v1/evalpen/evaluate",
        summary="Evaluate single PCR response", category="PCR Evaluation",
        params=[
            EndpointParam("response_id", "body", required=True, description="Response ID"),
            EndpointParam("question_id", "body", required=True, description="Question ID"),
        ],
        expected_statuses=[202],
    ),
    EndpointDef(
        id="evaluate_batch", method="POST", path="/api/v1/evalpen/evaluate/batch",
        summary="Batch evaluate PCR responses", category="PCR Evaluation",
        params=[
            EndpointParam("items", "body", required=True, default='[{"response_id":"r1","question_id":"q1"}]',
                          description="JSON array of {response_id, question_id}"),
        ],
        expected_statuses=[202],
    ),

    # 7. PCR Evaluations
    EndpointDef(
        id="evaluation_detail", method="GET", path="/api/v1/evalpen/evaluations/{evaluation_id}",
        summary="Get evaluation result", category="PCR Evaluations",
        params=[EndpointParam("evaluation_id", "path", required=True, description="Evaluation ID")],
    ),

    # 8. PCR Solutions
    EndpointDef(
        id="solution_get", method="GET", path="/api/v1/evalpen/solutions/{question_id}",
        summary="Get solution for question", category="PCR Solutions",
        params=[EndpointParam("question_id", "path", required=True, description="Question ID")],
    ),
    EndpointDef(
        id="solution_upsert", method="PUT", path="/api/v1/evalpen/solutions/{question_id}",
        summary="Upsert solution", category="PCR Solutions",
        params=[
            EndpointParam("question_id", "path", required=True, description="Question ID"),
            EndpointParam("reference_solution", "body", required=True, description="Solution text"),
            EndpointParam("solution_source", "body", required=True, options=["teacher", "llm"]),
            EndpointParam("model_used", "body", required=False, description="LLM model name"),
        ],
    ),

    # 9. PCR Questions
    EndpointDef(
        id="question_register", method="POST", path="/api/v1/evalpen/questions",
        summary="Register question metadata", category="PCR Questions",
        params=[
            EndpointParam("question_id", "body", required=True, description="Question ID"),
            EndpointParam("exam_id", "body", required=True, description="Exam ID"),
            EndpointParam("question_type", "body", required=True, description="Question type"),
            EndpointParam("max_marks", "body", required=True, default="10", description="Max marks"),
            EndpointParam("complexity", "body", required=True, options=["L1", "L2", "L3"]),
            EndpointParam("eval_template", "body", required=True, description="Evaluation template"),
            EndpointParam("expects_diagram", "body", required=False, default="false", description="Expects diagram"),
        ],
        expected_statuses=[202],
    ),

    # 10. PCR Practice
    EndpointDef(
        id="practice_evaluate", method="POST", path="/api/v1/evalpen/practice/evaluate",
        summary="Practice evaluation", category="PCR Practice",
        params=[
            EndpointParam("question_id", "body", required=True, description="Question ID"),
            EndpointParam("source_type", "body", required=True, options=["canvas", "camera"]),
            EndpointParam("text", "body", required=False, description="Student answer text"),
            EndpointParam("image_ref", "body", required=False, description="Image reference URL"),
        ],
        expected_statuses=[200],
    ),

    # 11. Review
    EndpointDef(
        id="review_submission_summary", method="GET",
        path="/api/v1/evalpen/review/submissions/{submission_id}/summary",
        summary="Get submission review summary", category="Review",
        params=[EndpointParam("submission_id", "path", required=True, description="Submission ID")],
    ),
    EndpointDef(
        id="review_exam_results", method="GET", path="/api/v1/evalpen/review/exams/{exam_id}/results",
        summary="Get exam results", category="Review",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),
    EndpointDef(
        id="review_score_override", method="POST",
        path="/api/v1/evalpen/review/evaluations/{evaluation_id}/override",
        summary="Override evaluation score", category="Review",
        params=[
            EndpointParam("evaluation_id", "path", required=True, description="Evaluation ID"),
            EndpointParam("new_score", "body", required=True, default="8.5", description="New score"),
            EndpointParam("reason", "body", required=True, default="Manual adjustment after review",
                          description="Reason (min 5 chars)"),
        ],
    ),
    EndpointDef(
        id="review_publish", method="POST",
        path="/api/v1/evalpen/review/submissions/{submission_id}/publish",
        summary="Publish submission results", category="Review",
        params=[
            EndpointParam("submission_id", "path", required=True, description="Submission ID"),
            EndpointParam("note", "body", required=False, description="Optional publication note"),
        ],
    ),

    # 12. Flagged Queue
    EndpointDef(
        id="flagged_queue", method="GET", path="/api/v1/evalpen/flagged/queue",
        summary="Get flagged response queue", category="Flagged Queue",
        params=[
            EndpointParam("exam_id", "query", required=False, description="Filter by exam ID"),
            EndpointParam("limit", "query", required=False, default="100", description="Page size"),
            EndpointParam("skip", "query", required=False, default="0", description="Offset"),
        ],
    ),
    EndpointDef(
        id="flagged_submission", method="GET", path="/api/v1/evalpen/flagged/queue/{submission_id}",
        summary="Get flagged for submission", category="Flagged Queue",
        params=[EndpointParam("submission_id", "path", required=True, description="Submission ID")],
    ),
    EndpointDef(
        id="flagged_review", method="POST", path="/api/v1/evalpen/flagged/{response_id}/review",
        summary="Review flagged response", category="Flagged Queue",
        params=[
            EndpointParam("response_id", "path", required=True, description="Response ID"),
            EndpointParam("action", "body", required=True, options=["accept", "reject", "manual_score"]),
            EndpointParam("reason", "body", required=True, default="Reviewed and accepted",
                          description="Reason (min 5 chars)"),
            EndpointParam("manual_score", "body", required=False, description="Score (for manual_score action)"),
            EndpointParam("manual_max_score", "body", required=False, description="Max score (for manual_score)"),
        ],
    ),
    EndpointDef(
        id="flagged_stats", method="GET", path="/api/v1/evalpen/flagged/stats",
        summary="Get flag statistics", category="Flagged Queue",
        params=[EndpointParam("exam_id", "query", required=False, description="Filter by exam ID")],
    ),

    # 13. LLM Gate Usage
    EndpointDef(
        id="usage_current", method="GET", path="/api/v1/evalpen/usage/current",
        summary="Get current LLM gate usage", category="LLM Gate Usage",
    ),
    EndpointDef(
        id="usage_history", method="GET", path="/api/v1/evalpen/usage/history",
        summary="Get usage history", category="LLM Gate Usage",
        params=[
            EndpointParam("period_type", "query", required=False, options=["daily", "weekly", "monthly"]),
            EndpointParam("limit", "query", required=False, default="30", description="Number of periods"),
        ],
    ),
    EndpointDef(
        id="usage_config", method="PUT", path="/api/v1/evalpen/usage/config",
        summary="Update gate config", category="LLM Gate Usage",
        params=[
            EndpointParam("daily_token_limit", "body", required=False, description="Daily token limit"),
            EndpointParam("weekly_token_limit", "body", required=False, description="Weekly token limit"),
            EndpointParam("monthly_token_limit", "body", required=False, description="Monthly token limit"),
        ],
    ),

    # 14. Teacher BFF
    EndpointDef(
        id="teacher_exams", method="GET", path="/api/v1/teacher/exams",
        summary="List teacher exams", category="Teacher BFF",
    ),
    EndpointDef(
        id="teacher_queue", method="GET", path="/api/v1/teacher/exams/{exam_id}/queue",
        summary="Get teacher work queue", category="Teacher BFF",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),

    # 15. Student BFF
    EndpointDef(
        id="student_exams", method="GET", path="/api/v1/student/exams",
        summary="List student exams", category="Student BFF",
    ),
    EndpointDef(
        id="student_scores", method="GET", path="/api/v1/student/exams/{exam_id}/scores",
        summary="Get student scores", category="Student BFF",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),

    # 16. Invigilator Console
    EndpointDef(
        id="invig_session", method="GET", path="/api/v1/invig/sessions/{exam_id}",
        summary="Get exam session state", category="Invigilator Console",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),
    EndpointDef(
        id="invig_hubs", method="GET", path="/api/v1/invig/sessions/{exam_id}/hubs",
        summary="Get hub status", category="Invigilator Console",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),
    EndpointDef(
        id="invig_pens", method="GET", path="/api/v1/invig/sessions/{exam_id}/pens",
        summary="Get connected pens", category="Invigilator Console",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),
    EndpointDef(
        id="invig_sync", method="GET", path="/api/v1/invig/sessions/{exam_id}/sync-progress",
        summary="Get sync progress", category="Invigilator Console",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),
    EndpointDef(
        id="invig_alerts", method="GET", path="/api/v1/invig/sessions/{exam_id}/alerts",
        summary="Get session alerts", category="Invigilator Console",
        params=[EndpointParam("exam_id", "path", required=True, description="Exam ID")],
    ),

    # 17. Hub Operations
    EndpointDef(
        id="hub_list", method="GET", path="/api/v1/hubs",
        summary="List hubs", category="Hub Operations",
    ),
    EndpointDef(
        id="hub_register", method="POST", path="/api/v1/hubs",
        summary="Register hub", category="Hub Operations",
        params=[
            EndpointParam("hub_id", "body", required=True, description="Hub ID"),
            EndpointParam("hub_code", "body", required=True, description="Provisioning code"),
            EndpointParam("institute_id", "body", required=True, description="Institute ID"),
        ],
    ),
    EndpointDef(
        id="hub_detail", method="GET", path="/api/v1/hubs/{hub_id}",
        summary="Get hub detail", category="Hub Operations",
        params=[EndpointParam("hub_id", "path", required=True, description="Hub ID")],
    ),

    # 18. Invigilator (code management)
    EndpointDef(
        id="invig_code_generate", method="POST", path="/api/v1/evalpen/invigilator/code/generate",
        summary="Generate invigilator code", category="Invigilator",
        params=[
            EndpointParam("exam_id", "body", required=True, description="Exam ID"),
            EndpointParam("valid_minutes", "body", required=False, default="60", description="Validity in minutes"),
        ],
    ),
    EndpointDef(
        id="invig_code_verify", method="POST", path="/api/v1/evalpen/invigilator/code/verify",
        summary="Verify invigilator code", category="Invigilator",
        params=[
            EndpointParam("code", "body", required=True, description="Invigilator code"),
            EndpointParam("exam_id", "body", required=True, description="Exam ID"),
        ],
    ),
]

CATEGORIES = sorted(set(ep.category for ep in ALL_ENDPOINTS))


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class ExamPenAPIClient:
    """Real HTTP client for ExamPen APIs."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None
        self.user_info: Optional[Dict[str, Any]] = None
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        self.request_log: List[Dict[str, Any]] = []

    async def login(self, user_type: str, username: str, password: str, tenant_id: str) -> Dict[str, Any]:
        """Login via Stoody auth endpoint."""
        if user_type == "admin":
            endpoint = "/api/v1/auth/2fa/login-2fa"
            payload = {
                "username": username,
                "password": password,
                "tenant_id": tenant_id or None,
                "user_type": "admin",
            }
        elif user_type == "tutor":
            endpoint = "/api/v1/auth/2fa/login-2fa"
            payload = {
                "username": username,
                "password": password,
                "tenant_id": tenant_id,
                "user_type": "tutor",
            }
        elif user_type == "student":
            endpoint = "/api/v1/auth/student/login"
            payload = {"username": username, "password": password, "tenant_id": tenant_id}
        else:
            raise ValueError(f"Unknown user type: {user_type}")

        resp = await self.client.post(endpoint, json=payload)
        data = resp.json()

        if resp.status_code in (200, 201) and data.get("success"):
            if user_type in {"admin", "tutor"}:
                next_step = str(data.get("next") or "").strip().upper()
                if next_step and next_step != "DONE":
                    return {
                        "success": False,
                        "status": 428,
                        "detail": {
                            "detail": "2FA required. Complete the 2FA login flow in the Stoody web app.",
                            "next": next_step,
                        },
                    }
                self.token = data.get("access_token")
                self.user_info = data.get("user", {})
            else:
                token_data = data.get("data", {})
                self.token = token_data.get("token") or token_data.get("access_token")
                self.user_info = token_data.get("user", {})
            if self.token:
                self.client.headers["Authorization"] = f"Bearer {self.token}"
            return {"success": True, "token": self.token, "user": self.user_info}
        else:
            return {"success": False, "status": resp.status_code, "detail": data}

    async def call_endpoint(self, endpoint: EndpointDef, path_params: Dict[str, str],
                            query_params: Dict[str, str], body: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a real API call to an endpoint."""
        path = endpoint.path
        for k, v in path_params.items():
            path = path.replace(f"{{{k}}}", v)

        url = f"{self.base_url}{path}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        start_time = time.time()
        try:
            if endpoint.content_type == "multipart/form-data":
                headers.pop("Content-Type", None)
                resp = await self.client.request(
                    endpoint.method, url, headers=headers, params=query_params,
                )
            elif endpoint.method in ("POST", "PUT", "PATCH"):
                resp = await self.client.request(
                    endpoint.method, url, headers=headers, json=body, params=query_params,
                )
            else:
                resp = await self.client.request(
                    endpoint.method, url, headers=headers, params=query_params,
                )

            elapsed = time.time() - start_time
            try:
                response_body = resp.json()
            except Exception:
                response_body = {"_raw": resp.text}

            status_ok = resp.status_code in endpoint.expected_statuses

            log_entry = {
                "id": endpoint.id,
                "method": endpoint.method,
                "url": url,
                "status": resp.status_code,
                "status_ok": status_ok,
                "elapsed_ms": round(elapsed * 1000, 1),
                "response": response_body,
                "headers": dict(resp.headers),
            }
            self.request_log.append(log_entry)

            return log_entry

        except httpx.RequestError as e:
            elapsed = time.time() - start_time
            log_entry = {
                "id": endpoint.id,
                "method": endpoint.method,
                "url": url,
                "status": 0,
                "status_ok": False,
                "elapsed_ms": round(elapsed * 1000, 1),
                "response": {"error": str(e)},
                "headers": {},
            }
            self.request_log.append(log_entry)
            return log_entry

    async def run_all_tests(self, test_params: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Run all endpoints with provided test parameters."""
        results = []
        for ep in ALL_ENDPOINTS:
            params = test_params.get(ep.id, {})
            path_params = {p.name: params.get(p.name, p.default) for p in ep.params if p.type == "path"}
            query_params = {p.name: params.get(p.name, p.default) for p in ep.params if p.type == "query" and params.get(p.name)}
            body = None
            if ep.method in ("POST", "PUT", "PATCH"):
                body = {}
                for p in ep.params:
                    if p.type == "body" and p.name in params:
                        val = params[p.name]
                        if p.name in ("roster", "items"):
                            try:
                                body[p.name] = json.loads(val)
                            except Exception:
                                body[p.name] = val
                        elif p.name in ("chunk_index", "total_chunks", "page_count",
                                        "max_marks", "duration_minutes", "valid_minutes"):
                            try:
                                body[p.name] = int(val)
                            except Exception:
                                body[p.name] = val
                        elif p.name in ("new_score", "manual_score", "manual_max_score",
                                        "daily_token_limit", "weekly_token_limit", "monthly_token_limit"):
                            try:
                                body[p.name] = float(val)
                            except Exception:
                                body[p.name] = val
                        elif p.name == "expects_diagram":
                            body[p.name] = val.lower() == "true"
                        else:
                            body[p.name] = val
                if not body:
                    body = None

            result = await self.call_endpoint(ep, path_params, query_params, body)
            results.append(result)
        return results

    async def close(self):
        await self.client.aclose()


# ---------------------------------------------------------------------------
# Login Screen
# ---------------------------------------------------------------------------

class LoginScreen(ModalScreen[Dict[str, Any]]):
    """Interactive login screen."""

    def compose(self) -> ComposeResult:
        yield Container(
            Label("ExamPen API TUI - Login", id="login-title"),
            Label("Base URL:", id="url-label"),
            Input(placeholder="http://localhost:5001", value="http://localhost:5001", id="base-url"),
            Label("User Type:", id="type-label"),
            Select(
                [("Admin", "admin"), ("Tutor", "tutor"), ("Student", "student")],
                value="admin", id="user-type",
            ),
            Label("Username / Email:", id="user-label"),
            Input(placeholder="admin@skillbot.app", id="username"),
            Label("Password:", id="pass-label"),
            Input(placeholder="password", password=True, id="password"),
            Label("Tenant ID (e.g. indl-1001):", id="tenant-label"),
            Input(placeholder="indl-1001", id="tenant-id"),
            Horizontal(
                Button("Login", variant="primary", id="login-btn"),
                Button("Skip", variant="default", id="skip-btn"),
                id="login-buttons",
            ),
            id="login-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "login-btn":
            self.dismiss({
                "base_url": self.query_one("#base-url", Input).value,
                "user_type": self.query_one("#user-type", Select).value,
                "username": self.query_one("#username", Input).value,
                "password": self.query_one("#password", Input).value,
                "tenant_id": self.query_one("#tenant-id", Input).value,
            })
        elif event.button.id == "skip-btn":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# Endpoint Form Screen
# ---------------------------------------------------------------------------

class EndpointFormScreen(ModalScreen[Dict[str, Any]]):
    """Dynamic form for endpoint parameters."""

    def __init__(self, endpoint: EndpointDef, existing_params: Optional[Dict] = None):
        super().__init__()
        self.endpoint = endpoint
        self.existing_params = existing_params or {}

    def compose(self) -> ComposeResult:
        widgets = [
            Label(f"{self.endpoint.method} {self.endpoint.path}", id="form-endpoint-title"),
            Label(self.endpoint.summary, id="form-endpoint-summary"),
        ]
        for param in self.endpoint.params:
            label_text = f"{param.name}{' *' if param.required else ''}"
            if param.description:
                label_text += f" — {param.description}"
            widgets.append(Label(label_text, classes="param-label"))
            if param.options:
                default_val = self.existing_params.get(param.name, param.options[0] if param.options else "")
                opts = [(o, o) for o in param.options]
                widgets.append(Select(opts, value=default_val, id=f"param-{param.name}"))
            else:
                default_val = self.existing_params.get(param.name, param.default)
                widgets.append(Input(value=default_val, placeholder=param.default or "", id=f"param-{param.name}"))
        widgets.append(
            Horizontal(
                Button("Send Request", variant="primary", id="form-send"),
                Button("Cancel", variant="default", id="form-cancel"),
                id="form-buttons",
            )
        )
        yield Container(*widgets, id="form-container")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "form-send":
            params = {}
            for param in self.endpoint.params:
                widget = self.query_one(f"#param-{param.name}")
                if isinstance(widget, Select):
                    params[param.name] = str(widget.value) if widget.value != Select.BLANK else ""
                elif isinstance(widget, Input):
                    params[param.name] = widget.value
            self.dismiss(params)
        elif event.button.id == "form-cancel":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# Response Viewer Screen
# ---------------------------------------------------------------------------

class ResponseViewerScreen(ModalScreen):
    """Display API response with validation."""

    def __init__(self, result: Dict[str, Any]):
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        status = self.result.get("status", 0)
        status_ok = self.result.get("status_ok", False)
        elapsed = self.result.get("elapsed_ms", 0)
        url = self.result.get("url", "")
        response = self.result.get("response", {})

        status_style = "green" if status_ok else "red"
        status_text = f"[{status}] {'PASS' if status_ok else 'FAIL'}"

        response_json = json.dumps(response, indent=2, default=str)

        yield Container(
            Label(f"Response: {url}", id="resp-url"),
            Label(f"Status: {status_text}  |  Time: {elapsed}ms", id="resp-status"),
            TextArea(response_json, language="json", read_only=True, id="resp-body"),
            Horizontal(
                Button("Copy JSON", variant="default", id="resp-copy"),
                Button("Close", variant="primary", id="resp-close"),
                id="resp-buttons",
            ),
            id="response-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in ("resp-close", "resp-copy"):
            self.dismiss()


# ---------------------------------------------------------------------------
# Batch Test Results Screen
# ---------------------------------------------------------------------------

class BatchResultsScreen(ModalScreen):
    """Display batch test results."""

    def __init__(self, results: List[Dict[str, Any]]):
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        passed = sum(1 for r in self.results if r.get("status_ok"))
        failed = len(self.results) - passed

        table = DataTable()
        table.add_columns("Endpoint", "Method", "Status", "Time", "Result")
        for r in self.results:
            ep_id = r.get("id", "?")
            method = r.get("method", "?")
            status = r.get("status", 0)
            elapsed = r.get("elapsed_ms", 0)
            ok = r.get("status_ok", False)
            table.add_row(ep_id, method, str(status), f"{elapsed}ms", "PASS" if ok else "FAIL")

        yield Container(
            Label(f"Batch Test Results: {passed} passed, {failed} failed out of {len(self.results)}",
                  id="batch-title"),
            table,
            Horizontal(
                Button("Close", variant="primary", id="batch-close"),
                id="batch-buttons",
            ),
            id="batch-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "batch-close":
            self.dismiss()


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------

class ExamPenAPITUI(App):
    """Main ExamPen API testing TUI application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #top-bar {
        height: 3;
        dock: top;
        background: $surface;
        padding: 0 1;
        layout: horizontal;
    }

    #connection-status {
        width: 1fr;
        content-align: left middle;
    }

    #user-info {
        width: auto;
        content-align: right middle;
    }

    #main-area {
        layout: horizontal;
        height: 1fr;
    }

    #sidebar {
        width: 35%;
        min-width: 30;
        border-right: solid $primary;
    }

    #sidebar-title {
        padding: 0 1;
        background: $primary;
        color: $text;
        height: 1;
    }

    #endpoint-tree {
        height: 1fr;
    }

    #detail-panel {
        width: 1fr;
        layout: vertical;
    }

    #detail-header {
        height: 5;
        padding: 0 1;
        background: $surface;
    }

    #endpoint-info {
        height: 1fr;
        padding: 0 1;
    }

    #action-bar {
        height: 3;
        dock: bottom;
        background: $surface;
        padding: 0 1;
        layout: horizontal;
    }

    #action-bar Button {
        margin: 0 1;
    }

    #log-panel {
        height: 12;
        border-top: solid $primary;
    }

    #log-title {
        padding: 0 1;
        background: $primary;
        color: $text;
        height: 1;
    }

    #request-log {
        height: 1fr;
    }

    #login-container {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }

    #login-title {
        text-align: center;
        text-style: bold;
        padding: 1 0;
    }

    #login-buttons {
        height: 3;
        align: center middle;
    }

    #login-buttons Button {
        margin: 0 1;
        width: 15;
    }

    #form-container {
        width: 80%;
        height: 80%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }

    #form-endpoint-title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #form-endpoint-summary {
        padding: 0 0 1 0;
        color: $text-muted;
    }

    .param-label {
        padding: 0 0 0 0;
    }

    #form-buttons {
        height: 3;
        align: center middle;
    }

    #form-buttons Button {
        margin: 0 1;
    }

    #response-container {
        width: 90%;
        height: 80%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }

    #resp-url {
        text-style: bold;
    }

    #resp-status {
        padding: 0 0 1 0;
    }

    #resp-body {
        height: 1fr;
    }

    #resp-buttons {
        height: 3;
        align: center middle;
    }

    #resp-buttons Button {
        margin: 0 1;
    }

    #batch-container {
        width: 90%;
        height: 80%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }

    #batch-title {
        text-style: bold;
        padding: 0 0 1 0;
    }

    #batch-buttons {
        height: 3;
        align: center middle;
    }

    #batch-buttons Button {
        margin: 0 1;
    }

    Tree {
        height: 1fr;
    }

    DataTable {
        height: 1fr;
    }

    Select {
        margin: 0 0 1 0;
    }

    Input {
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "login", "Login"),
        Binding("t", "test_all", "Run All Tests"),
        Binding("enter", "send_request", "Send Request"),
        Binding("r", "refresh_tree", "Refresh"),
    ]

    def __init__(self, base_url: str = "http://localhost:5001"):
        super().__init__()
        self.base_url = base_url
        self.client = ExamPenAPIClient(base_url)
        self.selected_endpoint: Optional[EndpointDef] = None
        self.endpoint_params: Dict[str, Dict] = {}
        self.test_results: List[Dict[str, Any]] = []

    def on_mount(self) -> None:
        asyncio.create_task(self._populate_tree())
        self.query_one("#connection-status", Static).update(
            f"[dim]Base: {self.base_url}[/]  [yellow](not authenticated)[/]"
        )

    def _on_login_result(self, result: Optional[Dict]) -> None:
        if result is None:
            return
        asyncio.create_task(self._do_login(result))

    async def _do_login(self, login_data: Dict) -> None:
        self.base_url = login_data["base_url"]
        self.client = ExamPenAPIClient(self.base_url)
        try:
            result = await self.client.login(
                login_data["user_type"],
                login_data["username"],
                login_data["password"],
                login_data["tenant_id"],
            )
            if result.get("success"):
                user = result.get("user", {})
                db_name = user.get("db_name", "?")
                self.query_one("#connection-status", Static).update(
                    f"[dim]Base: {self.base_url}[/]  [green]Authenticated[/]  [dim]tenant: {db_name}[/]"
                )
                self.query_one("#user-info", Static).update(
                    f"[bold]{login_data['user_type']}[/]"
                )
                self.notify(f"Logged in as {login_data['user_type']}", severity="information")
            else:
                self.notify(f"Login failed: {result.get('detail', 'Unknown error')}", severity="error")
        except Exception as e:
            self.notify(f"Connection error: {e}", severity="error")

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Not connected", id="connection-status"),
            Static("", id="user-info"),
            id="top-bar",
        )
        yield Container(
            Container(
                Label("ExamPen Endpoints", id="sidebar-title"),
                Tree("endpoints", id="endpoint-tree"),
                id="sidebar",
            ),
            Container(
                Container(
                    Label("Select an endpoint to view details", id="endpoint-info"),
                    id="detail-header",
                ),
                id="detail-panel",
            ),
            id="main-area",
        )
        yield Container(
            Label("Request Log", id="log-title"),
            DataTable(id="request-log"),
            id="log-panel",
        )
        yield Container(
            Button("Login [l]", variant="default", id="btn-login"),
            Button("Send [Enter]", variant="primary", id="btn-send"),
            Button("Run All Tests [t]", variant="warning", id="btn-test-all"),
            id="action-bar",
        )
        yield Footer()

    async def _populate_tree(self) -> None:
        tree = self.query_one("#endpoint-tree", Tree)
        tree.root.clear()
        for category in CATEGORIES:
            cat_node = tree.root.add(category, expanded=False)
            for ep in ALL_ENDPOINTS:
                if ep.category == category:
                    method_color = {
                        "GET": "green",
                        "POST": "blue",
                        "PUT": "yellow",
                        "PATCH": "magenta",
                        "DELETE": "red",
                    }.get(ep.method, "white")
                    label = f"[{method_color}]{ep.method}[/] {ep.summary}"
                    cat_node.add(label, data=ep)
        tree.root.expand()
        tree.refresh()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        ep = event.node.data
        if isinstance(ep, EndpointDef):
            self.selected_endpoint = ep
            self._show_endpoint_details(ep)

    def _show_endpoint_details(self, ep: EndpointDef) -> None:
        info = self.query_one("#endpoint-info", Label)
        params_text = ""
        if ep.params:
            params_text = "\nParameters:\n"
            for p in ep.params:
                req = "required" if p.required else "optional"
                opts = f" [{', '.join(p.options)}]" if p.options else ""
                params_text += f"  - {p.name} ({p.type}, {req}){opts}\n"

        info.update(
            f"[bold]{ep.method}[/] {ep.path}\n\n"
            f"{ep.summary}\n"
            f"Category: {ep.category}\n"
            f"Expected status: {', '.join(str(s) for s in ep.expected_statuses)}\n"
            f"Auth required: {'Yes' if ep.auth_required else 'No'}\n"
            f"{params_text}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-login":
            self.push_screen(LoginScreen(), callback=self._on_login_result)
        elif event.button.id == "btn-send":
            if self.selected_endpoint:
                self.push_screen(
                    EndpointFormScreen(self.selected_endpoint, self.endpoint_params.get(self.selected_endpoint.id)),
                    callback=self._on_form_result,
                )
        elif event.button.id == "btn-test-all":
            asyncio.create_task(self._run_all_tests())

    def _on_form_result(self, params: Optional[Dict]) -> None:
        if params and self.selected_endpoint:
            self.endpoint_params[self.selected_endpoint.id] = params
            asyncio.create_task(self._send_request(self.selected_endpoint, params))

    async def _send_request(self, ep: EndpointDef, params: Dict) -> None:
        path_params = {p.name: params.get(p.name, p.default) for p in ep.params if p.type == "path"}
        query_params = {p.name: params.get(p.name, p.default) for p in ep.params if p.type == "query" and params.get(p.name)}
        body = None
        if ep.method in ("POST", "PUT", "PATCH"):
            body = {}
            for p in ep.params:
                if p.type == "body" and p.name in params:
                    val = params[p.name]
                    if p.name in ("roster", "items"):
                        try:
                            body[p.name] = json.loads(val)
                        except Exception:
                            body[p.name] = val
                    elif p.name in ("chunk_index", "total_chunks", "page_count",
                                    "max_marks", "duration_minutes", "valid_minutes"):
                        try:
                            body[p.name] = int(val)
                        except Exception:
                            body[p.name] = val
                    elif p.name in ("new_score", "manual_score", "manual_max_score",
                                    "daily_token_limit", "weekly_token_limit", "monthly_token_limit"):
                        try:
                            body[p.name] = float(val)
                        except Exception:
                            body[p.name] = val
                    elif p.name == "expects_diagram":
                        body[p.name] = val.lower() == "true"
                    else:
                        body[p.name] = val
            if not body:
                body = None

        result = await self.client.call_endpoint(ep, path_params, query_params, body)
        self.test_results.append(result)
        self._add_log_entry(result)
        self.push_screen(ResponseViewerScreen(result))

    def _add_log_entry(self, result: Dict) -> None:
        table = self.query_one("#request-log", DataTable)
        if not table.columns:
            table.add_columns("Endpoint", "Method", "URL", "Status", "Time", "Result")

        status = result.get("status", 0)
        ok = result.get("status_ok", False)
        table.add_row(
            result.get("id", "?"),
            result.get("method", "?"),
            result.get("url", "?"),
            str(status),
            f"{result.get('elapsed_ms', 0)}ms",
            "PASS" if ok else "FAIL",
        )

    async def _run_all_tests(self) -> None:
        self.notify("Running all endpoint tests...", severity="information")
        results = await self.client.run_all_tests(self.endpoint_params)
        self.test_results.extend(results)
        for r in results:
            self._add_log_entry(r)
        self.push_screen(BatchResultsScreen(results))

    def action_login(self) -> None:
        self.push_screen(LoginScreen(), callback=self._on_login_result)

    def action_test_all(self) -> None:
        asyncio.create_task(self._run_all_tests())

    def action_send_request(self) -> None:
        if self.selected_endpoint:
            self.push_screen(
                EndpointFormScreen(self.selected_endpoint, self.endpoint_params.get(self.selected_endpoint.id)),
                callback=self._on_form_result,
            )

    def action_refresh_tree(self) -> None:
        asyncio.create_task(self._populate_tree())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base_url = "http://localhost:5001"
    for i, arg in enumerate(sys.argv):
        if arg in ("--base-url", "-u") and i + 1 < len(sys.argv):
            base_url = sys.argv[i + 1]

    app = ExamPenAPITUI(base_url=base_url)
    app.run()


if __name__ == "__main__":
    main()
