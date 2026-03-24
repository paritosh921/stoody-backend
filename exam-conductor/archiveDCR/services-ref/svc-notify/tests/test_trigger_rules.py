"""Unit tests for domain/trigger_rules.py — event-to-notification mapping.

ZERO I/O: these tests exercise pure functions only.
"""

import pytest

from src.domain.trigger_rules import (
    NotificationAction,
    NotificationChannel,
    WebhookAction,
    determine_notifications,
)


# ---------------------------------------------------------------------------
# score.updated
# ---------------------------------------------------------------------------


class TestScoreUpdated:
    """Tests for score.updated event rules."""

    def test_published_creates_email_push_and_webhook(self) -> None:
        payload = {
            "event_type": "score.updated",
            "exam_id": "e-100",
            "student_id": "s-42",
            "total_score": 87,
            "reason": "published",
        }
        actions = determine_notifications("score.updated", payload)

        # 2 personal notifications + 1 Stoody webhook
        assert len(actions) == 3

        personal = [a for a in actions if isinstance(a, NotificationAction)]
        webhooks = [a for a in actions if isinstance(a, WebhookAction)]

        assert len(personal) == 2
        assert len(webhooks) == 1

        channels = {a.channel for a in personal}
        assert channels == {"email", "push"}

        for a in personal:
            assert a.recipient_id == "s-42"
            assert a.template_name == "score_published"
            assert a.template_data["total_score"] == 87

        wh = webhooks[0]
        assert wh.webhook_type == "score"
        assert wh.webhook_data["exam_id"] == "e-100"

    def test_non_published_reason_produces_nothing(self) -> None:
        for reason in ("ai_draft_created", "override_applied", "finalized", "objection_rescored"):
            payload = {
                "event_type": "score.updated",
                "exam_id": "e-100",
                "student_id": "s-42",
                "total_score": 50,
                "reason": reason,
            }
            actions = determine_notifications("score.updated", payload)
            assert actions == [], f"Expected no actions for reason={reason}"

    def test_missing_student_id_produces_nothing(self) -> None:
        payload = {
            "event_type": "score.updated",
            "exam_id": "e-100",
            "student_id": "",
            "total_score": 87,
            "reason": "published",
        }
        assert determine_notifications("score.updated", payload) == []


# ---------------------------------------------------------------------------
# objection
# ---------------------------------------------------------------------------


class TestObjection:
    """Tests for objection event rules."""

    def test_filed_notifies_evaluator(self) -> None:
        payload = {
            "event_type": "objection",
            "exam_id": "e-200",
            "objection_id": "obj-1",
            "student_id": "s-10",
            "question_id": "q-5",
            "action": "filed",
            "state": "filed",
            "actor_id": "evaluator-99",
        }
        actions = determine_notifications("objection", payload)

        assert len(actions) == 1
        a = actions[0]
        assert a.recipient_id == "evaluator-99"
        assert a.channel == "email"
        assert a.template_name == "objection_filed"

    def test_filed_without_actor_id_uses_fallback(self) -> None:
        payload = {
            "event_type": "objection",
            "exam_id": "e-200",
            "objection_id": "obj-2",
            "student_id": "s-10",
            "question_id": "q-5",
            "action": "filed",
            "state": "filed",
        }
        actions = determine_notifications("objection", payload)

        assert len(actions) == 1
        assert actions[0].recipient_id == "evaluator:unassigned"

    def test_resolved_notifies_student_email_and_push(self) -> None:
        payload = {
            "event_type": "objection",
            "exam_id": "e-200",
            "objection_id": "obj-1",
            "student_id": "s-10",
            "question_id": "q-5",
            "action": "resolved",
            "state": "resolved",
            "actor_id": "evaluator-99",
        }
        actions = determine_notifications("objection", payload)

        assert len(actions) == 2
        channels = {a.channel for a in actions}
        assert channels == {"email", "push"}
        for a in actions:
            assert a.recipient_id == "s-10"
            assert a.template_name == "objection_resolved"

    def test_assigned_state_produces_nothing(self) -> None:
        payload = {
            "event_type": "objection",
            "exam_id": "e-200",
            "objection_id": "obj-1",
            "student_id": "s-10",
            "question_id": "q-5",
            "action": "assigned",
            "state": "assigned",
        }
        assert determine_notifications("objection", payload) == []

    def test_reviewing_state_produces_nothing(self) -> None:
        payload = {
            "event_type": "objection",
            "exam_id": "e-200",
            "objection_id": "obj-1",
            "student_id": "s-10",
            "question_id": "q-5",
            "action": "reviewing",
            "state": "reviewing",
        }
        assert determine_notifications("objection", payload) == []


# ---------------------------------------------------------------------------
# exam.lifecycle
# ---------------------------------------------------------------------------


class TestExamLifecycle:
    """Tests for exam.lifecycle event rules."""

    def test_armed_creates_email_and_push(self) -> None:
        payload = {
            "event_type": "exam.lifecycle",
            "exam_id": "e-300",
            "from_state": "scheduled",
            "to_state": "armed",
            "actor_id": "teacher-1",
        }
        actions = determine_notifications("exam.lifecycle", payload)

        assert len(actions) == 2
        channels = {a.channel for a in actions}
        assert channels == {"email", "push"}
        for a in actions:
            assert a.recipient_id == "exam:e-300:students"
            assert a.template_name == "exam_reminder"

    def test_non_armed_non_webhook_state_produces_nothing(self) -> None:
        # "created" and "upload_complete" now produce webhook actions,
        # so they are not included here.
        for to_state in ("scheduled", "in_progress", "completed", "cancelled"):
            payload = {
                "event_type": "exam.lifecycle",
                "exam_id": "e-300",
                "from_state": "draft",
                "to_state": to_state,
                "actor_id": "teacher-1",
            }
            assert determine_notifications("exam.lifecycle", payload) == []

    def test_created_produces_webhook_only(self) -> None:
        payload = {
            "event_type": "exam.lifecycle",
            "exam_id": "e-301",
            "to_state": "created",
            "subject_id": "subj-1",
            "class_id": "cls-1",
            "date": "2026-04-01",
            "duration": 120,
        }
        actions = determine_notifications("exam.lifecycle", payload)
        assert len(actions) == 1
        assert isinstance(actions[0], WebhookAction)
        assert actions[0].webhook_type == "exam_created"

    def test_upload_complete_produces_webhook_only(self) -> None:
        payload = {
            "event_type": "exam.lifecycle",
            "exam_id": "e-302",
            "to_state": "upload_complete",
            "pens_synced": 35,
            "upload_status": "complete",
        }
        actions = determine_notifications("exam.lifecycle", payload)
        assert len(actions) == 1
        assert isinstance(actions[0], WebhookAction)
        assert actions[0].webhook_type == "exam_completed"
        assert actions[0].webhook_data["pens_synced"] == 35


# ---------------------------------------------------------------------------
# Unknown event types
# ---------------------------------------------------------------------------


class TestUnknownEvents:
    """Unknown event types should produce zero actions, not crash."""

    def test_unknown_event_type(self) -> None:
        assert determine_notifications("something.unknown", {"foo": "bar"}) == []

    def test_empty_payload(self) -> None:
        assert determine_notifications("score.updated", {}) == []


# ---------------------------------------------------------------------------
# NotificationAction immutability
# ---------------------------------------------------------------------------


class TestNotificationAction:
    """Verify dataclass properties."""

    def test_frozen(self) -> None:
        a = NotificationAction(
            recipient_id="r-1",
            channel="email",
            template_name="score_published",
            template_data={"k": "v"},
        )
        with pytest.raises(AttributeError):
            a.recipient_id = "changed"  # type: ignore[misc]
