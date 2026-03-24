"""Unit tests for domain/templates.py — template rendering.

ZERO I/O: these tests exercise pure functions only.
"""

import pytest

from src.domain.templates import RenderedNotification, render_template


# ---------------------------------------------------------------------------
# score_published
# ---------------------------------------------------------------------------


class TestScorePublished:
    """Tests for score_published template."""

    def test_renders_with_full_data(self) -> None:
        data = {"exam_id": "e-100", "student_id": "s-42", "total_score": 87}
        result = render_template("score_published", data)

        assert isinstance(result, RenderedNotification)
        assert "published" in result.subject.lower()
        assert "e-100" in result.body_text
        assert "87" in result.body_text
        assert "<strong>87</strong>" in result.body_html

    def test_renders_with_missing_fields(self) -> None:
        result = render_template("score_published", {})
        assert isinstance(result, RenderedNotification)
        assert "unknown" in result.body_text
        assert "N/A" in result.body_text


# ---------------------------------------------------------------------------
# objection_filed
# ---------------------------------------------------------------------------


class TestObjectionFiled:
    """Tests for objection_filed template."""

    def test_renders_with_full_data(self) -> None:
        data = {
            "exam_id": "e-200",
            "objection_id": "obj-1",
            "student_id": "s-10",
            "question_id": "q-5",
        }
        result = render_template("objection_filed", data)

        assert "objection" in result.subject.lower()
        assert "obj-1" in result.body_text
        assert "q-5" in result.body_text
        assert "e-200" in result.body_html

    def test_renders_with_empty_data(self) -> None:
        result = render_template("objection_filed", {})
        assert "unknown" in result.body_text


# ---------------------------------------------------------------------------
# objection_resolved
# ---------------------------------------------------------------------------


class TestObjectionResolved:
    """Tests for objection_resolved template."""

    def test_renders_with_full_data(self) -> None:
        data = {
            "exam_id": "e-200",
            "objection_id": "obj-1",
            "student_id": "s-10",
            "question_id": "q-5",
        }
        result = render_template("objection_resolved", data)

        assert "resolved" in result.subject.lower()
        assert "obj-1" in result.body_text
        assert "q-5" in result.body_html
        assert "student portal" in result.body_text.lower()

    def test_renders_with_empty_data(self) -> None:
        result = render_template("objection_resolved", {})
        assert "unknown" in result.body_text


# ---------------------------------------------------------------------------
# exam_reminder
# ---------------------------------------------------------------------------


class TestExamReminder:
    """Tests for exam_reminder template."""

    def test_renders_with_full_data(self) -> None:
        data = {"exam_id": "e-300", "from_state": "scheduled", "to_state": "armed"}
        result = render_template("exam_reminder", data)

        assert "reminder" in result.subject.lower()
        assert "e-300" in result.body_text
        assert "armed" in result.body_text
        assert "<strong>e-300</strong>" in result.body_html

    def test_renders_with_empty_data(self) -> None:
        result = render_template("exam_reminder", {})
        assert "unknown" in result.body_text


# ---------------------------------------------------------------------------
# Unknown templates
# ---------------------------------------------------------------------------


class TestUnknownTemplate:
    """Unknown template names should raise KeyError."""

    def test_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            render_template("nonexistent", {})


# ---------------------------------------------------------------------------
# RenderedNotification
# ---------------------------------------------------------------------------


class TestRenderedNotification:
    """Verify RenderedNotification properties."""

    def test_frozen(self) -> None:
        r = RenderedNotification(subject="s", body_text="t", body_html="<p>h</p>")
        with pytest.raises(AttributeError):
            r.subject = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        r = RenderedNotification(subject="Sub", body_text="Text", body_html="<p>Html</p>")
        assert r.subject == "Sub"
        assert r.body_text == "Text"
        assert r.body_html == "<p>Html</p>"
