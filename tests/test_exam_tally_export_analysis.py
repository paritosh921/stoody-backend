import asyncio
import io
import zipfile

import pytest
from fastapi import HTTPException

from api.v1.exam_tally_async import (
    TallyDocumentContext,
    TallyExportRequest,
    TallyMarkCorrectionSaveRequest,
    TallyQuestionEvidence,
    TallyStudentContext,
    TallyTargetedRecheck,
    _apply_tally_mark_corrections,
    _apply_tally_auto_resolutions,
    _build_analysis_rows,
    _focused_rechecks_to_auto_resolutions,
    _normalise_uncertain_question_cells,
    _strict_ocr_mark_value,
    _tally_question_crop_boxes,
    _validate_tally_mark_correction,
    _validate_tally_result,
    export_tally,
    get_tally_mark_corrections,
    save_tally_mark_correction,
)
from services.tally_question_map_service import build_tally_question_map


def test_exam_tally_analysis_falls_back_to_overall_without_question_map():
    document = TallyDocumentContext(
        standard="10",
        section="A",
        subject="Physics",
        num_questions=2,
        max_marks_per_question=2,
    )
    rows = [
        {"Selected Student": "Aadit", "Selected Student ID": "aadit2403", "Q1": 1, "Q2": 2},
        {"Selected Student": "Aarti", "Selected Student ID": "STU_Aarti_45245", "Q1": 0, "Q2": 2},
    ]

    (
        summary_rows,
        topic_rows,
        subtopic_rows,
        class_topic_rows,
        class_subtopic_rows,
        question_rows,
        intervention_rows,
    ) = _build_analysis_rows(
        rows,
        ["Selected Student", "Selected Student ID", "Q1", "Q2"],
        [],
        document,
    )

    assert len(summary_rows) == 2
    assert summary_rows[0]["Total Obtained"] == 3
    assert summary_rows[0]["Total Max"] == 4
    assert summary_rows[0]["Percentage"] == 75.0
    assert {row["Topic"] for row in topic_rows} == {"Overall"}
    assert {row["Sub-topic"] for row in subtopic_rows} == {"Overall"}
    assert len(question_rows) == 4
    assert intervention_rows == [
        {
            "Student": "Aarti",
            "Student ID": "STU_Aarti_45245",
            "Class": "10",
            "Section": "A",
            "Subject": "Physics",
            "Topic": "Overall",
            "Sub-topic": "Overall",
            "Percentage": 50.0,
            "Priority": "Medium",
            "Suggested Action": "Re-teach and assign targeted practice",
        }
    ]
    assert class_topic_rows == [
        {
            "Class": "10",
            "Section": "A",
            "Subject": "Physics",
            "Topic": "Overall",
            "Students": 2,
            "Marks Obtained": 5.0,
            "Max Marks": 8.0,
            "Average Marks": 2.5,
            "Average Max Marks": 4.0,
            "Percentage": 62.5,
            "Question Count": 2,
            "Scored Opportunities": 4,
            "Class Status": "Developing",
        }
    ]
    assert class_subtopic_rows[0]["Topic"] == "Overall"
    assert class_subtopic_rows[0]["Sub-topic"] == "Overall"


def test_exam_tally_analysis_keeps_topic_and_subtopic_separate():
    document = TallyDocumentContext(subject="Physics", num_questions=2, max_marks_per_question=2)
    rows = [{"Selected Student": "Aadit", "Q1": 1, "Q2": 2}]
    question_map = [
        {"question_number": 1, "topic": "Mechanics", "sub_topic": "Newton's Laws", "max_marks": 2},
        {"question_number": 2, "topic": "Mechanics", "sub_topic": "Kinematics", "max_marks": 2},
    ]

    (
        summary_rows,
        topic_rows,
        subtopic_rows,
        class_topic_rows,
        class_subtopic_rows,
        question_rows,
        intervention_rows,
    ) = _build_analysis_rows(
        rows,
        ["Selected Student", "Q1", "Q2"],
        question_map,
        document,
    )

    assert summary_rows[0]["Weak Topic"] == ""
    assert topic_rows == [
        {
            "Student": "Aadit",
            "Student ID": "",
            "Class": "",
            "Section": "",
            "Subject": "Physics",
            "Topic": "Mechanics",
            "Marks Obtained": 3.0,
            "Max Marks": 4.0,
            "Percentage": 75.0,
            "Question Count": 2,
        }
    ]
    assert {row["Sub-topic"] for row in subtopic_rows} == {"Newton's Laws", "Kinematics"}
    assert class_topic_rows[0]["Topic"] == "Mechanics"
    assert {row["Sub-topic"] for row in class_subtopic_rows} == {"Newton's Laws", "Kinematics"}
    assert {row["Topic"] for row in question_rows} == {"Mechanics"}
    assert len(intervention_rows) == 1
    assert intervention_rows[0]["Sub-topic"] == "Newton's Laws"


def test_exam_tally_export_creates_teacher_report_sheets():
    class FakeCollection:
        async def find_one(self, _query):
            return None

    class FakeTenantDatabase:
        def __getitem__(self, _collection_name):
            return FakeCollection()

    class FakeDatabase:
        async def get_tenant_db(self, _db_name):
            return FakeTenantDatabase()

    async def create_export() -> bytes:
        response = await export_tally(
            TallyExportRequest(
                columns=["Selected Student", "Q1", "Q2"],
                rows=[{"Selected Student": "Aadit", "Q1": 1, "Q2": 2}],
                document=TallyDocumentContext(
                    document_id="exam-1",
                    subject="Physics",
                    num_questions=2,
                    max_marks_per_question=2,
                    question_map=[
                        {"question_number": 1, "topic": "Mechanics", "sub_topic": "Newton's Laws"},
                        {"question_number": 2, "topic": "Mechanics", "sub_topic": "Kinematics"},
                    ],
                ),
                corrections=[
                    {
                        "question_number": 1,
                        "column": "Q1",
                        "original_ocr_value": "",
                        "resolved_value": "1",
                        "resolution_source": "focused_ocr",
                        "reason": "Focused crop contains one exact digit.",
                    },
                    {
                        "question_number": 2,
                        "column": "Q2",
                        "original_ocr_value": "1",
                        "approved_value": "2",
                        "decision": "set",
                        "resolution_source": "teacher_override",
                        "reason": "Teacher corrected the mark.",
                    },
                ],
            ),
            current_user={"db_name": "test", "user_type": "tutor"},
            db=FakeDatabase(),
        )
        return b"".join([chunk async for chunk in response.body_iterator])

    workbook = zipfile.ZipFile(io.BytesIO(asyncio.run(create_export())))
    workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
    for sheet_name in (
        "Overview",
        "Student Summary",
        "Class Topic Analysis",
        "Class Sub-topic Analysis",
        "Intervention Plan",
        "Question Analysis",
        "Question Map",
        "OCR Corrections",
    ):
        assert f'name="{sheet_name}"' in workbook_xml
    shared_strings = workbook.read("xl/sharedStrings.xml").decode("utf-8")
    assert "Focused OCR" in shared_strings
    assert "Teacher Override" in shared_strings


def test_reviewed_topic_map_is_preserved_when_final_ocr_finishes(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    map_doc = asyncio.run(
        build_tally_question_map(
            tally_document_id="exam-1",
            source_document_id="exam-1",
            questions=[
                {
                    "id": "q1",
                    "question_number": 1,
                    "text": "State Newton's first law of motion.",
                    "points": 2,
                }
            ],
            reviewed_items=[
                {
                    "question_number": 1,
                    "topic": "Mechanics",
                    "sub_topic": "Newton's Laws",
                    "source": "reviewed",
                }
            ],
        )
    )

    assert map_doc["items"][0]["topic"] == "Mechanics"
    assert map_doc["items"][0]["sub_topic"] == "Newton's Laws"
    assert map_doc["items"][0]["source"] == "reviewed"


def test_tally_correction_keeps_raw_ocr_and_only_applies_to_matching_cell_evidence():
    raw_rows = [{"Q18": "2", "Q20": "2"}]
    correction = {
        "row_index": 0,
        "question_number": 18,
        "column": "Q18",
        "original_ocr_value": "2",
        "approved_value": "1",
        "decision": "set",
        "crop_hash": "same-cell-hash",
        "copy_id": "tally-exam-aadit",
    }
    evidence = [
        TallyQuestionEvidence(
            row_index=0,
            question_number=18,
            column="Q18",
            crop_hash="same-cell-hash",
        ),
        TallyQuestionEvidence(
            row_index=0,
            question_number=20,
            column="Q20",
            crop_hash="other-cell-hash",
        ),
    ]

    _, effective_rows, applied, stale = _apply_tally_mark_corrections(
        columns=["Q18", "Q20"],
        raw_rows=raw_rows,
        corrections=[correction],
        question_evidence=evidence,
        copy_id="tally-exam-aadit",
    )

    assert raw_rows == [{"Q18": "2", "Q20": "2"}]
    assert effective_rows == [{"Q18": "1", "Q20": "2"}]
    assert len(applied) == 1
    assert stale == []

    changed_evidence = [
        TallyQuestionEvidence(
            row_index=0,
            question_number=18,
            column="Q18",
            crop_hash="changed-cell-hash",
        )
    ]
    _, effective_rows, applied, stale = _apply_tally_mark_corrections(
        columns=["Q18", "Q20"],
        raw_rows=raw_rows,
        corrections=[correction],
        question_evidence=changed_evidence,
        copy_id="tally-exam-aadit",
    )

    assert effective_rows == raw_rows
    assert applied == []
    assert len(stale) == 1


def test_targeted_recheck_parser_and_correction_validation_do_not_guess_or_clamp():
    assert _strict_ocr_mark_value("1") == "1"
    assert _strict_ocr_mark_value("1 or 2") is None
    assert _strict_ocr_mark_value("Q18: 1") is None

    document = TallyDocumentContext(
        num_questions=20,
        marking_scheme=[{"from": 1, "to": 20, "marks": 1}],
    )
    valid_request = TallyMarkCorrectionSaveRequest(
        source_extraction_id="source-1",
        row_index=0,
        question_number=18,
        crop_hash="1234567890abcdef",
        decision="set",
        approved_value=1,
    )
    assert _validate_tally_mark_correction(request=valid_request, document=document) == "1"

    above_max_request = valid_request.model_copy(update={"approved_value": 2})
    with pytest.raises(HTTPException, match="exceeds the configured maximum"):
        _validate_tally_mark_correction(request=above_max_request, document=document)


def test_focused_readable_candidate_auto_resolves_any_configured_question_without_changing_raw_ocr():
    question_number = 7
    question_column = f"Q{question_number}"
    document = TallyDocumentContext(
        num_questions=question_number,
        marking_scheme=[{"from": 1, "to": question_number, "marks": 3}],
    )
    raw_rows = [{question_column: ""}]
    evidence = [
        TallyQuestionEvidence(
            row_index=0,
            question_number=question_number,
            column=question_column,
            crop_hash="focused-cell-evidence",
        )
    ]
    rechecks = [
        TallyTargetedRecheck(
            id="dynamic-question",
            row_index=0,
            question_number=question_number,
            column=question_column,
            original_value="",
            candidate_value="2",
            confidence=0.37,
            status="resolved",
            reason="The focused crop contains one exact digit.",
            crop_hash="focused-cell-evidence",
        )
    ]

    resolutions = _focused_rechecks_to_auto_resolutions(
        rechecks,
        document=document,
        copy_id="copy-a",
    )
    _, effective_rows, applied = _apply_tally_auto_resolutions(
        columns=[question_column],
        raw_rows=raw_rows,
        resolutions=resolutions,
        document=document,
        question_evidence=evidence,
        copy_id="copy-a",
    )

    assert raw_rows == [{question_column: ""}]
    assert effective_rows == [{question_column: "2"}]
    assert len(applied) == 1
    assert applied[0].question_number == question_number
    assert applied[0].resolved_value == "2"
    assert applied[0].resolution_source == "focused_ocr"

    issues = _validate_tally_result(
        [question_column],
        effective_rows,
        document,
        TallyStudentContext(),
    )
    assert not any(issue.question_number == question_number for issue in issues)


def test_focused_candidate_never_clamps_or_resolves_when_it_exceeds_the_real_maximum():
    question_number = 7
    question_column = f"Q{question_number}"
    document = TallyDocumentContext(
        num_questions=question_number,
        marking_scheme=[{"from": 1, "to": question_number, "marks": 3}],
    )
    raw_rows = [{question_column: ""}]
    rechecks = [
        TallyTargetedRecheck(
            id="above-max",
            row_index=0,
            question_number=question_number,
            column=question_column,
            original_value="",
            candidate_value="4",
            status="resolved",
            crop_hash="focused-cell-evidence",
        )
    ]
    resolutions = _focused_rechecks_to_auto_resolutions(
        rechecks,
        document=document,
        copy_id="copy-a",
    )
    _, effective_rows, applied = _apply_tally_auto_resolutions(
        columns=[question_column],
        raw_rows=raw_rows,
        resolutions=resolutions,
        document=document,
        question_evidence=[
            TallyQuestionEvidence(
                row_index=0,
                question_number=question_number,
                column=question_column,
                crop_hash="focused-cell-evidence",
            )
        ],
        copy_id="copy-a",
    )

    assert resolutions == []
    assert effective_rows == raw_rows
    assert applied == []


def test_teacher_override_wins_over_focused_ocr_resolution():
    question_number = 7
    question_column = f"Q{question_number}"
    document = TallyDocumentContext(
        num_questions=question_number,
        marking_scheme=[{"from": 1, "to": question_number, "marks": 3}],
    )
    evidence = [
        TallyQuestionEvidence(
            row_index=0,
            question_number=question_number,
            column=question_column,
            crop_hash="focused-cell-evidence",
        )
    ]
    auto_records = _focused_rechecks_to_auto_resolutions(
        [
            TallyTargetedRecheck(
                id="focused",
                row_index=0,
                question_number=question_number,
                column=question_column,
                candidate_value="1",
                status="resolved",
                crop_hash="focused-cell-evidence",
            )
        ],
        document=document,
        copy_id="copy-a",
    )
    auto_columns, auto_rows, auto_applied = _apply_tally_auto_resolutions(
        columns=[question_column],
        raw_rows=[{question_column: ""}],
        resolutions=auto_records,
        document=document,
        question_evidence=evidence,
        copy_id="copy-a",
    )
    _, effective_rows, teacher_applied, _ = _apply_tally_mark_corrections(
        columns=auto_columns,
        raw_rows=auto_rows,
        corrections=[
            {
                "row_index": 0,
                "question_number": question_number,
                "column": question_column,
                "approved_value": "2",
                "decision": "set",
                "crop_hash": "focused-cell-evidence",
                "copy_id": "copy-a",
            }
        ],
        question_evidence=evidence,
        copy_id="copy-a",
    )

    assert auto_applied[0].resolved_value == "1"
    assert teacher_applied[0].approved_value == "2"
    assert effective_rows == [{question_column: "2"}]


def test_question_cell_crops_do_not_bleed_into_neighbouring_question_columns():
    grid = {
        "horizontal": [0, 10, 20, 30, 40, 50, 60, 70, 80],
        "vertical": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }
    q18 = _tally_question_crop_boxes(grid, 18, width=100, height=100)
    q19 = _tally_question_crop_boxes(grid, 19, width=100, height=100)

    assert q18 is not None
    assert q19 is not None
    q18_mark, _ = q18
    q19_mark, _ = q19
    assert q18_mark["right"] < q19_mark["left"]


def test_ambiguous_question_marks_are_blocked_instead_of_becoming_scores():
    document = TallyDocumentContext(
        num_questions=2,
        marking_scheme=[{"from": 1, "to": 2, "marks": 1}],
    )
    issues = _validate_tally_result(
        ["Q1", "Q2"],
        [{"Q1": "1 or 2", "Q2": "1"}],
        document,
        TallyStudentContext(),
    )

    unreadable = [issue for issue in issues if issue.code == "mark_unreadable"]
    assert len(unreadable) == 1
    assert unreadable[0].question_number == 1
    assert unreadable[0].severity == "error"

    uncertain = _normalise_uncertain_question_cells(
        {"uncertain_cells": ["Q1"], "cell_confidence": {"Q2": 0.4}}
    )
    assert uncertain == {0: [1]}


def test_confirmed_correction_is_persisted_and_returns_effective_rows():
    class MemoryCollection:
        def __init__(self, docs=None):
            self.docs = {doc["_id"]: dict(doc) for doc in (docs or [])}

        async def find_one(self, query):
            for document in self.docs.values():
                if all(document.get(key) == value for key, value in query.items()):
                    return document
            return None

        async def update_one(self, query, update, upsert=False):
            key = query["_id"]
            created = key not in self.docs
            if created and not upsert:
                return None
            document = dict(self.docs.get(key) or {"_id": key})
            if created:
                document.update(update.get("$setOnInsert") or {})
            for field, value in (update.get("$set") or {}).items():
                if "." in field:
                    parent, child = field.split(".", 1)
                    document.setdefault(parent, {})[child] = value
                else:
                    document[field] = value
            for field, value in (update.get("$inc") or {}).items():
                document[field] = int(document.get(field) or 0) + int(value)
            for field, value in (update.get("$push") or {}).items():
                document.setdefault(field, []).append(value)
            self.docs[key] = document
            return None

    class MemoryTenantDatabase:
        def __init__(self, extraction):
            self.collections = {
                "exam_tally_extractions": MemoryCollection([extraction]),
                "exam_tally_mark_reviews": MemoryCollection(),
                "documents": MemoryCollection(
                    [{"_id": "stored-exam-1", "document_id": "exam-1", "teacher_ids": ["teacher-1"]}]
                ),
                "students": MemoryCollection(
                    [{"_id": "stored-student-1", "student_id": "student-1", "teacher_ids": ["teacher-1"]}]
                ),
                "tutors": MemoryCollection(),
            }

        def __getitem__(self, name):
            return self.collections[name]

    class MemoryDatabase:
        def __init__(self, tenant):
            self.tenant = tenant

        async def get_tenant_db(self, _db_name):
            return self.tenant

    document = TallyDocumentContext(
        document_id="exam-1",
        num_questions=20,
        marking_scheme=[{"from": 1, "to": 20, "marks": 1}],
    )
    extraction = {
        "_id": "source-1",
        "document": document.model_dump(exclude_none=True, by_alias=True),
        "student": {"student_id": "student-1"},
        "copy_id": "tally-exam-1-student-1",
        "columns": ["Q17", "Q18"],
        "rows": [{"Q17": "", "Q18": "2"}],
        "question_evidence": [
            {
                "row_index": 0,
                "question_number": 17,
                "column": "Q17",
                "crop_hash": "auto-resolution-evidence",
                "crop_box": {"left": 1, "top": 1, "right": 2, "bottom": 2},
            },
            {
                "row_index": 0,
                "question_number": 18,
                "column": "Q18",
                "crop_hash": "1234567890abcdef",
                "crop_box": {"left": 1, "top": 1, "right": 2, "bottom": 2},
            }
        ],
        "targeted_rechecks": [
            {
                "id": "target-18",
                "row_index": 0,
                "question_number": 18,
                "candidate_value": "1",
            }
        ],
        "auto_resolved_marks": [
            {
                "row_index": 0,
                "question_number": 17,
                "column": "Q17",
                "original_ocr_value": "",
                "resolved_value": "1",
                "crop_hash": "auto-resolution-evidence",
                "evidence_scope": "cell",
                "reason": "Focused OCR resolved one exact visible mark.",
                "resolution_source": "focused_ocr",
            }
        ],
    }
    tenant = MemoryTenantDatabase(extraction)

    response = asyncio.run(
        save_tally_mark_correction(
            document_id="exam-1",
            student_id="student-1",
            payload=TallyMarkCorrectionSaveRequest(
                source_extraction_id="source-1",
                row_index=0,
                question_number=18,
                crop_hash="1234567890abcdef",
                decision="set",
                approved_value=1,
            ),
            current_user={"db_name": "test", "user_id": "teacher-1", "user_type": "tutor"},
            db=MemoryDatabase(tenant),
        )
    )

    assert response.rows == [{"Q17": "1", "Q18": "1"}]
    assert not any(issue.code == "mark_above_max" for issue in response.validation_issues)
    assert response.corrections[0].original_ocr_value == "2"
    assert response.corrections[0].approved_value == "1"
    assert tenant.collections["exam_tally_extractions"].docs["source-1"]["rows"] == [{"Q17": "", "Q18": "2"}]
    assert tenant.collections["exam_tally_extractions"].docs["source-1"]["effective_rows"] == [{"Q17": "1", "Q18": "1"}]
    assert response.auto_resolved_marks[0].question_number == 17

    with pytest.raises(HTTPException, match="not assigned"):
        asyncio.run(
            get_tally_mark_corrections(
                document_id="exam-1",
                student_id="student-1",
                current_user={"db_name": "test", "user_id": "teacher-2", "user_type": "tutor"},
                db=MemoryDatabase(tenant),
            )
        )
