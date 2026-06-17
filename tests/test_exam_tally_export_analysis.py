from api.v1.exam_tally_async import TallyDocumentContext, _build_analysis_rows


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

    summary_rows, topic_rows, class_topic_rows, question_rows = _build_analysis_rows(
        rows,
        ["Selected Student", "Selected Student ID", "Q1", "Q2"],
        [],
        document,
    )

    assert len(summary_rows) == 2
    assert summary_rows[0]["Total Obtained"] == 3
    assert summary_rows[0]["Total Max"] == 4
    assert summary_rows[0]["Percentage"] == "75%"
    assert {row["Sub-topic"] for row in topic_rows} == {"Overall"}
    assert len(question_rows) == 4
    assert class_topic_rows == [
        {
            "Class": "10",
            "Section": "A",
            "Subject": "Physics",
            "Sub-topic": "Overall",
            "Students": 2,
            "Marks Obtained": 5.0,
            "Max Marks": 8.0,
            "Average Marks": 2.5,
            "Average Max Marks": 4.0,
            "Percentage": "62.5%",
            "Question Attempts": 4,
            "Class Status": "Developing",
        }
    ]
