#!/usr/bin/env python3
"""ExamPen seed data generator.

Generates realistic test data for development and testing:
  - Students, tutors, exams with rubrics
  - Binary stroke data (P05 14-byte coordinate frames)
  - AI recognition results, scores, objections
  - Plagiarism flags, chat messages
  - Idempotent PostgreSQL seed SQL

Usage:
  python scripts/seed_data.py --students 40 --exams 3 --questions-per-exam 10

Reference: TEST_SUITE_SPEC.md §5, CLAUDE.md §Pen & BLE Protocol
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Ensure the scripts/ directory is on sys.path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed.entities import gen_exams, gen_students, gen_tutors
from seed.results import (
    gen_ai_results,
    gen_chat_messages,
    gen_objections,
    gen_plagiarism_flags,
    gen_scores,
)
from seed.sql_gen import generate_sql
from seed.strokes import gen_stroke_data


def main() -> None:
    """Entry point for seed data generation."""
    parser = argparse.ArgumentParser(description="ExamPen seed data generator")
    parser.add_argument("--students", type=int, default=40)
    parser.add_argument("--exams", type=int, default=3)
    parser.add_argument("--questions-per-exam", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="test-suite/fixtures")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)

    print(f"Generating seed data (seed={args.seed})...")

    # --- Generate entities ---
    students = gen_students(rng, args.students)
    tutors = gen_tutors(rng)
    exams = gen_exams(rng, args.exams, args.questions_per_exam, tutors)

    print(f"  Students: {len(students)}")
    print(f"  Tutors:   {len(tutors)}")
    print(f"  Exams:    {len(exams)}")

    # --- Stroke data (binary fixtures) ---
    stroke_dir = output / "strokes"
    stroke_dir.mkdir(parents=True, exist_ok=True)
    stroke_count = 0
    for exam in exams:
        for student in students:
            for q in exam["questions"]:
                data = gen_stroke_data(rng, student, exam, q["question_number"])
                fname = (
                    f"pen_{student['stoody_user_id']}"
                    f"_exam_{exam['id'][:8]}_q{q['question_number']:02d}.bin"
                )
                (stroke_dir / fname).write_bytes(data)
                stroke_count += 1
    print(f"  Stroke files: {stroke_count}")

    # --- AI results, scores, objections, plagiarism, chat ---
    all_ai, all_scores = _gen_pipeline_data(rng, students, exams)
    objections = gen_objections(rng, all_scores)
    plag_flags = _gen_plagiarism(rng, students, exams)
    chat_msgs = _gen_chat(rng, students, tutors, exams)

    print(f"  AI results: {len(all_ai)}")
    print(f"  Scores: {len(all_scores)}")
    print(f"  Objections: {len(objections)}")
    print(f"  Plagiarism flags: {len(plag_flags)}")
    print(f"  Chat messages: {len(chat_msgs)}")

    # --- Write all outputs ---
    _write_json_fixtures(output, exams, students, tutors, all_ai,
                         all_scores, objections, plag_flags, chat_msgs)
    _write_ble_fixture(output)
    _write_pages_placeholder(output)

    sql = generate_sql(students, tutors, exams, all_scores, all_ai,
                       objections, plag_flags, chat_msgs)
    (output / "seed.sql").write_text(sql, encoding="utf-8")
    print(f"  SQL: {output / 'seed.sql'}")

    print("\nSeed data generation complete.")


def _gen_pipeline_data(
    rng: random.Random,
    students: list[dict],
    exams: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Generate AI results and scores for all student-exam pairs."""
    all_ai: list[dict] = []
    all_scores: list[dict] = []
    for exam in exams:
        for si, student in enumerate(students):
            results = gen_ai_results(rng, student, exam)
            all_ai.extend(results)
            student_results = [
                r for r in results
                if r["student_id"] == student["id"]
            ]
            scores = gen_scores(rng, student_results, exam, si)
            all_scores.extend(scores)
    return all_ai, all_scores


def _gen_plagiarism(
    rng: random.Random,
    students: list[dict],
    exams: list[dict],
) -> list[dict]:
    """Generate plagiarism flags for all exams (no cap/sampling)."""
    flags: list[dict] = []
    for exam in exams:
        flags.extend(gen_plagiarism_flags(rng, students, exam))
    return flags


def _gen_chat(
    rng: random.Random,
    students: list[dict],
    tutors: list[dict],
    exams: list[dict],
) -> list[dict]:
    msgs: list[dict] = []
    for exam in exams:
        msgs.extend(gen_chat_messages(rng, students, tutors, exam))
    return msgs


def _write_json_fixtures(
    output: Path,
    exams: list[dict],
    students: list[dict],
    tutors: list[dict],
    all_ai: list[dict],
    all_scores: list[dict],
    objections: list[dict],
    plag_flags: list[dict],
    chat_msgs: list[dict],
) -> None:
    """Write JSON fixture files."""
    exams_dir = output / "exams"
    exams_dir.mkdir(parents=True, exist_ok=True)
    for i, exam in enumerate(exams):
        slug = exam["subject"].lower().replace(" ", "_")
        fname = f"exam_{slug}_{i+1:02d}.json"
        (exams_dir / fname).write_text(
            json.dumps(exam, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    plag_dir = output / "plagiarism"
    plag_dir.mkdir(parents=True, exist_ok=True)
    if plag_flags:
        (plag_dir / "known_pairs.json").write_text(
            json.dumps(plag_flags, indent=2), encoding="utf-8",
        )

    for name, data in [
        ("students.json", students),
        ("tutors.json", tutors),
        ("ai_results.json", all_ai),
        ("scores.json", all_scores),
        ("objections.json", objections),
        ("chat_messages.json", chat_msgs),
    ]:
        (output / name).write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )


def _write_ble_fixture(output: Path) -> None:
    """Write BLE pen simulator configuration."""
    ble_dir = output / "ble"
    ble_dir.mkdir(parents=True, exist_ok=True)
    fixture = {
        "description": "GATT characteristic dumps for pen simulator",
        "pen_model": "P05",
        "gatt_service_uuid": "0000ae30-0000-1000-8000-00805f9b34fb",
        "characteristics": {
            "AE10": {
                "uuid": "0000ae10-0000-1000-8000-00805f9b34fb",
                "properties": ["write-with-response"],
                "description": "Command write characteristic",
            },
            "AE02": {
                "uuid": "0000ae02-0000-1000-8000-00805f9b34fb",
                "properties": ["notify"],
                "description": "Response/data notify characteristic",
            },
        },
        "sample_commands": [
            {"name": "get_pen_info", "hex": "5A5A0000000101000100XXXX"},
            {"name": "start_sync", "hex": "5A5A0000000102000100XXXX"},
        ],
        "note": "Replace XXXX with CRC-16/XMODEM (poly 0x1021, init 0x0000)",
    }
    (ble_dir / "pen_simulator_config.json").write_text(
        json.dumps(fixture, indent=2), encoding="utf-8",
    )


def _write_pages_placeholder(output: Path) -> None:
    """Write placeholder for rendered page images."""
    pages_dir = output / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / ".gitkeep").write_text(
        "# Placeholder for rendered page images (PNG)\n"
        "# Generated by svc-doc-assembly from stroke data\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
