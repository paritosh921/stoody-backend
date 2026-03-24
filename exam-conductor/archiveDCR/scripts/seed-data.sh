#!/usr/bin/env bash
# seed-data.sh — Generate and load ExamPen test data
#
# Usage:
#   ./scripts/seed-data.sh --students 40 --exams 3 --questions-per-exam 10
#   ./scripts/seed-data.sh --help
#
# This script is idempotent: running it multiple times with the same seed
# produces the same data and safely replaces existing fixtures.
#
# Reference: TEST_SUITE_SPEC.md §5, CLAUDE.md §Commands

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Defaults ---
STUDENTS=40
EXAMS=3
QUESTIONS_PER_EXAM=10
SEED=42
OUTPUT_DIR="$PROJECT_ROOT/test-suite/fixtures"
LOAD_SQL=false
DB_URL=""

# --- Parse arguments ---
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Generate realistic ExamPen test data for development and testing.

Options:
  --students N            Number of students to generate (default: 40)
  --exams N               Number of exams to generate (default: 3)
  --questions-per-exam N  Questions per exam (default: 10)
  --seed N                Random seed for reproducibility (default: 42)
  --output DIR            Output directory (default: test-suite/fixtures/)
  --load-sql              Load generated SQL into database
  --db-url URL            PostgreSQL connection URL (required with --load-sql)
  --help                  Show this help message

Examples:
  $(basename "$0") --students 40 --exams 3 --questions-per-exam 10
  $(basename "$0") --students 10 --exams 1 --questions-per-exam 5 --seed 123
  $(basename "$0") --load-sql --db-url postgresql://user:pass@localhost:5432/exampen
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --students)       STUDENTS="$2";           shift 2 ;;
        --exams)          EXAMS="$2";               shift 2 ;;
        --questions-per-exam) QUESTIONS_PER_EXAM="$2"; shift 2 ;;
        --seed)           SEED="$2";                shift 2 ;;
        --output)         OUTPUT_DIR="$2";          shift 2 ;;
        --load-sql)       LOAD_SQL=true;            shift ;;
        --db-url)         DB_URL="$2";              shift 2 ;;
        --help|-h)        usage ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# --- Validate ---
if [[ "$LOAD_SQL" == true ]] && [[ -z "$DB_URL" ]]; then
    echo "ERROR: --load-sql requires --db-url"
    exit 1
fi

# --- Find Python ---
PYTHON=""
if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif [[ -f "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: Python not found. Install Python 3.12+ or create a venv."
    exit 1
fi

echo "=== ExamPen Seed Data Generator ==="
echo "  Python:             $PYTHON"
echo "  Students:           $STUDENTS"
echo "  Exams:              $EXAMS"
echo "  Questions/Exam:     $QUESTIONS_PER_EXAM"
echo "  Seed:               $SEED"
echo "  Output:             $OUTPUT_DIR"
echo ""

# --- Create output directories ---
mkdir -p "$OUTPUT_DIR/strokes"
mkdir -p "$OUTPUT_DIR/pages"
mkdir -p "$OUTPUT_DIR/exams"
mkdir -p "$OUTPUT_DIR/plagiarism"
mkdir -p "$OUTPUT_DIR/ble"

# --- Run Python seed script ---
"$PYTHON" "$SCRIPT_DIR/seed_data.py" \
    --students "$STUDENTS" \
    --exams "$EXAMS" \
    --questions-per-exam "$QUESTIONS_PER_EXAM" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR"

echo ""

# --- Optionally load SQL ---
if [[ "$LOAD_SQL" == true ]]; then
    echo "=== Loading seed.sql into database ==="
    if command -v psql &>/dev/null; then
        psql "$DB_URL" -f "$OUTPUT_DIR/seed.sql"
        echo "SQL loaded successfully."
    else
        echo "WARNING: psql not found. Install PostgreSQL client to load SQL."
        echo "Manual load: psql \$DATABASE_URL -f $OUTPUT_DIR/seed.sql"
    fi
fi

echo ""
echo "=== Seed data generation complete ==="
echo "  Fixtures:  $OUTPUT_DIR/"
echo "  SQL:       $OUTPUT_DIR/seed.sql"
echo "  Strokes:   $OUTPUT_DIR/strokes/ ($(find "$OUTPUT_DIR/strokes" -name '*.bin' 2>/dev/null | wc -l) files)"
echo "  Exams:     $OUTPUT_DIR/exams/ ($(find "$OUTPUT_DIR/exams" -name '*.json' 2>/dev/null | wc -l) files)"
