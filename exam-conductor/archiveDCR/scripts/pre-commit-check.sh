#!/usr/bin/env bash
# pre-commit-check.sh — ExamPen pre-commit validation
#
# Enforces:
#   1. File size limits (Python 300, TS 250, SQL 200, Config 150 lines)
#   2. Domain purity (no I/O imports in domain/ layers)
#   3. Cross-service import prohibition
#
# Reference: COMPONENT_INDEPENDENCE_MAP.md §1, CLAUDE.md §File Size Limits
#
# Usage:
#   ./scripts/pre-commit-check.sh          # Check staged files only
#   ./scripts/pre-commit-check.sh --all    # Check all tracked files
#
# Install as git hook:
#   ln -sf ../../scripts/pre-commit-check.sh .git/hooks/pre-commit

set -euo pipefail

# --- Configuration ---
MAX_PYTHON_LINES=300
MAX_TS_LINES=250
MAX_SQL_LINES=200
MAX_CONFIG_LINES=150

FORBIDDEN_DOMAIN_IMPORTS="asyncio|aiohttp|sqlalchemy|nats|httpx|requests"

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

violations=0
warnings=0

# --- Helpers ---
log_error() {
    echo -e "${RED}ERROR${NC}: $1"
    violations=$((violations + 1))
}

log_warn() {
    echo -e "${YELLOW}WARN${NC}: $1"
    warnings=$((warnings + 1))
}

log_ok() {
    echo -e "${GREEN}OK${NC}: $1"
}

# --- Determine files to check ---
if [[ "${1:-}" == "--all" ]]; then
    # Check all tracked files
    get_files() {
        git ls-files "$@" 2>/dev/null || find . -name "$1" -not -path './.git/*' 2>/dev/null
    }
    FILES_MODE="all tracked files"
else
    # Check staged files only
    get_files() {
        git diff --cached --name-only --diff-filter=ACM -- "$@" 2>/dev/null || true
    }
    FILES_MODE="staged files"
fi

echo "=== ExamPen Pre-Commit Check ($FILES_MODE) ==="
echo ""

# ============================================================================
# 1. FILE SIZE ENFORCEMENT
# ============================================================================
echo "--- [1/3] File Size Limits ---"

# Python files: max 300 lines
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ ! -f "$f" ]] && continue
    # Check for EXEMPT header in first 5 lines
    if head -5 "$f" | grep -q '# EXEMPT:'; then
        reason=$(head -5 "$f" | grep '# EXEMPT:' | sed 's/.*# EXEMPT: //')
        log_warn "$f is EXEMPT: $reason"
        continue
    fi
    lines=$(wc -l < "$f")
    if (( lines > MAX_PYTHON_LINES )); then
        log_error "$f has $lines lines (max $MAX_PYTHON_LINES). Add '# EXEMPT: <reason>' or split."
    fi
done < <(get_files '*.py')

# TypeScript files: max 250 lines
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ ! -f "$f" ]] && continue
    if head -5 "$f" | grep -q '// EXEMPT:'; then
        reason=$(head -5 "$f" | grep '// EXEMPT:' | sed 's/.*\/\/ EXEMPT: //')
        log_warn "$f is EXEMPT: $reason"
        continue
    fi
    lines=$(wc -l < "$f")
    if (( lines > MAX_TS_LINES )); then
        log_error "$f has $lines lines (max $MAX_TS_LINES). Add '// EXEMPT: <reason>' or split."
    fi
done < <(get_files '*.ts' '*.tsx')

# SQL files: max 200 lines
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ ! -f "$f" ]] && continue
    lines=$(wc -l < "$f")
    if (( lines > MAX_SQL_LINES )); then
        log_error "$f has $lines lines (max $MAX_SQL_LINES)."
    fi
done < <(get_files '*.sql')

# Config files: max 150 lines
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ ! -f "$f" ]] && continue
    # Skip CI workflow files and infra configs (they have their own structure)
    [[ "$f" == .github/* ]] && continue
    lines=$(wc -l < "$f")
    if (( lines > MAX_CONFIG_LINES )); then
        log_error "$f has $lines lines (max $MAX_CONFIG_LINES). Split into per-service configs."
    fi
done < <(get_files '*.yml' '*.yaml' '*.toml')

echo ""

# ============================================================================
# 2. DOMAIN PURITY CHECK
# ============================================================================
echo "--- [2/3] Domain Purity ---"

# Scan */domain/*.py for forbidden I/O imports
domain_files=$(find services/ hub/ -path '*/domain/*.py' -not -name '__init__.py' 2>/dev/null || true)
if [[ -n "$domain_files" ]]; then
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        # Check for forbidden imports: import asyncio, from asyncio import, etc.
        matches=$(grep -nE "^(import|from)\s+($FORBIDDEN_DOMAIN_IMPORTS)" "$f" 2>/dev/null || true)
        if [[ -n "$matches" ]]; then
            log_error "Domain purity violation in $f — forbidden I/O imports:"
            echo "    $matches"
        fi
    done <<< "$domain_files"
else
    echo "  No domain/ files found (OK for early project stage)."
fi

echo ""

# ============================================================================
# 3. CROSS-SERVICE IMPORT CHECK
# ============================================================================
echo "--- [3/3] Cross-Service Imports ---"

# Ensure no service imports from another service's src/
service_dirs=$(find services/ -maxdepth 1 -mindepth 1 -type d 2>/dev/null || true)
if [[ -n "$service_dirs" ]]; then
    while IFS= read -r svc_dir; do
        [[ -z "$svc_dir" ]] && continue
        svc_name=$(basename "$svc_dir")
        # Check all Python files in this service
        py_files=$(find "$svc_dir" -name '*.py' 2>/dev/null || true)
        [[ -z "$py_files" ]] && continue

        while IFS= read -r py_file; do
            [[ -z "$py_file" ]] && continue
            # Look for imports from other services (e.g., "from services.svc_other" or "import services.svc_other")
            while IFS= read -r other_svc; do
                [[ -z "$other_svc" ]] && continue
                other_name=$(basename "$other_svc")
                [[ "$other_name" == "$svc_name" ]] && continue
                # Convert dashes to underscores for Python import names
                other_import=$(echo "$other_name" | tr '-' '_')
                if grep -qE "(from|import)\s+.*$other_import" "$py_file" 2>/dev/null; then
                    log_error "Cross-service import: $py_file imports from $other_name"
                fi
            done <<< "$service_dirs"
        done <<< "$py_files"
    done <<< "$service_dirs"
else
    echo "  No service directories found (OK for early project stage)."
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=== Summary ==="
echo "  Violations: $violations"
echo "  Warnings:   $warnings"
echo ""

if (( violations > 0 )); then
    echo -e "${RED}FAILED${NC}: $violations violation(s) found. Fix before committing."
    exit 1
fi

if (( warnings > 0 )); then
    echo -e "${YELLOW}PASSED with warnings${NC}: $warnings exemption(s) noted."
else
    log_ok "All checks passed."
fi

exit 0
