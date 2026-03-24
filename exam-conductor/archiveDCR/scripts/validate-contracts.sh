#!/usr/bin/env bash
# validate-contracts.sh — Validate OpenAPI specs and event JSON Schemas
# Exit 1 on any invalid file.
# Requires: Python 3.8+ (stdlib only; pyyaml used if available, else fallback)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_DIR="$REPO_ROOT/new-docs/api"
EVENTS_DIR="$REPO_ROOT/new-docs/contracts/events"

ERRORS=0

fail() {
  echo "FAIL: $1" >&2
  ERRORS=$((ERRORS + 1))
}

PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [ -z "$PYTHON" ]; then
  echo "ERROR: Python 3.8+ required but not found." >&2
  exit 2
fi

# --- Validate OpenAPI YAML specs ---
echo "=== Validating OpenAPI specs in $API_DIR ==="

if [ ! -d "$API_DIR" ]; then
  echo "ERROR: API spec directory not found: $API_DIR" >&2
  exit 1
fi

for spec in "$API_DIR"/*.openapi.yaml; do
  [ -f "$spec" ] || continue
  basename_spec="$(basename "$spec")"

  if ! "$PYTHON" -c "
import sys
# Try pyyaml first; fall back to regex-based structural check
try:
    import yaml
    with open(sys.argv[1]) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        print('Not a YAML mapping', file=sys.stderr)
        sys.exit(1)
    missing = [k for k in ('openapi', 'info', 'paths') if k not in doc]
    if missing:
        print(f'Missing keys: {missing}', file=sys.stderr)
        sys.exit(1)
    if not str(doc['openapi']).startswith('3.'):
        print(f'Bad openapi version: {doc[\"openapi\"]}', file=sys.stderr)
        sys.exit(1)
except ImportError:
    import re
    with open(sys.argv[1]) as f:
        text = f.read()
    has_openapi = bool(re.search(r'^openapi:\s', text, re.MULTILINE))
    has_info    = bool(re.search(r'^info:\s', text, re.MULTILINE))
    has_paths   = bool(re.search(r'^paths:\s', text, re.MULTILINE))
    missing = []
    if not has_openapi: missing.append('openapi')
    if not has_info:    missing.append('info')
    if not has_paths:   missing.append('paths')
    if missing:
        print(f'Missing top-level keys: {missing}', file=sys.stderr)
        sys.exit(1)
    ver = re.search(r'^openapi:\s*[\"'\'']?([\d.]+)', text, re.MULTILINE)
    if ver and not ver.group(1).startswith('3.'):
        print(f'Bad openapi version: {ver.group(1)}', file=sys.stderr)
        sys.exit(1)
" "$spec" 2>&1; then
    fail "$basename_spec — invalid or missing OpenAPI structure"
    continue
  fi

  echo "  OK  $basename_spec"
done

# --- Validate Event JSON Schemas ---
echo ""
echo "=== Validating event schemas in $EVENTS_DIR ==="

if [ ! -d "$EVENTS_DIR" ]; then
  echo "WARN: Events directory not found: $EVENTS_DIR (skipping)" >&2
else
  for schema in "$EVENTS_DIR"/*.schema.json; do
    [ -f "$schema" ] || continue
    basename_schema="$(basename "$schema")"

    if ! "$PYTHON" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        doc = json.load(f)
except Exception as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
if not isinstance(doc, dict):
    print('Not a JSON object', file=sys.stderr)
    sys.exit(1)
if 'type' not in doc and '\$schema' not in doc:
    print('Missing type and \$schema — not a JSON Schema', file=sys.stderr)
    sys.exit(1)
" "$schema" 2>&1; then
      fail "$basename_schema — invalid JSON or missing schema structure"
      continue
    fi

    echo "  OK  $basename_schema"
  done
fi

# --- Summary ---
echo ""
if [ "$ERRORS" -gt 0 ]; then
  echo "VALIDATION FAILED: $ERRORS error(s) found." >&2
  exit 1
else
  echo "All contracts valid."
  exit 0
fi
