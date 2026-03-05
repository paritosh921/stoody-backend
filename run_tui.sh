#!/bin/bash
# ========================================
# SkillBot Backend TUI Launcher (Linux/Mac)
# ========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Starting SkillBot Backend TUI"
echo "========================================"
echo ""

VENV_PY="venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "[ERROR] Virtual environment Python not found: $VENV_PY"
  echo ""
  echo "Create/setup backend venv first:"
  echo "  python -m venv venv"
  echo "  source venv/bin/activate"
  echo "  pip install -r requirements.txt"
  echo ""
  exit 1
fi

echo "[INFO] Using $VENV_PY"
echo "[INFO] Launching TUI... (press q to quit)"
echo ""

"$VENV_PY" -m scripts.tui
