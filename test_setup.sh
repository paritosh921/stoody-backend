#!/bin/bash
# ========================================
# SkillBot Backend Setup Verification
# ========================================

echo "========================================"
echo "Testing SkillBot Backend Setup"
echo "========================================"
echo ""

FAILED=0

# Check Python
echo "[1/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    python3 --version
    echo "[PASS] Python is installed"
else
    echo "[FAIL] Python not found"
    FAILED=1
fi
echo ""

# Check venv
echo "[2/5] Checking virtual environment..."
if [ -f "venv/bin/activate" ]; then
    echo "[PASS] Virtual environment exists"
else
    echo "[FAIL] Virtual environment not found"
    echo "Run ./setup_backend.sh first"
    FAILED=1
fi
echo ""

# Activate venv and check packages
echo "[3/5] Checking installed packages..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    if pip show fastapi &> /dev/null; then
        echo "[PASS] FastAPI is installed"
    else
        echo "[FAIL] FastAPI not installed"
        echo "Run ./setup_backend.sh first"
        FAILED=1
    fi
else
    echo "[SKIP] Cannot check packages (venv not found)"
    FAILED=1
fi
echo ""

# Check .env file
echo "[4/5] Checking .env configuration..."
if [ -f ".env" ]; then
    echo "[PASS] .env file exists"
else
    echo "[WARN] .env file not found"
    echo "Some features may not work without proper configuration"
fi
echo ""

# Check main_async.py
echo "[5/5] Checking main application file..."
if [ -f "main_async.py" ]; then
    echo "[PASS] main_async.py exists"
else
    echo "[FAIL] main_async.py not found"
    FAILED=1
fi
echo ""

# Summary
if [ $FAILED -eq 0 ]; then
    echo "========================================"
    echo "[SUCCESS] Setup verification completed!"
    echo "========================================"
    echo ""
    echo "Your backend is ready to run."
    echo "Start the server with: ./run_backend.sh"
    echo ""
    echo "Quick checks:"
    echo "- Health endpoint: http://localhost:5001/health"
    echo "- API docs: http://localhost:5001/docs"
    echo ""
else
    echo "========================================"
    echo "[FAILED] Setup verification failed"
    echo "========================================"
    echo ""
    echo "Please run ./setup_backend.sh first to complete setup."
    echo ""
    exit 1
fi
