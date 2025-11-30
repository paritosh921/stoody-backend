#!/bin/bash
# ========================================
# SkillBot Backend Server Launcher (Linux/Mac)
# ========================================

echo "========================================"
echo "Starting SkillBot Backend Server"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please create a virtual environment first:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo "[OK] Virtual environment activated"
echo ""

# Display environment info
echo "========================================"
echo "Server Configuration:"
echo "========================================"
echo "- Host: 0.0.0.0 (all interfaces)"
echo "- Port: 5001 (configured in .env)"
echo "- Mode: Development (auto-reload enabled)"
echo "- API Docs: http://localhost:5001/docs"
echo "- Health Check: http://localhost:5001/health"
echo ""
echo "========================================"
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the FastAPI server using uvicorn via main_async.py
python main_async.py

# If server exits with error, show error code
if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "[ERROR] Server exited with error code $?"
    echo "========================================"
fi
