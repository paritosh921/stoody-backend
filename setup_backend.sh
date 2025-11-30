#!/bin/bash
# ========================================
# SkillBot Backend Setup Script (Linux/Mac)
# ========================================

echo "========================================"
echo "SkillBot Backend Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

echo "[OK] Python found:"
python3 --version
echo ""

# Check if venv already exists
if [ -f "venv/bin/activate" ]; then
    echo "[WARNING] Virtual environment already exists"
    read -p "Recreate? This will delete the existing venv (y/N): " RECREATE
    if [[ ! "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "Keeping existing virtual environment"
        INSTALL_DEPS=true
    else
        echo "Removing existing virtual environment..."
        rm -rf venv
        INSTALL_DEPS=false
    fi
else
    INSTALL_DEPS=false
fi

# Create virtual environment if needed
if [ "$INSTALL_DEPS" = false ]; then
    echo ""
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi

    echo "[OK] Virtual environment created"
    echo ""
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

# Upgrade pip
echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

echo ""

# Install dependencies
echo "[INFO] Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
echo ""
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "[SUCCESS] Setup completed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Ensure your .env file is configured with correct API keys"
echo "2. Start the server by running: ./run_backend.sh"
echo ""
echo "API Endpoints:"
echo "- Health Check: http://localhost:5001/health"
echo "- API Docs: http://localhost:5001/docs"
echo "- API Root: http://localhost:5001/"
echo ""
