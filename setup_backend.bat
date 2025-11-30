@echo off
REM ========================================
REM SkillBot Backend Setup Script
REM ========================================

echo ========================================
echo SkillBot Backend Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM Check if venv already exists
if exist "venv\Scripts\activate.bat" (
    echo [WARNING] Virtual environment already exists
    echo Do you want to recreate it? This will delete the existing venv.
    set /p "RECREATE=Recreate? (y/N): "
    if /i not "%RECREATE%"=="y" (
        echo Keeping existing virtual environment
        goto :INSTALL_DEPS
    )
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

REM Create virtual environment
echo.
echo [INFO] Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment created
echo.

:INSTALL_DEPS
REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo.

REM Install dependencies
echo [INFO] Installing dependencies from requirements.txt...
echo This may take several minutes...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Ensure your .env file is configured with correct API keys
echo 2. Start the server by running: run_backend.bat
echo.
echo API Endpoints:
echo - Health Check: http://localhost:5001/health
echo - API Docs: http://localhost:5001/docs
echo - API Root: http://localhost:5001/
echo.
pause
