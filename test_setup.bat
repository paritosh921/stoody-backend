@echo off
REM ========================================
REM SkillBot Backend Setup Verification
REM ========================================

echo ========================================
echo Testing SkillBot Backend Setup
echo ========================================
echo.

REM Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found
    goto :END_TEST
) else (
    python --version
    echo [PASS] Python is installed
)
echo.

REM Check venv
echo [2/5] Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo [PASS] Virtual environment exists
) else (
    echo [FAIL] Virtual environment not found
    echo Run setup_backend.bat first
    goto :END_TEST
)
echo.

REM Activate venv and check packages
echo [3/5] Checking installed packages...
call venv\Scripts\activate.bat
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [FAIL] FastAPI not installed
    echo Run setup_backend.bat first
    goto :END_TEST
) else (
    echo [PASS] FastAPI is installed
)
echo.

REM Check .env file
echo [4/5] Checking .env configuration...
if exist ".env" (
    echo [PASS] .env file exists
) else (
    echo [WARN] .env file not found
    echo Some features may not work without proper configuration
)
echo.

REM Check main_async.py
echo [5/5] Checking main application file...
if exist "main_async.py" (
    echo [PASS] main_async.py exists
) else (
    echo [FAIL] main_async.py not found
    goto :END_TEST
)
echo.

REM Summary
echo ========================================
echo [SUCCESS] Setup verification completed!
echo ========================================
echo.
echo Your backend is ready to run.
echo Start the server with: run_backend.bat
echo.
echo Quick checks:
echo - Health endpoint: http://localhost:5001/health
echo - API docs: http://localhost:5001/docs
echo.
goto :EOF

:END_TEST
echo.
echo ========================================
echo [FAILED] Setup verification failed
echo ========================================
echo.
echo Please run setup_backend.bat first to complete setup.
echo.
pause
