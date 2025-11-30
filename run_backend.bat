@echo off
REM ========================================
REM SkillBot Backend Server Launcher
REM ========================================

echo ========================================
echo Starting SkillBot Backend Server
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM Display environment info
echo ========================================
echo Server Configuration:
echo ========================================
echo - Host: 0.0.0.0 (all interfaces)
echo - Port: 5001 (configured in .env)
echo - Mode: Development (auto-reload enabled)
echo - API Docs: http://localhost:5001/docs
echo - Health Check: http://localhost:5001/health
echo.
echo ========================================
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the FastAPI server using uvicorn via main_async.py
python main_async.py

REM If server exits with error, pause to show the error
if errorlevel 1 (
    echo.
    echo ========================================
    echo [ERROR] Server exited with error code %errorlevel%
    echo ========================================
    pause
)
