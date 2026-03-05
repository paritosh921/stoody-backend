@echo off
setlocal
REM ========================================
REM SkillBot Backend TUI Launcher (Windows)
REM ========================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

echo ========================================
echo Starting SkillBot Backend TUI
echo ========================================
echo.

REM Prefer direct venv Python to avoid activation issues
set "VENV_PY=venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment Python not found: %VENV_PY%
    echo.
    echo Create/setup backend venv first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    popd >nul
    pause
    exit /b 1
)

echo [INFO] Using %VENV_PY%
echo [INFO] Launching TUI... (press q to quit)
echo.

"%VENV_PY%" -m scripts.tui
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] TUI exited with code %EXIT_CODE%
    popd >nul
    pause
    exit /b %EXIT_CODE%
)

popd >nul
exit /b 0
