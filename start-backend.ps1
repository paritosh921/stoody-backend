# Start Backend Server Script
# This script starts the FastAPI async backend server

Write-Host "Starting Stoody Backend Server..." -ForegroundColor Cyan

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Change to the backend directory (where the script is located)
Set-Location $ScriptDir

# Check if we're in the backend directory
if (Test-Path "main_async.py") {
    Write-Host "✓ Found main_async.py" -ForegroundColor Green
} else {
    Write-Host "✗ Error: main_async.py not found in backend directory." -ForegroundColor Red
    exit 1
}

# Check if .env exists
if (Test-Path ".env") {
    Write-Host "✓ Found .env configuration" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: .env file not found. Using default configuration." -ForegroundColor Yellow
}

# Start the server
Write-Host "`nStarting server on http://localhost:5001..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Yellow

# Check if venv exists, if so use it
if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "Using virtual environment" -ForegroundColor Green
    & "venv\Scripts\python.exe" main_async.py
} else {
    # Use system Python
    python main_async.py
}
