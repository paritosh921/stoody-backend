"""
Production Startup Script for Windows
Handles 1000+ concurrent users with optimized settings for Windows platform
"""

import subprocess
import sys

def start_server():
    """
    Start Uvicorn server with Windows-optimized settings

    On Windows, --workers doesn't work reliably due to multiprocessing limitations.
    Instead, we use a single worker with high concurrency settings:
    - limit_concurrency: 1000 (max concurrent connections)
    - backlog: 2048 (socket backlog queue)
    - timeout_keep_alive: 5 (keep connections alive)
    - loop: uvloop (faster event loop, if available)
    """

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main_async:app",
        "--host", "0.0.0.0",
        "--port", "5001",
        "--limit-concurrency", "1000",  # Handle 1000 concurrent connections
        "--backlog", "2048",            # Socket backlog
        "--timeout-keep-alive", "5",    # Keep-alive timeout
        "--no-access-log",              # Reduce I/O overhead
    ]

    print("Starting SkillBot Production Server (Windows Optimized)")
    print(f"Concurrency: 1000 connections")
    print(f"Binding to: http://0.0.0.0:5001")
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()
