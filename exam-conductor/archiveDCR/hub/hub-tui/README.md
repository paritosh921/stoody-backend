# ExamPen Hub TUI

Textual-based terminal UI for the ExamPen Raspberry Pi hub. Runs on HDMI console or USB serial (80x24 minimum).

## Quick Start

```bash
cd hub/hub-tui

# Create venv and install
python3.12 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -e ".[dev]"

# Run
python -m src.main
```

## Screens

| Key | Screen       | Description                              |
|-----|-------------|------------------------------------------|
| 1   | Setup       | First-boot config (hub code, URL, mode)  |
| 2   | Status      | Live dashboard (dongles, sync, storage)  |
| 3   | WiFi        | Network scan, connect, status            |
| 4   | Dongles     | BLE dongle management                    |
| 5   | Exams       | Exam session history                     |
| 6   | Diagnostics | H1-H7, S1-S5, B1-B4 test runner         |
| 7   | Logs        | Tabbed log viewer with level filter      |
| 8   | Shutdown    | Safe power-off with pre-checks           |

Press a number key from the main menu to navigate. Press Escape to return.

## Tests

```bash
pytest tests/ -v
```

## Architecture

All screens are shells with placeholder data. Real IPC data binding (polling `hub-supervisor` via Unix domain sockets) is wired in Wave 4.

```
src/
  main.py              App entry point
  screens/
    menu.py            Main menu
    setup.py           First-boot config
    status.py          Live dashboard
    wifi.py            WiFi management
    dongles.py         Dongle management
    exams.py           Exam history
    diagnostics.py     Test runner
    logs.py            Log viewer
    shutdown.py        Safe shutdown
  widgets/
    footer.py          Hub info footer
    progress_bar.py    Sync progress bar
    status_table.py    Table with health coloring
tests/
  test_app.py          App startup + navigation tests
```
