"""ExamPen Hub TUI — Textual App entry point.

Run:  python -m src.main   (from hub-tui/ directory)

The app creates a shared ``HubIpcBridge`` instance that connects to
hub-supervisor and other module IPC sockets.  All screens access the
bridge via ``self.app._ipc_bridge``.
"""

from textual.app import App

from src.ipc_bridge import HubIpcBridge
from src.screens.diagnostics import DiagnosticsScreen
from src.screens.dongles import DonglesScreen
from src.screens.exams import ExamsScreen
from src.screens.logs import LogsScreen
from src.screens.menu import MenuScreen
from src.screens.setup import SetupScreen
from src.screens.shutdown import ShutdownScreen
from src.screens.status import StatusScreen
from src.screens.wifi import WiFiScreen

SCREEN_MAP = {
    "setup": SetupScreen,
    "status": StatusScreen,
    "wifi": WiFiScreen,
    "dongles": DonglesScreen,
    "exams": ExamsScreen,
    "diagnostics": DiagnosticsScreen,
    "logs": LogsScreen,
    "shutdown": ShutdownScreen,
}


class HubTuiApp(App):
    """ExamPen Hub TUI — main application."""

    TITLE = "ExamPen Hub TUI"
    SUB_TITLE = "Raspberry Pi Hub Management Console"

    SCREENS = SCREEN_MAP

    def __init__(self, bridge: HubIpcBridge | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ipc_bridge: HubIpcBridge = bridge or HubIpcBridge()

    async def on_mount(self) -> None:
        """Start IPC bridge and show the main menu."""
        try:
            await self._ipc_bridge.start()
        except Exception:
            pass  # Screens degrade gracefully when bridge is disconnected.
        self.push_screen(MenuScreen())

    async def on_unmount(self) -> None:
        """Stop IPC bridge on exit."""
        try:
            await self._ipc_bridge.stop()
        except Exception:
            pass


def main() -> None:
    app = HubTuiApp()
    app.run()


if __name__ == "__main__":
    main()
