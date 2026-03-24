"""Textual tests — verify app starts, menu renders, each screen loads."""

import pytest
from textual.pilot import Pilot

from src.main import HubTuiApp
from src.screens.diagnostics import DiagnosticsScreen
from src.screens.dongles import DonglesScreen
from src.screens.exams import ExamsScreen
from src.screens.logs import LogsScreen
from src.screens.menu import MenuScreen
from src.screens.setup import SetupScreen
from src.screens.shutdown import ShutdownScreen
from src.screens.status import StatusScreen
from src.screens.wifi import WiFiScreen


@pytest.fixture
def app() -> HubTuiApp:
    return HubTuiApp()


async def test_app_starts(app: HubTuiApp) -> None:
    """App mounts without error and shows the menu screen."""
    async with app.run_test() as pilot:
        assert isinstance(app.screen, MenuScreen)


async def test_menu_has_eight_items(app: HubTuiApp) -> None:
    """Menu screen displays all 8 navigation options."""
    async with app.run_test() as pilot:
        # Query all Static children with the menu-item class.
        menu_items = app.screen.query(".menu-item")
        assert len(menu_items) == 8
        assert isinstance(app.screen, MenuScreen)


async def test_navigate_to_setup(app: HubTuiApp) -> None:
    """Pressing '1' navigates to the Setup screen."""
    async with app.run_test() as pilot:
        await pilot.press("1")
        assert isinstance(app.screen, SetupScreen)


async def test_navigate_to_status(app: HubTuiApp) -> None:
    """Pressing '2' navigates to the Status screen."""
    async with app.run_test() as pilot:
        await pilot.press("2")
        assert isinstance(app.screen, StatusScreen)


async def test_navigate_to_wifi(app: HubTuiApp) -> None:
    """Pressing '3' navigates to the WiFi screen."""
    async with app.run_test() as pilot:
        await pilot.press("3")
        assert isinstance(app.screen, WiFiScreen)


async def test_navigate_to_dongles(app: HubTuiApp) -> None:
    """Pressing '4' navigates to the Dongles screen."""
    async with app.run_test() as pilot:
        await pilot.press("4")
        assert isinstance(app.screen, DonglesScreen)


async def test_navigate_to_exams(app: HubTuiApp) -> None:
    """Pressing '5' navigates to the Exams screen."""
    async with app.run_test() as pilot:
        await pilot.press("5")
        assert isinstance(app.screen, ExamsScreen)


async def test_navigate_to_diagnostics(app: HubTuiApp) -> None:
    """Pressing '6' navigates to the Diagnostics screen."""
    async with app.run_test() as pilot:
        await pilot.press("6")
        assert isinstance(app.screen, DiagnosticsScreen)


async def test_navigate_to_logs(app: HubTuiApp) -> None:
    """Pressing '7' navigates to the Logs screen."""
    async with app.run_test() as pilot:
        await pilot.press("7")
        assert isinstance(app.screen, LogsScreen)


async def test_navigate_to_shutdown(app: HubTuiApp) -> None:
    """Pressing '8' navigates to the Shutdown screen."""
    async with app.run_test() as pilot:
        await pilot.press("8")
        assert isinstance(app.screen, ShutdownScreen)


async def test_escape_returns_to_menu(app: HubTuiApp) -> None:
    """Pressing Escape from a sub-screen returns to the menu."""
    async with app.run_test() as pilot:
        await pilot.press("2")  # Go to Status
        assert isinstance(app.screen, StatusScreen)
        await pilot.press("escape")
        assert isinstance(app.screen, MenuScreen)


async def test_diagnostics_screen_loads(app: HubTuiApp) -> None:
    """Diagnostics screen loads with all 16 tests in PENDING state."""
    async with app.run_test() as pilot:
        await pilot.press("6")
        assert isinstance(app.screen, DiagnosticsScreen)
        diag = app.screen
        # 7 HW + 5 SW + 4 BLE = 16 tests, all start PENDING
        assert diag._state.pending_count == 16
        assert diag._state.pass_count == 0
