"""Generic status table widget with per-row health coloring."""

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable


# Map health/status strings to Rich markup colors.
_STATUS_COLORS: dict[str, str] = {
    "ok": "green",
    "healthy": "green",
    "pass": "green",
    "complete": "green",
    "connected": "green",
    "degraded": "yellow",
    "warning": "yellow",
    "partial": "yellow",
    "in-progress": "yellow",
    "running": "yellow",
    "failed": "red",
    "error": "red",
    "disconnected": "red",
    "pending": "dim",
    "unknown": "dim",
    "skip": "dim",
}


def colorize_status(value: str) -> str:
    """Wrap a status string in Rich color markup based on its value."""
    color = _STATUS_COLORS.get(value.lower(), "")
    if color:
        return f"[{color}]{value}[/{color}]"
    return value


class StatusTable(Widget):
    """A thin wrapper around DataTable that applies status coloring.

    Usage:
        table = StatusTable(id="dongle-table")
        table.columns = ["Dongle ID", "hci", "Pens", "Health"]
        table.status_column = 3  # index of the health column
        table.rows = [
            ["D1", "hci0", "8/8", "OK"],
            ["D2", "hci1", "7/8", "DEGRADED"],
        ]
    """

    DEFAULT_CSS = """
    StatusTable {
        height: auto;
        max-height: 16;
    }
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        rows: list[list[str]] | None = None,
        status_column: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._columns = columns or []
        self._rows = rows or []
        self._status_column = status_column

    def compose(self) -> ComposeResult:
        yield DataTable(id="inner-table")

    def on_mount(self) -> None:
        table = self.query_one("#inner-table", DataTable)
        for col in self._columns:
            table.add_column(col, key=col)
        self._populate(table)

    def set_data(
        self,
        rows: list[list[str]],
        columns: list[str] | None = None,
        status_column: int | None = None,
    ) -> None:
        """Replace table data. Optionally update columns / status column."""
        if columns is not None:
            self._columns = columns
        if status_column is not None:
            self._status_column = status_column
        self._rows = rows
        try:
            table = self.query_one("#inner-table", DataTable)
            table.clear()
            self._populate(table)
        except Exception:
            pass

    def _populate(self, table: DataTable) -> None:
        for row in self._rows:
            display_row = list(row)
            if self._status_column is not None and 0 <= self._status_column < len(display_row):
                display_row[self._status_column] = colorize_status(
                    display_row[self._status_column]
                )
            table.add_row(*display_row)
