"""Sync progress bar widget with status breakdown."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class SyncProgressBar(Widget):
    """Displays a block-character progress bar with status counts.

    Example rendering:
        ████████████░░░░ 37/40 (92%)
        Complete: 34  In-progress: 3  Failed: 0  Pending: 3
    """

    DEFAULT_CSS = """
    SyncProgressBar {
        height: 4;
        padding: 0 1;
    }
    SyncProgressBar .bar-line {
        width: 1fr;
    }
    SyncProgressBar .counts-line {
        width: 1fr;
        color: $text-muted;
    }
    """

    total: reactive[int] = reactive(40)
    complete: reactive[int] = reactive(0)
    in_progress: reactive[int] = reactive(0)
    failed: reactive[int] = reactive(0)
    pending: reactive[int] = reactive(0)

    BAR_WIDTH = 20

    def compose(self) -> ComposeResult:
        yield Static(self._render_bar(), id="bar-line", classes="bar-line")
        yield Static(self._render_counts(), id="counts-line", classes="counts-line")

    def _render_bar(self) -> str:
        if self.total == 0:
            pct = 0
            filled = 0
        else:
            done = self.complete + self.in_progress
            pct = int(done / self.total * 100)
            filled = int(done / self.total * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled
        done_count = self.complete + self.in_progress
        return f"{'█' * filled}{'░' * empty} {done_count}/{self.total} ({pct}%)"

    def _render_counts(self) -> str:
        return (
            f"Complete: {self.complete}  "
            f"In-progress: {self.in_progress}  "
            f"Failed: {self.failed}  "
            f"Pending: {self.pending}"
        )

    def _on_mount(self) -> None:
        self._refresh()

    def watch_total(self) -> None:
        self._refresh()

    def watch_complete(self) -> None:
        self._refresh()

    def watch_in_progress(self) -> None:
        self._refresh()

    def watch_failed(self) -> None:
        self._refresh()

    def watch_pending(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.query_one("#bar-line", Static).update(self._render_bar())
            self.query_one("#counts-line", Static).update(self._render_counts())
        except Exception:
            pass
