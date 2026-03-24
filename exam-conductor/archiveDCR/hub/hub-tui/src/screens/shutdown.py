"""Safe shutdown screen — pre-checks, confirmation, and shutdown sequence."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Static

from src.widgets.footer import HubFooter


class ShutdownScreen(Screen):
    """Safe shutdown with pre-flight checks and confirmation dialog."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    DEFAULT_CSS = """
    ShutdownScreen {
        align: center middle;
    }
    ShutdownScreen #shutdown-container {
        width: 60;
        height: auto;
        border: solid $error;
        padding: 1 2;
    }
    ShutdownScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    ShutdownScreen .check-line {
        margin: 0 1;
    }
    ShutdownScreen .warning-line {
        color: $warning;
        margin: 0 1;
    }
    ShutdownScreen .ok-line {
        color: $success;
        margin: 0 1;
    }
    ShutdownScreen .section-title {
        text-style: bold;
        margin-top: 1;
    }
    ShutdownScreen .action-row {
        layout: horizontal;
        height: 3;
        margin-top: 1;
    }
    ShutdownScreen .action-row Button {
        margin-right: 1;
    }
    ShutdownScreen .sequence-line {
        margin: 0 1;
        color: $text-muted;
    }
    """

    _confirmed: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="shutdown-container"):
            yield Static("[8] Safe Shutdown", classes="screen-title")

            # Pre-shutdown checks
            yield Static("Pre-Shutdown Checks", classes="section-title")
            yield Static(
                "[yellow]WARNING:[/yellow] No active exam session detected",
                id="check-exam",
                classes="ok-line",
            )
            yield Static(
                "[yellow]WARNING:[/yellow] 2 pending uploads remaining",
                id="check-uploads",
                classes="warning-line",
            )
            yield Static(
                "[green]OK:[/green] Filesystem healthy",
                id="check-fs",
                classes="ok-line",
            )

            # Confirmation
            yield Static("Shutdown Sequence", classes="section-title")
            yield Static("  1. Flush pending writes (sync)", classes="sequence-line")
            yield Static(
                "  2. Unmount USB (/mnt/exampen-backup)", classes="sequence-line"
            )
            yield Static("  3. Power off (systemctl poweroff)", classes="sequence-line")

            with Horizontal(classes="action-row"):
                yield Button(
                    "Confirm Shutdown",
                    id="btn-confirm",
                    variant="error",
                )
                yield Button("Cancel", id="btn-cancel", variant="default")

            yield Static("", id="shutdown-result")
        yield HubFooter()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        result = self.query_one("#shutdown-result", Static)

        if event.button.id == "btn-cancel":
            self.app.pop_screen()
            return

        if event.button.id == "btn-confirm":
            if not self._confirmed:
                # First press — ask for double confirmation.
                self._confirmed = True
                result.styles.color = "yellow"
                result.update(
                    "Press Confirm Shutdown again to proceed. "
                    "This will power off the hub."
                )
                return

            # Second press — execute shutdown sequence (stub).
            result.styles.color = "red"
            result.update(
                "Shutdown sequence initiated (stub):\n"
                "  [1/3] sync ... done\n"
                "  [2/3] umount /mnt/exampen-backup ... done\n"
                "  [3/3] systemctl poweroff (stub -- not executed in dev)"
            )
            # In production: subprocess.run(["systemctl", "poweroff"])

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
