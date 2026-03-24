"""Setup screen — first-boot configuration for hub provisioning."""

import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, Select, Static

from src.widgets.footer import HubFooter

_HUB_CODE_RE = re.compile(r"^[A-Za-z0-9]{12}$")
_HTTPS_RE = re.compile(r"^https://\S+$")

UPLINK_MODES = [
    ("wifi", "WiFi"),
    ("mobile", "Mobile"),
    ("auto", "Auto"),
]


class SetupScreen(Screen):
    """First-boot setup: hub code, backend URL, uplink mode."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]

    DEFAULT_CSS = """
    SetupScreen {
        align: center middle;
    }
    SetupScreen #setup-container {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    SetupScreen .field-label {
        margin-top: 1;
        text-style: bold;
    }
    SetupScreen .validation-msg {
        color: $error;
        height: 1;
    }
    SetupScreen .success-msg {
        color: $success;
        height: 1;
    }
    SetupScreen #save-btn {
        margin-top: 1;
        width: 100%;
    }
    SetupScreen .screen-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="setup-container"):
            yield Static("[1] Setup - First Boot Configuration", classes="screen-title")

            yield Label("Hub Unique Code (12 alphanumeric chars):", classes="field-label")
            yield Input(
                placeholder="e.g. ABCD12345678",
                id="hub-code",
                max_length=12,
            )
            yield Static("", id="hub-code-msg", classes="validation-msg")

            yield Label("Backend URL (HTTPS):", classes="field-label")
            yield Input(
                placeholder="https://api.exampen.example.com",
                id="backend-url",
            )
            yield Static("", id="backend-url-msg", classes="validation-msg")

            yield Label("Uplink Mode:", classes="field-label")
            yield Select(
                [(label, value) for value, label in UPLINK_MODES],
                id="uplink-mode",
                value="wifi",
            )

            yield Button("Save Configuration", id="save-btn", variant="primary")
            yield Static("", id="save-result", classes="validation-msg")
        yield HubFooter()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live validation feedback."""
        if event.input.id == "hub-code":
            msg = self.query_one("#hub-code-msg", Static)
            if not event.value:
                msg.update("")
            elif _HUB_CODE_RE.match(event.value):
                msg.styles.color = "green"
                msg.update("Valid hub code")
            else:
                msg.styles.color = "red"
                remaining = 12 - len(event.value)
                if remaining > 0:
                    msg.update(f"Need {remaining} more character(s), alphanumeric only")
                else:
                    msg.update("Must be exactly 12 alphanumeric characters")

        elif event.input.id == "backend-url":
            msg = self.query_one("#backend-url-msg", Static)
            if not event.value:
                msg.update("")
            elif _HTTPS_RE.match(event.value):
                msg.styles.color = "green"
                msg.update("Valid HTTPS URL")
            else:
                msg.styles.color = "red"
                msg.update("Must start with https://")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save-btn":
            return

        hub_code = self.query_one("#hub-code", Input).value.strip()
        backend_url = self.query_one("#backend-url", Input).value.strip()
        uplink_mode = self.query_one("#uplink-mode", Select).value
        result = self.query_one("#save-result", Static)

        # Validate
        errors: list[str] = []
        if not _HUB_CODE_RE.match(hub_code):
            errors.append("Invalid hub code")
        if not _HTTPS_RE.match(backend_url):
            errors.append("Invalid backend URL")
        if uplink_mode not in ("wifi", "mobile", "auto"):
            errors.append("Invalid uplink mode")

        if errors:
            result.styles.color = "red"
            result.update("; ".join(errors))
            return

        # Placeholder: write to /etc/exampen/hub.conf
        # In production, this writes the config and triggers backend verification.
        result.styles.color = "green"
        result.update(
            f"Saved: code={hub_code}, url={backend_url}, mode={uplink_mode} "
            f"(stub -- /etc/exampen/hub.conf write pending)"
        )

    def action_pop_screen(self) -> None:
        self.app.pop_screen()
