"""Desktop bug reports screen - inspect Help-tab messages from agent users."""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Static

from ..widgets.confirm_dialog import ConfirmDialog

TRANSFER_CONFIG_PATH = Path.home() / ".stoody_tui_transfer.json"


class BugReportsScreen(Screen):
    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("enter", "inspect_selected", "Inspect"),
        ("c", "copy_details", "Copy Full Msg"),
        ("o", "send_images_to_pc", "Send Images"),
        ("x", "delete_selected", "Delete Selected"),
        ("X", "delete_all", "Delete All"),
        ("j", "details_down", "Details Down"),
        ("k", "details_up", "Details Up"),
        ("pagedown", "details_page_down", "Details PgDn"),
        ("pageup", "details_page_up", "Details PgUp"),
        ("home", "details_home", "Details Top"),
        ("end", "details_end", "Details End"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="bug-reports-container"):
            yield Static(" Desktop User Messages ", classes="screen-title")
            yield DataTable(id="bug-table")
            yield Static(
                " Enter=inspect  c=copy  o=send images  x=delete selected  X=delete all",
                id="bug-status-bar",
            )
            with Vertical(id="bug-transfer-panel"):
                with Horizontal(classes="bug-transfer-row"):
                    yield Static("SSH Target", classes="field-label")
                    yield Input(placeholder="user@host or ssh-alias", id="bug-ssh-target")
                    yield Static("Port", classes="field-label")
                    yield Input(value="22", id="bug-ssh-port")
                with Horizontal(classes="bug-transfer-row"):
                    yield Static("Remote Path", classes="field-label")
                    yield Input(placeholder="/home/user/stoody-images", id="bug-remote-path")
                with Horizontal(classes="bug-transfer-row"):
                    yield Static("SSH Key (optional)", classes="field-label")
                    yield Input(placeholder="~/.ssh/id_ed25519", id="bug-ssh-key")
                with Horizontal(id="bug-transfer-actions"):
                    yield Button("Save Transfer Settings", id="bug-save-transfer")
                    yield Button("Test SSH", id="bug-test-ssh")
            with Horizontal(id="bug-actions"):
                yield Button("Copy Full Message", id="bug-copy", variant="primary")
                yield Button("Send Images To PC", id="bug-send-images", variant="success")
                yield Button("Delete Selected", id="bug-delete-selected", variant="warning")
                yield Button("Delete All", id="bug-delete-all", variant="error")
            with VerticalScroll(id="bug-details-pane"):
                yield Static("", id="bug-details")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#bug-table", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Created",
            "Ticket",
            "User",
            "Tenant",
            "DB",
            "App",
            "Pen",
            "Att",
            "Title",
        )
        self._rows: list[dict] = []
        self._detail_cache: dict[str, str] = {}
        self._detail_obj_cache: dict[str, dict] = {}
        self._loading_key: str | None = None
        self._current_detail_text: str = ""
        self._load_transfer_config()
        self.load_data()

    def _set_status(self, message: str) -> None:
        self.query_one("#bug-status-bar", Static).update(f" {message}")

    def _set_details(self, text: str) -> None:
        self._current_detail_text = text
        self.query_one("#bug-details", Static).update(Text(text))
        try:
            self.query_one("#bug-details-pane", VerticalScroll).scroll_home(animate=False)
        except Exception:
            pass

    def _selected_row(self) -> dict | None:
        table = self.query_one("#bug-table", DataTable)
        if table.cursor_row is None or table.cursor_row < 0:
            return None
        if table.cursor_row >= len(self._rows):
            return None
        return self._rows[table.cursor_row]

    def _detail_key(self, row: dict) -> str:
        return f"{row.get('db_name')}::{row.get('_id')}"

    def _read_transfer_config(self) -> Dict[str, str]:
        return {
            "ssh_target": self.query_one("#bug-ssh-target", Input).value.strip(),
            "ssh_port": self.query_one("#bug-ssh-port", Input).value.strip() or "22",
            "remote_path": self.query_one("#bug-remote-path", Input).value.strip(),
            "ssh_key": self.query_one("#bug-ssh-key", Input).value.strip(),
        }

    def _load_transfer_config(self) -> None:
        data: Dict[str, Any] = {}
        if TRANSFER_CONFIG_PATH.exists():
            try:
                data = json.loads(TRANSFER_CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        self.query_one("#bug-ssh-target", Input).value = str(data.get("ssh_target", ""))
        self.query_one("#bug-ssh-port", Input).value = str(data.get("ssh_port", "22"))
        self.query_one("#bug-remote-path", Input).value = str(data.get("remote_path", ""))
        self.query_one("#bug-ssh-key", Input).value = str(data.get("ssh_key", ""))

    def action_save_transfer_settings(self) -> None:
        cfg = self._read_transfer_config()
        TRANSFER_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self._set_status(f"Saved transfer settings: {TRANSFER_CONFIG_PATH}")

    def _validate_transfer_config(self, cfg: Dict[str, str]) -> str | None:
        if not cfg.get("ssh_target"):
            return "SSH target is required."
        if not cfg.get("remote_path"):
            return "Remote path is required."
        return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "bug-copy":
            self.action_copy_details()
        elif bid == "bug-send-images":
            self.action_send_images_to_pc()
        elif bid == "bug-delete-selected":
            self.action_delete_selected()
        elif bid == "bug-delete-all":
            self.action_delete_all()
        elif bid == "bug-save-transfer":
            self.action_save_transfer_settings()
        elif bid == "bug-test-ssh":
            self.action_test_ssh()

    def action_copy_details(self) -> None:
        text = (self._current_detail_text or "").strip()
        if not text:
            self._set_status("No message details loaded to copy.")
            return
        copy_fn = getattr(self.app, "copy_to_clipboard", None)
        if not callable(copy_fn):
            self._set_status("Clipboard copy not supported in this terminal.")
            return
        try:
            copy_fn(text)
            self._set_status(f"Copied full message text ({len(text)} chars) to clipboard.")
        except Exception as exc:
            self._set_status(f"Copy failed: {exc}")

    def action_test_ssh(self) -> None:
        cfg = self._read_transfer_config()
        error = self._validate_transfer_config(cfg)
        if error:
            self._set_status(error)
            return
        self._test_ssh_worker(cfg)

    @work(thread=True)
    def _test_ssh_worker(self, cfg: Dict[str, str]) -> None:
        cmd = ["ssh", "-p", cfg.get("ssh_port", "22"), "-o", "BatchMode=yes"]
        ssh_key = cfg.get("ssh_key", "").strip()
        if ssh_key:
            cmd.extend(["-i", str(Path(ssh_key).expanduser())])
        cmd.extend([cfg["ssh_target"], "echo SSH_OK"])
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
            out = (res.stdout or "").strip()
            self.app.call_from_thread(self._set_status, f"SSH test success: {out or 'OK'}")
        except Exception as exc:
            self.app.call_from_thread(self._set_status, f"SSH test failed: {exc}")

    def action_send_images_to_pc(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No message selected.")
            return
        cfg = self._read_transfer_config()
        error = self._validate_transfer_config(cfg)
        if error:
            self._set_status(error)
            return
        self._send_images_worker(row, cfg)

    @work(thread=True)
    def _send_images_worker(self, row: dict, cfg: Dict[str, str]) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        detail = db.get_desktop_bug_report_details(
            report_id=row["_id"],
            db_name=row["db_name"],
        )
        attachments = [
            a for a in (detail.get("attachments") or [])
            if str(a.get("content_type", "")).lower().startswith("image/")
        ]
        if not attachments:
            self.app.call_from_thread(self._set_status, "No image attachments in selected message.")
            return

        ticket = str(detail.get("ticket_id") or "report")
        temp_root = Path(tempfile.mkdtemp(prefix=f"stoody_msg_{ticket}_"))
        written_files: list[Path] = []

        for item in attachments:
            storage_path = str(item.get("storage_path") or "")
            raw = db._download_storage_bytes(storage_path)  # noqa: SLF001 (internal helper reuse)
            if not raw:
                continue
            filename = str(item.get("filename") or f"img_{len(written_files)+1}.png")
            target = temp_root / filename
            target.write_bytes(raw)
            written_files.append(target)

        if not written_files:
            self.app.call_from_thread(self._set_status, "Failed to download image bytes from storage.")
            return

        html_path = temp_root / "index.html"
        html_parts = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>Stoody Message Images</title></head><body>",
            f"<h2>Ticket {ticket}</h2>",
        ]
        for f in written_files:
            html_parts.append(f"<div style='margin-bottom:16px'><p>{f.name}</p><img src='{f.name}' style='max-width:100%;height:auto;border:1px solid #ccc'/></div>")
        html_parts.append("</body></html>")
        html_path.write_text("\n".join(html_parts), encoding="utf-8")
        written_files.append(html_path)

        remote_base = cfg["remote_path"].rstrip("/")
        remote_dir = f"{remote_base}/{ticket}"
        ssh_cmd = ["ssh", "-p", cfg.get("ssh_port", "22"), "-o", "BatchMode=yes"]
        scp_cmd = ["scp", "-P", cfg.get("ssh_port", "22"), "-o", "BatchMode=yes"]
        ssh_key = cfg.get("ssh_key", "").strip()
        if ssh_key:
            key_path = str(Path(ssh_key).expanduser())
            ssh_cmd.extend(["-i", key_path])
            scp_cmd.extend(["-i", key_path])

        try:
            mkdir_remote = f"mkdir -p {shlex.quote(remote_dir)}"
            subprocess.run(
                [*ssh_cmd, cfg["ssh_target"], mkdir_remote],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            subprocess.run(
                [*scp_cmd, *[str(p) for p in written_files], f"{cfg['ssh_target']}:{remote_dir}/"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.app.call_from_thread(
                self._set_status,
                f"Sent {len(written_files)-1} image(s) to {cfg['ssh_target']}:{remote_dir} (open index.html).",
            )
        except Exception as exc:
            self.app.call_from_thread(self._set_status, f"Image send failed: {exc}")

    def action_delete_selected(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No message selected.")
            return
        ticket = row.get("ticket_id", "")
        self.app.push_screen(
            ConfirmDialog(
                "Delete Selected Message",
                f"Delete message {ticket} and all stored image attachments?",
                confirm_label="Delete",
            ),
            lambda ok: self._confirm_delete_selected(bool(ok)),
        )

    def _confirm_delete_selected(self, confirmed: bool) -> None:
        if not confirmed:
            self._set_status("Delete cancelled.")
            return
        row = self._selected_row()
        if not row:
            self._set_status("No message selected.")
            return
        self._delete_selected_worker(str(row.get("_id")), str(row.get("db_name")), str(row.get("ticket_id", "")))

    @work(thread=True)
    def _delete_selected_worker(self, report_id: str, db_name: str, ticket_id: str) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        result = db.delete_desktop_bug_report(report_id=report_id, db_name=db_name)
        if result.get("deleted"):
            removed = int(result.get("deleted_storage", 0) or 0)
            self.app.call_from_thread(self._set_status, f"Deleted {ticket_id} and {removed} attachment object(s).")
            self.app.call_from_thread(self.load_data)
        else:
            self.app.call_from_thread(self._set_status, f"Delete failed for {ticket_id}.")

    def action_delete_all(self) -> None:
        self.app.push_screen(
            ConfirmDialog(
                "Delete All Messages",
                "Delete ALL desktop user messages and all stored attachment objects across tenants?",
                confirm_label="Delete All",
            ),
            lambda ok: self._confirm_delete_all(bool(ok)),
        )

    def _confirm_delete_all(self, confirmed: bool) -> None:
        if not confirmed:
            self._set_status("Delete all cancelled.")
            return
        self._delete_all_worker()

    @work(thread=True)
    def _delete_all_worker(self) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        result = db.delete_all_desktop_bug_reports()
        docs = int(result.get("deleted_docs", 0) or 0)
        storage = int(result.get("deleted_storage", 0) or 0)
        self.app.call_from_thread(
            self._set_status,
            f"Deleted {docs} messages and {storage} attachment objects.",
        )
        self.app.call_from_thread(self.load_data)

    def _scroll_details(self, amount: int) -> None:
        try:
            pane = self.query_one("#bug-details-pane", VerticalScroll)
            pane.scroll_relative(y=amount, animate=False)
        except Exception:
            pass

    def action_details_down(self) -> None:
        self._scroll_details(3)

    def action_details_up(self) -> None:
        self._scroll_details(-3)

    def action_details_page_down(self) -> None:
        self._scroll_details(18)

    def action_details_page_up(self) -> None:
        self._scroll_details(-18)

    def action_details_home(self) -> None:
        try:
            self.query_one("#bug-details-pane", VerticalScroll).scroll_home(animate=False)
        except Exception:
            pass

    def action_details_end(self) -> None:
        try:
            self.query_one("#bug-details-pane", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def action_refresh(self) -> None:
        self.load_data()

    @work(thread=True)
    def load_data(self) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        rows = db.list_desktop_bug_reports(limit=500)
        self.app.call_from_thread(self._populate, rows)

    def _populate(self, rows: list[dict]) -> None:
        table = self.query_one("#bug-table", DataTable)
        table.clear()
        self._rows = rows
        for row in rows:
            created_at = row.get("created_at")
            if isinstance(created_at, datetime):
                created = created_at.strftime("%Y-%m-%d %H:%M")
            else:
                created = str(created_at or "")
            title = str(row.get("title") or "").strip()
            table.add_row(
                created,
                row.get("ticket_id", ""),
                str(row.get("username") or row.get("user_id") or ""),
                str(row.get("tenant_name") or row.get("tenant_id") or ""),
                str(row.get("db_name") or ""),
                str(row.get("app_version") or ""),
                str(row.get("pen_mac") or ""),
                str(row.get("attachment_count") or 0),
                title[:48] + ("..." if len(title) > 48 else ""),
            )
        if rows:
            self._set_status(f"Loaded {len(rows)} desktop user messages.")
            self.action_inspect_selected()
        else:
            self._set_status("No desktop user messages found.")
            self._set_details(" No desktop user messages available.")

    def action_inspect_selected(self) -> None:
        row = self._selected_row()
        if not row:
            self._set_status("No message selected.")
            return
        key = self._detail_key(row)
        if key in self._detail_cache:
            self._set_details(self._detail_cache[key])
            self._set_status(f"Showing cached details for {row.get('ticket_id', '')}.")
            return
        if self._loading_key == key:
            return
        self._loading_key = key
        self._set_details(" Loading message details...")
        self._set_status(f"Inspecting {row.get('ticket_id', '')} ...")
        self._load_details(row)

    @work(thread=True)
    def _load_details(self, row: dict) -> None:
        db = self.app.db  # type: ignore[attr-defined]
        key = self._detail_key(row)
        try:
            detail = db.get_desktop_bug_report_details(
                report_id=row["_id"],
                db_name=row["db_name"],
            )
            text = detail.get("pretty_text", "No detail.")
            status = f"Loaded details for {row.get('ticket_id', '')}."
            self.app.call_from_thread(self._store_detail_obj, key, detail)
        except Exception as exc:
            text = f"Failed to load message details:\n{exc}"
            status = f"Failed to inspect {row.get('ticket_id', '')}."
        self.app.call_from_thread(self._apply_details, key, text, status)

    def _store_detail_obj(self, key: str, detail: dict) -> None:
        self._detail_obj_cache[key] = detail

    def _apply_details(self, key: str, text: str, status: str) -> None:
        self._loading_key = None
        self._detail_cache[key] = text
        self._set_details(text)
        self._set_status(status)

