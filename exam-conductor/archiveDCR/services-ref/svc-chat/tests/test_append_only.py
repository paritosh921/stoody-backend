"""Append-only contract verification for MessageRepo.

Test IDs: U-CHAT-AO-01 through U-CHAT-AO-04

These tests verify the CRITICAL invariant: MessageRepo exposes NO
update or delete methods. If anyone adds such methods, these tests
fail — catching the violation before it reaches production.
"""

import inspect

from src.storage.message_repo import MessageRepo


# -- U-CHAT-AO-01: No update method exists --------------------------------


def test_no_update_method():
    """U-CHAT-AO-01: MessageRepo has no method containing 'update'."""
    methods = _public_methods(MessageRepo)
    update_methods = [m for m in methods if "update" in m.lower()]
    assert update_methods == [], (
        f"APPEND-ONLY VIOLATION: MessageRepo has update methods: {update_methods}"
    )


# -- U-CHAT-AO-02: No delete method exists --------------------------------


def test_no_delete_method():
    """U-CHAT-AO-02: MessageRepo has no method containing 'delete'."""
    methods = _public_methods(MessageRepo)
    delete_methods = [m for m in methods if "delete" in m.lower()]
    assert delete_methods == [], (
        f"APPEND-ONLY VIOLATION: MessageRepo has delete methods: {delete_methods}"
    )


# -- U-CHAT-AO-03: No remove method exists --------------------------------


def test_no_remove_method():
    """U-CHAT-AO-03: MessageRepo has no method containing 'remove'."""
    methods = _public_methods(MessageRepo)
    remove_methods = [m for m in methods if "remove" in m.lower()]
    assert remove_methods == [], (
        f"APPEND-ONLY VIOLATION: MessageRepo has remove methods: {remove_methods}"
    )


# -- U-CHAT-AO-04: No edit method exists ----------------------------------


def test_no_edit_method():
    """U-CHAT-AO-04: MessageRepo has no method containing 'edit'."""
    methods = _public_methods(MessageRepo)
    edit_methods = [m for m in methods if "edit" in m.lower()]
    assert edit_methods == [], (
        f"APPEND-ONLY VIOLATION: MessageRepo has edit methods: {edit_methods}"
    )


# -- U-CHAT-AO-05: Only expected public methods exist ---------------------


def test_only_append_and_read_methods():
    """U-CHAT-AO-05: MessageRepo exposes only the expected API surface."""
    allowed = {
        "append_message",
        "get_thread",
        "list_threads",
        "append_read_receipt",
    }
    actual = set(_public_methods(MessageRepo))
    unexpected = actual - allowed
    assert unexpected == set(), (
        f"Unexpected methods on MessageRepo: {unexpected}. "
        "Review for append-only contract compliance."
    )


# -- U-CHAT-AO-06: No SQL UPDATE/DELETE in source -------------------------


def test_no_sql_update_in_source():
    """U-CHAT-AO-06: No SQL UPDATE statement in message_repo source."""
    source = inspect.getsource(MessageRepo)
    # The only acceptable UPDATE is in the read_receipts upsert
    # (ON CONFLICT ... DO UPDATE SET read_at), which is an append-style
    # upsert for the receipt, not a message mutation.
    lines = source.split("\n")
    for i, line in enumerate(lines):
        upper = line.upper().strip()
        if "UPDATE" in upper and "chat_messages" in line:
            raise AssertionError(
                f"SQL UPDATE on chat_messages found at line {i}: {line.strip()}"
            )


def test_no_sql_delete_in_source():
    """U-CHAT-AO-07: No SQL DELETE statement in message_repo source."""
    source = inspect.getsource(MessageRepo)
    assert "DELETE FROM chat_messages" not in source, (
        "APPEND-ONLY VIOLATION: SQL DELETE on chat_messages found in source"
    )
    assert "DELETE FROM read_receipts" not in source, (
        "APPEND-ONLY VIOLATION: SQL DELETE on read_receipts found in source"
    )


# -- Helpers ---------------------------------------------------------------


def _public_methods(cls: type) -> list[str]:
    """Return names of public (non-dunder) methods on *cls*."""
    return [
        name
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
