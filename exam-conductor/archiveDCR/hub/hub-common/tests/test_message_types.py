"""Tests for IPC message type constants and payload dataclass validation.

Test IDs: U-MSG-01 .. U-MSG-10
Validation level: L3 (unit, no I/O)
"""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import pytest

from hub_common import message_types as mt


# -------------------------------------------------------------------
# U-MSG-01: All constant strings are namespaced
# -------------------------------------------------------------------

# Collect every module-level str constant that looks like a message type.
_MSG_TYPE_CONSTANTS: list[tuple[str, str]] = [
    (name, getattr(mt, name))
    for name in dir(mt)
    if name.isupper() and isinstance(getattr(mt, name), str)
]


class TestMessageTypeConstants:
    @pytest.mark.parametrize("name,value", _MSG_TYPE_CONSTANTS)
    def test_namespaced_format(self, name: str, value: str) -> None:
        """U-MSG-01: every constant has at least two dot-separated segments."""
        parts = value.split(".")
        assert len(parts) >= 2, f"{name} = {value!r} is not namespaced"

    def test_no_duplicate_values(self) -> None:
        """U-MSG-02: no two constants share the same string value."""
        values = [v for _, v in _MSG_TYPE_CONSTANTS]
        assert len(values) == len(set(values)), "Duplicate message type values found"


# -------------------------------------------------------------------
# U-MSG-03..U-MSG-10: Payload dataclass validation
# -------------------------------------------------------------------

# All exported dataclasses whose name ends with "Payload".
_PAYLOAD_CLASSES: list[type] = [
    getattr(mt, name)
    for name in dir(mt)
    if name.endswith("Payload") and dataclasses.is_dataclass(getattr(mt, name))
]


class TestPayloadDataclasses:
    @pytest.mark.parametrize("cls", _PAYLOAD_CLASSES, ids=lambda c: c.__name__)
    def test_has_type_hints(self, cls: type) -> None:
        """U-MSG-03: every payload field has a type annotation."""
        hints = get_type_hints(cls)
        fields = dataclasses.fields(cls)  # type: ignore[arg-type]
        for f in fields:
            assert f.name in hints, f"{cls.__name__}.{f.name} missing type hint"

    @pytest.mark.parametrize("cls", _PAYLOAD_CLASSES, ids=lambda c: c.__name__)
    def test_uses_slots(self, cls: type) -> None:
        """U-MSG-04: payload classes use __slots__ for memory efficiency."""
        assert hasattr(cls, "__slots__"), f"{cls.__name__} missing __slots__"

    def test_store_write_request_fields(self) -> None:
        """U-MSG-05: StoreWriteRequestPayload has all required fields."""
        p = mt.StoreWriteRequestPayload(
            exam_id="E1",
            pen_mac="AA:BB:CC:DD:EE:FF",
            chunk_index=0,
            chunk_b64="AAAA",
            checksum_crc32="deadbeef",
        )
        assert p.exam_id == "E1"
        assert p.chunk_index == 0

    def test_store_write_result_fields(self) -> None:
        """U-MSG-06: StoreWriteResultPayload contains persistence flags."""
        p = mt.StoreWriteResultPayload(
            exam_id="E1",
            pen_mac="AA:BB",
            chunk_index=0,
            sd_persisted=True,
            usb_persisted=False,
        )
        assert p.sd_persisted is True
        assert p.usb_persisted is False

    def test_timer_arm_request_fields(self) -> None:
        """U-MSG-07: TimerArmRequestPayload contains duration."""
        p = mt.TimerArmRequestPayload(
            exam_id="E1", duration_sec=3600, armed_by="invig-001"
        )
        assert p.duration_sec == 3600

    def test_pen_sync_complete_fields(self) -> None:
        """U-MSG-08: PenSyncCompleteEventPayload captures status."""
        p = mt.PenSyncCompleteEventPayload(
            exam_id="E1",
            pen_mac="AA:BB",
            total_chunks=5,
            checksum_crc32="abcd1234",
            status="complete",
        )
        assert p.status == "complete"

    def test_error_payload(self) -> None:
        """U-MSG-09: ErrorPayload carries code and message."""
        p = mt.ErrorPayload(code="timeout", message="No reply in 10s")
        assert p.code == "timeout"

    def test_fsm_transition_request_all_fields(self) -> None:
        """U-MSG-10: FsmTransitionRequestPayload is fully typed."""
        p = mt.FsmTransitionRequestPayload(
            exam_id="E1",
            from_state="created",
            to_state="armed",
            reason="invigilator command",
            actor="invig-001",
        )
        assert p.to_state == "armed"
