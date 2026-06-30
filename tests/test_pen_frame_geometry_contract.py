import struct

import pytest

from api.v1.pen_frames import _ingest_single_frame
from core.stroke_engine_adapter import StrokeEngineAdapter
from models.raw_frames import RawFrameCanonical, RawFrameIn


COORD = struct.Struct("<BBHHHBBL")


def coord_payload(
    *,
    x: int,
    y: int,
    pressure: int,
    pen_prop: int,
    timestamp: int,
    page_no: int = 4,
    book_type: int = 0x05,
) -> bytes:
    return COORD.pack(book_type, 1, page_no, x, y, pressure, pen_prop, timestamp)


def test_stroke_engine_adapter_preserves_coordinate_geometry_without_repair():
    adapter = StrokeEngineAdapter()
    state = adapter.create_state("pen-1")

    down_1 = RawFrameCanonical(
        pen_id="pen-1",
        session_id="session-1",
        seq=1,
        payload=coord_payload(x=65535, y=0, pressure=255, pen_prop=1, timestamp=1000),
    )
    down_2 = RawFrameCanonical(
        pen_id="pen-1",
        session_id="session-1",
        seq=2,
        payload=coord_payload(x=13, y=65534, pressure=127, pen_prop=1, timestamp=1016),
    )
    up = RawFrameCanonical(
        pen_id="pen-1",
        session_id="session-1",
        seq=3,
        payload=coord_payload(x=99, y=99, pressure=0, pen_prop=2, timestamp=1032),
    )

    strokes, state = adapter.process_frame(state, down_1)
    assert strokes == []
    strokes, state = adapter.process_frame(state, down_2)
    assert strokes == []
    strokes, _state = adapter.process_frame(state, up)

    assert len(strokes) == 1
    assert [(point.x, point.y, point.pressure, point.timestamp) for point in strokes[0].points] == [
        (65535, 0, 255, 1000),
        (13, 65534, 127, 1016),
    ]


@pytest.mark.asyncio
async def test_pen_frame_ingest_enqueues_raw_payload_without_geometry_repair():
    raw_payload = coord_payload(x=4321, y=1234, pressure=88, pen_prop=1, timestamp=2000)
    frame = RawFrameIn(
        hub_id="hub-1",
        pen_mac="aa:bb:cc:dd:ee:ff",
        session_id="session-2",
        seq=9,
        payload_hex=raw_payload.hex(),
        ts_edge=12.5,
        metadata={"page_no": 4, "book_type": "MS"},
    )

    class Router:
        async def resolve_pen_id(self, pen_mac: str) -> str:
            assert pen_mac == "AA:BB:CC:DD:EE:FF"
            return "pen-99"

    class Workers:
        def __init__(self) -> None:
            self.enqueued = []

        async def enqueue(self, canonical: RawFrameCanonical) -> None:
            self.enqueued.append(canonical)

    class State:
        def __init__(self) -> None:
            self.pen_router = Router()
            self.pen_workers = Workers()

    state = State()

    await _ingest_single_frame(frame, state)

    assert len(state.pen_workers.enqueued) == 1
    canonical = state.pen_workers.enqueued[0]
    assert canonical.pen_id == "pen-99"
    assert canonical.payload == raw_payload
    assert canonical.seq == 9
    assert canonical.hub_id == "hub-1"
    assert canonical.metadata == {
        "page_no": 4,
        "book_type": "MS",
        "pen_mac": "AA:BB:CC:DD:EE:FF",
    }

