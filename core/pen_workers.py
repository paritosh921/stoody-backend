from __future__ import annotations

import asyncio
import logging
import heapq
import itertools
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Optional, List

from models.raw_frames import RawFrameCanonical, Stroke

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
def _get_question_attempt_funcs():
    from api.v1.question_attempts import get_active_attempt_for_pen, add_stroke_to_attempt
    return get_active_attempt_for_pen, add_stroke_to_attempt


class PenWorkers:
    """Dispatches canonical frames to per-pen async workers."""

    def __init__(
        self,
        stroke_adapter,
        storage,
        *,
        queue_size: int = 512,
        dashboard_callback: Optional[
            Callable[[str, str, dict], Awaitable[None]]
        ] = None,
        max_pens: int = 6,
    ) -> None:
        self._stroke_adapter = stroke_adapter
        self._storage = storage
        self._queue_size = queue_size
        self.max_pens = max_pens
        self._dashboard_callback = dashboard_callback

        self._queues: Dict[str, asyncio.Queue[RawFrameCanonical]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._states: Dict[str, object] = {}
        self._pen_meta: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._metrics = defaultdict(int)
        self._stroke_counters: Dict[str, int] = {}
        self._buffers: Dict[str, list] = defaultdict(list)
        self._buffer_order = itertools.count()
        self._last_serial: Dict[str, Optional[int]] = {}
        self._last_sessions: Dict[str, str] = {}
        self._max_reorder_gap = 8
        self._max_buffer = queue_size

    async def enqueue(self, frame: RawFrameCanonical) -> None:
        self._metrics["frames_enqueued"] += 1

        self._pen_meta.setdefault(frame.pen_id, {})
        pen_meta = self._pen_meta[frame.pen_id]
        pen_meta["connected"] = True
        pen_meta["last_frame_ts"] = frame.ts_ingress
        pen_meta["hub_id"] = frame.hub_id
        pen_meta["pen_mac"] = frame.metadata.get("pen_mac", "")
        pen_meta.setdefault("color", "#000000")

        if "page_no" in frame.metadata:
            pen_meta["page_no"] = frame.metadata["page_no"]
        if "book_type" in frame.metadata:
            pen_meta["book_type"] = frame.metadata["book_type"]
        if "battery_pct" in frame.metadata:
            pen_meta["battery"] = frame.metadata["battery_pct"]

        if self._dashboard_callback:
            await self._dashboard_callback(
                "pen_status",
                frame.pen_id,
                {
                    "pen_mac": pen_meta["pen_mac"],
                    "connected": True,
                    "hub_id": pen_meta["hub_id"],
                    "timestamp": frame.ts_ingress,
                    "battery": pen_meta.get("battery"),
                    "page_no": pen_meta.get("page_no"),
                    "book_type": pen_meta.get("book_type"),
                    "color": pen_meta.get("color"),
                },
            )

        await self._buffer_frame(frame)

    async def _ensure_queue(self, pen_id: str) -> asyncio.Queue[RawFrameCanonical]:
        async with self._lock:
            if pen_id in self._queues:
                return self._queues[pen_id]

            queue: asyncio.Queue[RawFrameCanonical] = asyncio.Queue(
                maxsize=self._queue_size
            )
            self._queues[pen_id] = queue

            task = asyncio.create_task(self._worker_loop(pen_id, queue))
            self._tasks[pen_id] = task
            logger.info("Started pen worker for %s", pen_id)

            return queue

    async def _buffer_frame(self, frame: RawFrameCanonical) -> None:
        pen_id = frame.pen_id
        session = self._last_sessions.get(pen_id)
        if session != frame.session_id:
            self._last_sessions[pen_id] = frame.session_id
            self._last_serial[pen_id] = None
            self._buffers[pen_id].clear()

        serial = frame.seq
        buffer = self._buffers[pen_id]
        heapq.heappush(buffer, (serial, next(self._buffer_order), frame))
        await self._flush_ready_frames(pen_id)

    async def _flush_ready_frames(self, pen_id: str) -> None:
        buffer = self._buffers[pen_id]
        if not buffer:
            return

        queue = await self._ensure_queue(pen_id)
        while buffer:
            serial, _, frame = buffer[0]
            last_serial = self._last_serial.get(pen_id)

            if last_serial is None or serial <= last_serial:
                heapq.heappop(buffer)
                if last_serial == serial:
                    continue
                self._last_serial[pen_id] = serial
                await queue.put(frame)
                continue

            if serial == last_serial + 1:
                heapq.heappop(buffer)
                self._last_serial[pen_id] = serial
                await queue.put(frame)
                continue

            if len(buffer) > self._max_buffer or serial - last_serial > self._max_reorder_gap:
                logger.warning(
                    "Reordering window exceeded for pen %s (serial=%s last=%s). Flushing frame.",
                    pen_id,
                    serial,
                    last_serial,
                )
                heapq.heappop(buffer)
                self._last_serial[pen_id] = serial
                await queue.put(frame)
                continue

            break

    async def _worker_loop(
        self, pen_id: str, queue: asyncio.Queue[RawFrameCanonical]
    ) -> None:
        state = self._stroke_adapter.create_state(pen_id)
        self._states[pen_id] = state
        last_seq = -1
        last_session = None
        last_page = None
        last_book_type = None

        while True:
            frame = await queue.get()

            try:
                if last_session is None:
                    last_session = frame.session_id
                if frame.session_id != last_session:
                    last_session = frame.session_id
                    last_seq = -1
                    self._pen_meta.setdefault(pen_id, {})["session_id"] = frame.session_id
                    if self._dashboard_callback:
                        await self._dashboard_callback(
                            "pen_clear",
                            pen_id,
                            {
                                "reason": "session_change",
                                "new_session": frame.session_id,
                                "timestamp": frame.ts_ingress,
                            },
                        )

                if frame.seq <= last_seq:
                    logger.warning(
                        "Dropping out-of-order frame pen=%s seq=%s last=%s",
                        pen_id,
                        frame.seq,
                        last_seq,
                    )
                    continue
                last_seq = frame.seq

                coords = self._stroke_adapter._parse_coordinates(frame.payload)  # type: ignore[attr-defined]
                page_changed = False
                for coord in coords:
                    if coord.page_no is not None and coord.page_no != last_page:
                        last_page = coord.page_no
                        self._pen_meta.setdefault(pen_id, {})["page_no"] = coord.page_no
                        page_changed = True
                        if self._dashboard_callback:
                            await self._dashboard_callback(
                                "pen_status",
                                pen_id,
                                {
                                    "page_no": coord.page_no,
                                    "timestamp": frame.ts_ingress,
                                },
                            )
                    if coord.book_type_name and coord.book_type_name != last_book_type:
                        last_book_type = coord.book_type_name
                        self._pen_meta.setdefault(pen_id, {})["book_type"] = coord.book_type_name
                        if self._dashboard_callback:
                            await self._dashboard_callback(
                                "pen_status",
                                pen_id,
                                {
                                    "book_type": coord.book_type_name,
                                    "timestamp": frame.ts_ingress,
                                },
                            )

                if page_changed and self._dashboard_callback:
                    await self._dashboard_callback(
                        "pen_clear",
                        pen_id,
                        {
                            "reason": "page_change",
                            "new_page": last_page,
                            "timestamp": frame.ts_ingress,
                        },
                    )

                strokes, state = self._stroke_adapter.process_frame(state, frame)
                self._states[pen_id] = state

                if strokes:
                    last_stroke = strokes[-1]
                    if last_stroke.page_no is not None:
                        self._pen_meta.setdefault(pen_id, {})["page_no"] = last_stroke.page_no
                    if last_stroke.book_type:
                        self._pen_meta.setdefault(pen_id, {})["book_type"] = last_stroke.book_type

                    color = self._pen_meta.setdefault(pen_id, {}).get("color", "#000000")
                    storage_payloads: List[Dict[str, Any]] = []
                    strokes_payload = []
                    for stroke in strokes:
                        counter = self._stroke_counters.get(pen_id, 0) + 1
                        self._stroke_counters[pen_id] = counter
                        stroke_id = f"{frame.session_id}:{frame.seq}:{counter}"
                        point_payload = [
                            {"x": point.x, "y": point.y, "pressure": point.pressure}
                            for point in stroke.points
                        ]
                        payload = {
                            "id": stroke_id,
                            "points": point_payload,
                            "timestamp": getattr(stroke, "start_ts", None),
                            "strokeWidth": getattr(stroke, "stroke_width", 1.3),
                            "color": color,
                        }
                        strokes_payload.append(payload)
                        storage_payloads.append(
                            {
                                **payload,
                                "page_no": stroke.page_no,
                                "book_type": stroke.book_type,
                                "start_ts": stroke.start_ts,
                                "end_ts": stroke.end_ts,
                            }
                        )
                        
                        # Tag stroke with question attempt if pen is in active lock
                        try:
                            get_active_attempt, add_stroke = _get_question_attempt_funcs()
                            attempt_id = get_active_attempt(pen_id)
                            if attempt_id:
                                page_no = stroke.page_no if stroke.page_no is not None else 0
                                add_stroke(attempt_id, payload, pen_id, page_no)
                        except Exception as e:
                            logger.debug(f"Could not tag stroke with attempt: {e}")

                    await self._storage.write_strokes(
                        pen_id=pen_id,
                        session_id=frame.session_id,
                        strokes=strokes,
                        frame=frame,
                        stroke_payloads=storage_payloads,
                    )
                    self._metrics["strokes_written"] += len(strokes)

                    if self._dashboard_callback:
                        await self._dashboard_callback(
                            "pen_strokes",
                            pen_id,
                            {
                                "strokes": strokes_payload,
                                "book_type": self._pen_meta.get(pen_id, {}).get("book_type"),
                                "page_no": self._pen_meta.get(pen_id, {}).get("page_no"),
                            },
                        )

            except Exception as exc:
                logger.exception("Worker error for pen %s: %s", pen_id, exc)
            finally:
                queue.task_done()

    def stats(self) -> Dict[str, int]:
        """Expose lightweight metrics for admin endpoints."""
        return dict(self._metrics)

    def iter_pen_states(self):
        for pen_id, meta in self._pen_meta.items():
            yield pen_id, meta.copy()

    async def set_pen_color(self, pen_id: str, color: str) -> None:
        pen_meta = self._pen_meta.setdefault(pen_id, {})
        pen_meta["color"] = color
        if self._dashboard_callback:
            await self._dashboard_callback(
                "pen_status",
                pen_id,
                {
                    "color": color,
                    "pen_mac": pen_meta.get("pen_mac", ""),
                    "connected": pen_meta.get("connected", False),
                    "hub_id": pen_meta.get("hub_id"),
                    "battery": pen_meta.get("battery"),
                    "page_no": pen_meta.get("page_no"),
                    "book_type": pen_meta.get("book_type"),
                    "timestamp": pen_meta.get("last_frame_ts"),
                },
            )

