from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

from models.raw_frames import RawFrameCanonical, Stroke


class Storage:
    """Simple JSONL storage for prototype deployments."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._locks = {}

    async def write_strokes(
        self,
        pen_id: str,
        session_id: str,
        strokes: List[Stroke],
        frame: RawFrameCanonical,
        stroke_payloads: List[Dict[str, Any]] | None = None,
    ) -> None:
        """Persist strokes for a pen/session combination."""

        if not strokes:
            return

        record: Dict[str, Any] = {
            "pen_id": pen_id,
            "session_id": session_id,
            "strokes": stroke_payloads
            if stroke_payloads is not None
            else [
                {
                    "page_no": stroke.page_no,
                    "book_type": stroke.book_type,
                    "start_ts": stroke.start_ts,
                    "end_ts": stroke.end_ts,
                    "points": [
                        {
                            "x": point.x,
                            "y": point.y,
                            "pressure": point.pressure,
                            "timestamp": point.timestamp,
                        }
                        for point in stroke.points
                    ],
                }
                for stroke in strokes
            ],
            "frame_metadata": frame.metadata,
            "ts_edge": frame.ts_edge,
            "ts_ingress": frame.ts_ingress,
        }

        path = self._data_dir / pen_id / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(self._append_jsonl, path, record)

    async def list_page_strokes(self, pen_id: str, page_no: int) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._read_page_strokes, pen_id, page_no)

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    def _session_files(self, pen_id: str) -> Iterable[Path]:
        pen_dir = self._data_dir / pen_id
        if not pen_dir.exists():
            return []
        return sorted(pen_dir.glob("*.jsonl"))

    def _read_page_strokes(self, pen_id: str, page_no: int) -> List[Dict[str, Any]]:
        strokes: List[Dict[str, Any]] = []
        for session_file in self._session_files(pen_id):
            try:
                with session_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        for stroke in record.get("strokes", []):
                            if stroke.get("page_no") != page_no:
                                continue

                            strokes.append(
                                {
                                    "id": stroke.get("id"),
                                    "points": stroke.get("points", []),
                                    "color": stroke.get("color", "#000000"),
                                    "strokeWidth": stroke.get("strokeWidth", 1.3),
                                    "timestamp": stroke.get("start_ts"),
                                }
                            )
            except FileNotFoundError:
                continue

        return strokes

