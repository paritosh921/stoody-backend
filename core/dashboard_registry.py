from __future__ import annotations

import asyncio
from typing import Dict, List, Any, Set


class PenDashboardRegistry:
    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}
        self._strokes: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._stroke_ids: Dict[str, Set[str]] = {}

    async def update_status(self, pen_id: str, **updates) -> Dict[str, Any]:
        async with self._lock:
            state = self._states.setdefault(pen_id, {})
            state.update(updates)
            return state.copy()

    async def append_strokes(self, pen_id: str, strokes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async with self._lock:
            existing = self._strokes.setdefault(pen_id, [])
            id_set = self._stroke_ids.setdefault(pen_id, set())
            for stroke in strokes:
                stroke_id = stroke.get("id")
                if stroke_id and stroke_id in id_set:
                    continue
                existing.append(stroke)
                if stroke_id:
                    id_set.add(stroke_id)
            if len(existing) > 400:
                self._strokes[pen_id] = existing[-400:]
            return list(self._strokes[pen_id])

    async def get_pen_state(self, pen_id: str) -> Dict[str, Any]:
        async with self._lock:
            state = self._states.get(pen_id, {}).copy()
            state["strokes"] = list(self._strokes.get(pen_id, []))
            state["pen_id"] = pen_id
            return state

    async def list_pen_states(self) -> List[Dict[str, Any]]:
        async with self._lock:
            results = []
            for pen_id, state in self._states.items():
                entry = state.copy()
                entry["pen_id"] = pen_id
                entry["strokes"] = list(self._strokes.get(pen_id, []))
                results.append(entry)
            return results

    async def clear_strokes(self, pen_id: str) -> None:
        async with self._lock:
            self._strokes[pen_id] = []
            self._stroke_ids[pen_id] = set()

