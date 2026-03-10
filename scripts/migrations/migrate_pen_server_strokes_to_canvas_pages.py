"""
One-time migration: import historical pen-server stroke batches into the main
backend's `canvas_pages` collection.

Why this exists:
- Older Stoody pen history may still live only in the pen backend `strokes`
  collection from before page persistence was moved fully to main backend.
- The current live/runtime path intentionally no longer depends on pen-server
  history, so old pages must be migrated once instead of read at runtime.

What it does:
- Connects to MongoDB using env vars
- Reads pen-server stroke batches grouped by (user_id, book_type, page_number)
- Merges them into main-backend `canvas_pages`
- Preserves existing canvas_pages strokes and appends only missing legacy
  strokes by id

Environment:
- `MONGODB_URI`         Mongo connection string
- `MONGO_DB`            Main backend DB name
- `PEN_MONGO_DB_NAME`   Pen backend DB name

Usage:
- Dry run:
    python backend/scripts/migrations/migrate_pen_server_strokes_to_canvas_pages.py --dry-run
- Apply:
    python backend/scripts/migrations/migrate_pen_server_strokes_to_canvas_pages.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MAIN_DB_NAME = os.getenv("MONGO_DB", "skillbot_db")
PEN_DB_NAME = os.getenv("PEN_MONGO_DB_NAME", MAIN_DB_NAME)


def _stroke_id(stroke: Dict[str, Any]) -> str:
    sid = stroke.get("id") or stroke.get("strokeId")
    if sid:
        return str(sid)
    # Fallback should be stable enough for migration de-dupe within a page.
    return f"anon:{hash(str(stroke))}"


def _normalize_points(points: Any) -> List[Any]:
    if not isinstance(points, list):
        return []
    out: List[Any] = []
    for pt in points:
        if isinstance(pt, dict):
            arr = [pt.get("x", 0), pt.get("y", 0), pt.get("pressure", 0.5)]
            for extra in ("tiltX", "tiltY", "timestamp"):
                if extra in pt:
                    arr.append(pt[extra])
            out.append(arr)
        else:
            out.append(pt)
    return out


def _normalize_stroke(raw: Dict[str, Any], book_type: str, page_number: int) -> Dict[str, Any]:
    return {
        "id": str(raw.get("id") or raw.get("strokeId") or f"legacy:{hash(str(raw))}"),
        "points": _normalize_points(raw.get("points", [])),
        "strokeWidth": float(raw.get("strokeWidth", 1.3)),
        "color": raw.get("color", "#000000"),
        "tool": raw.get("tool", "pen"),
        "timestamp": raw.get("timestamp"),
        "svgPath": raw.get("svgPath"),
        "baseWidthMm": raw.get("baseWidthMm"),
        "sourceMode": raw.get("sourceMode"),
        "startedAt": raw.get("startedAt"),
        "endedAt": raw.get("endedAt"),
        "pageNumber": raw.get("pageNumber", page_number),
        "bookType": raw.get("bookType", book_type),
        "penMac": raw.get("penMac") or raw.get("pen_mac"),
    }


def _merge_unique(existing: Iterable[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = {_stroke_id(s): s for s in existing}
    ordered = list(existing)
    for stroke in incoming:
        sid = _stroke_id(stroke)
        if sid in seen:
            continue
        seen[sid] = stroke
        ordered.append(stroke)
    return ordered


async def migrate(apply_changes: bool) -> None:
    client = AsyncIOMotorClient(MONGO_URI)
    main_db = client[MAIN_DB_NAME]
    pen_db = client[PEN_DB_NAME]

    pen_strokes = pen_db["strokes"]
    canvas_pages = main_db["canvas_pages"]

    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)

    print(f"Mongo URI: {MONGO_URI}")
    print(f"Main DB:   {MAIN_DB_NAME}")
    print(f"Pen DB:    {PEN_DB_NAME}")
    print()

    cursor = pen_strokes.find({}).sort("timestamp", 1)
    async for doc in cursor:
        user_id = doc.get("user_id")
        book_type = str(doc.get("book_type") or "").upper()
        page_number = doc.get("page_number")
        if not user_id or not book_type or page_number is None:
            continue
        groups[(str(user_id), book_type, int(page_number))].append(doc)

    print(f"Discovered {len(groups)} user/book/page groups from pen-server strokes.")

    migrated = 0
    skipped = 0

    for (user_id, book_type, page_number), docs in groups.items():
        merged_legacy_strokes: List[Dict[str, Any]] = []
        first_ts: float | None = None
        last_ts: float | None = None
        session_id = None
        pen_mac = None
        page_style = None
        canvas_background = None

        for doc in docs:
            ts = doc.get("timestamp")
            if isinstance(ts, datetime):
                ts_ms = int(ts.timestamp() * 1000)
                first_ts = ts_ms if first_ts is None else min(first_ts, ts_ms)
                last_ts = ts_ms if last_ts is None else max(last_ts, ts_ms)
            if not session_id:
                session_id = doc.get("session_id")
            if not pen_mac:
                pen_mac = doc.get("pen_mac")
            if not page_style:
                page_style = doc.get("page_style")
            if not canvas_background:
                canvas_background = doc.get("canvas_background")

            for stroke in doc.get("strokes", []):
                merged_legacy_strokes.append(_normalize_stroke(stroke, book_type, page_number))

        if not merged_legacy_strokes:
            skipped += 1
            continue

        existing = await canvas_pages.find_one({
            "user_id": user_id,
            "book_type": book_type,
            "page_number": page_number,
        })

        existing_strokes = existing.get("strokes", []) if existing else []
        final_strokes = _merge_unique(existing_strokes, merged_legacy_strokes)

        if len(final_strokes) == len(existing_strokes):
            skipped += 1
            continue

        now = datetime.now(timezone.utc)
        base_doc = existing or {}
        doc = {
            "user_id": user_id,
            "admin_id": base_doc.get("admin_id"),
            "book_type": book_type,
            "page_number": page_number,
            "strokes": final_strokes,
            "page_style": base_doc.get("page_style") or page_style,
            "canvas_background": base_doc.get("canvas_background") or canvas_background,
            "stroke_count": len(final_strokes),
            "pen_mac": base_doc.get("pen_mac") or pen_mac,
            "source": base_doc.get("source") or "pen_server_migration",
            "last_modified": now,
            "client_last_modified": base_doc.get("client_last_modified") or last_ts,
            "version": int(base_doc.get("version", 0)) + 1,
            "session_id": base_doc.get("session_id") or session_id,
            "first_activity": base_doc.get("first_activity") or first_ts,
            "last_activity": max(base_doc.get("last_activity") or 0, last_ts or 0) or None,
        }

        if apply_changes:
            await canvas_pages.replace_one(
                {
                    "user_id": user_id,
                    "book_type": book_type,
                    "page_number": page_number,
                },
                doc,
                upsert=True,
            )
        migrated += 1
        print(
            f"{'Would migrate' if not apply_changes else 'Migrated'} "
            f"user={user_id} book={book_type} page={page_number} "
            f"legacy={len(merged_legacy_strokes)} existing={len(existing_strokes)} final={len(final_strokes)}"
        )

    print()
    print(f"{'Dry-run' if not apply_changes else 'Migration'} complete.")
    print(f"Migrated groups: {migrated}")
    print(f"Skipped groups:  {skipped}")

    client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply writes to main canvas_pages")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    args = parser.parse_args()

    apply_changes = bool(args.apply)
    if not args.apply and not args.dry_run:
        parser.error("Choose one: --dry-run or --apply")

    asyncio.run(migrate(apply_changes=apply_changes))


if __name__ == "__main__":
    main()
