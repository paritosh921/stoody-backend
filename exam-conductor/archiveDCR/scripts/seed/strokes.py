"""Generate binary stroke data in P05 14-byte coordinate frame format.

Reference: CLAUDE.md - Pen & BLE Protocol section.
Coordinate frame: bookType, bookSeq, pageNo, coordX, coordY,
                  pressure, penProp, padding, timestamp
"""

from __future__ import annotations

import math
import random
import struct

from .constants import COORD_FRAME_FORMAT, PAGE_HEIGHT_PU, PAGE_WIDTH_PU


def gen_stroke_data(
    rng: random.Random,
    student: dict,
    exam: dict,
    question_num: int,
) -> bytes:
    """Generate binary stroke data simulating pen handwriting output.

    Produces 3-8 strokes per question, each with 15-80 coordinate points.
    Uses the P05 14-byte frame format.
    """
    frames: list[bytes] = []
    bbox = exam["questions"][question_num - 1]["region_bbox"]
    num_strokes = rng.randint(3, 8)
    ts = rng.randint(1000, 60000)

    for _ in range(num_strokes):
        sx = bbox["x"] + rng.randint(20, bbox["width"] - 20)
        sy = bbox["y"] + rng.randint(20, bbox["height"] - 20)
        points = rng.randint(15, 80)

        for p in range(points):
            angle = rng.uniform(0, 2 * math.pi)
            dx = int(3 * math.cos(angle + p * 0.1))
            dy = int(3 * math.sin(angle + p * 0.1))
            x = max(0, min(PAGE_WIDTH_PU, sx + dx * p))
            y = max(0, min(PAGE_HEIGHT_PU, sy + dy * p))
            pressure = rng.randint(100, 800) if p > 0 else 0
            pen_prop = 0x01 if p > 0 else 0x00

            frame = struct.pack(
                COORD_FRAME_FORMAT,
                0x01, 0x01, question_num,
                x, y, pressure, pen_prop, 0x00,
                ts & 0xFFFF,
            )
            frames.append(frame)
            ts += rng.randint(5, 25)

        # Pen-up frame
        frames.append(struct.pack(
            COORD_FRAME_FORMAT,
            0x01, 0x01, question_num,
            x, y, 0, 0x02, 0x00, ts & 0xFFFF,
        ))
        ts += rng.randint(200, 1000)

    return b"".join(frames)
