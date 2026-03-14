"""
Bug #4 — Live canvas notes don't appear in the My Copy page list.

The copies list endpoint merges canvas_pages into the strokes-based list.
_list_canvas_pages_for_user queries with {"stroke_count": {"$gt": 0}}.

If the canvas page has stroke_count=0 or stroke_count=None despite having
strokes, it won't appear. Also, if the frontend sends stroke_count as None
and the strokes array is large, the validator computes len(strokes), which
should be correct. But there may be an edge case.

This test validates the Pydantic model and the query logic.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_canvas_page_upsert_stroke_count_defaults_to_strokes_length():
    """
    Ensure that when stroke_count is None, the validator sets it to len(strokes).
    """
    from api.v1.strokes_async import CanvasPageUpsert

    page = CanvasPageUpsert(
        book_type="MS",
        page_number=0,
        strokes=[
            {"id": "s1", "points": [[0, 0, 0.5]], "strokeWidth": 2, "color": "#000", "tool": "pen"},
            {"id": "s2", "points": [[1, 1, 0.5]], "strokeWidth": 2, "color": "#000", "tool": "pen"},
        ],
        stroke_count=None,  # Frontend might send None
    )

    assert page.stroke_count == 2, (
        f"BUG: stroke_count should default to len(strokes)=2, got {page.stroke_count}"
    )


def test_canvas_page_upsert_stroke_count_zero_with_strokes():
    """
    Edge case: frontend sends stroke_count=0 but strokes array has items.
    The validator should correct this to len(strokes).
    """
    from api.v1.strokes_async import CanvasPageUpsert

    page = CanvasPageUpsert(
        book_type="MS",
        page_number=0,
        strokes=[
            {"id": "s1", "points": [[0, 0, 0.5]], "strokeWidth": 2, "color": "#000", "tool": "pen"},
        ],
        stroke_count=0,  # Explicit zero despite having strokes
    )

    # stroke_count=0 means the page won't appear in the copies list
    # (query has stroke_count > 0). The validator should correct this.
    assert page.stroke_count > 0, (
        f"BUG REPRODUCED: stroke_count is {page.stroke_count} despite having "
        f"{len(page.strokes)} strokes. This page will be invisible in My Copy list."
    )


def test_list_canvas_pages_query_includes_stroke_count_filter():
    """
    Verify that _list_canvas_pages_for_user uses stroke_count > 0 filter.
    This is intentional but means canvas_pages with stroke_count=0 are hidden.
    """
    import ast

    copies_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "v1", "copies_async.py"
    )
    copies_path = os.path.normpath(copies_path)

    with open(copies_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Find _list_canvas_pages_for_user function
    func_start = source.find("async def _list_canvas_pages_for_user")
    assert func_start != -1

    # Find the next function after it
    func_end = source.find("\nasync def ", func_start + 10)
    if func_end == -1:
        func_end = len(source)
    func_body = source[func_start:func_end]

    has_stroke_count_filter = '"stroke_count"' in func_body or "'stroke_count'" in func_body
    assert has_stroke_count_filter, (
        "_list_canvas_pages_for_user doesn't filter by stroke_count. "
        "This is fine, but means empty pages would show up."
    )

    # If there IS a stroke_count filter, validate that the upsert always sets
    # a correct stroke_count. The real bug is the validator not correcting 0.
    # This test documents the dependency.


def test_frontend_flush_sends_stroke_count():
    """
    Verify the frontend canvasSync.ts sends stroke_count: data.strokes.length
    when flushing pages to the server.
    """
    sync_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "frontend", "src", "services", "stoody", "canvasSync.ts",
    )
    sync_path = os.path.normpath(sync_path)

    assert os.path.exists(sync_path), f"File not found: {sync_path}"

    with open(sync_path, "r", encoding="utf-8") as f:
        source = f.read()

    # The flush function should send stroke_count derived from strokes array
    assert "stroke_count: data.strokes.length" in source, (
        "BUG: canvasSync.ts does not send stroke_count: data.strokes.length. "
        "The server may store stroke_count=0 or None, hiding the page from My Copy."
    )


def test_canvas_page_pen_mac_value_for_live_canvas():
    """
    Live canvas pages have pen_mac=null or "canvas". The copies list query
    in _list_canvas_pages_for_user applies pen_mac filter only when the
    caller passes a pen_mac. Verify this doesn't filter out live canvas pages.
    """
    copies_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "v1", "copies_async.py"
    )
    copies_path = os.path.normpath(copies_path)

    with open(copies_path, "r", encoding="utf-8") as f:
        source = f.read()

    func_start = source.find("async def _list_canvas_pages_for_user")
    func_end = source.find("\nasync def ", func_start + 10)
    if func_end == -1:
        func_end = len(source)
    func_body = source[func_start:func_end]

    # pen_mac should be conditional: only added to query if caller provides it
    has_conditional_pen_mac = 'if pen_mac:' in func_body
    assert has_conditional_pen_mac, (
        "BUG: pen_mac is always included in query, which would filter out "
        "live canvas pages that have pen_mac=null or 'canvas'."
    )
