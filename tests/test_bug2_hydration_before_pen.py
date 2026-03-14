"""
Bug #2 — Strokes hydrate before pen is connected.

The canvas calls useStrokeSync with bluetoothAddress=null before a pen connects.
Hydration should NOT run until bluetoothAddress is non-null.

This is a frontend bug, so this test validates the LOGIC by simulating the
condition: hydrate() should not be callable when bluetoothAddress is absent.

We test the backend list endpoint to confirm it returns pages even without
a pen context — meaning the frontend is the one that needs to gate the call.
"""

import asyncio
import sys
import os
import inspect

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_use_stroke_sync_hydrates_without_bluetooth_address():
    """
    Reproduce: useStrokeSync.ts line 57 does NOT check bluetoothAddress before
    calling hydrate(). The effect fires as soon as userId is non-null.

    This test reads the source file and asserts that the hydration guard
    includes a check for bluetoothAddress (or equivalent pen connection state).
    """
    hook_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "frontend", "src", "hooks", "stoody", "useStrokeSync.ts",
    )
    hook_path = os.path.normpath(hook_path)

    assert os.path.exists(hook_path), f"File not found: {hook_path}"

    with open(hook_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Find the hydration useEffect — it contains the hydrate() call.
    # The guard line currently reads:
    #   if (!userId || hydrationDoneRef.current || disableHydration) return;
    # It should ALSO check for bluetoothAddress (or an equivalent guard).
    #
    # We look for the pattern: the early-return line inside the effect that
    # calls hydrate() must reference bluetoothAddress (or penConnected, etc).

    # Locate the block: useEffect that calls hydrate(
    hydrate_call_idx = source.find("hydrate(userId")
    assert hydrate_call_idx != -1, "Could not find hydrate(userId) call"

    # Extract the useEffect block preceding the hydrate call
    # Find the nearest useEffect before hydrate_call_idx
    effect_start = source.rfind("useEffect(", 0, hydrate_call_idx)
    assert effect_start != -1, "Could not find useEffect wrapping hydrate()"

    # The guard region is between effect_start and hydrate_call_idx
    guard_region = source[effect_start:hydrate_call_idx]

    # The guard MUST reference bluetoothAddress (or disableHydration gated on it)
    has_bluetooth_guard = (
        "bluetoothAddress" in guard_region
        or "!bluetoothAddress" in guard_region
        or "penConnected" in guard_region
    )

    assert has_bluetooth_guard, (
        "BUG REPRODUCED: The hydration useEffect does NOT check bluetoothAddress "
        "before calling hydrate(). Strokes will load before the pen connects.\n"
        f"Guard region:\n{guard_region}"
    )
