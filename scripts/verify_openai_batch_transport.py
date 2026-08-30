"""Verify OpenAI Files -> Batch handoff without spending model tokens.

This probe submits one moderation request through Batch, waits for validation,
and removes every provider file it creates. It is intended for deployment and
credential checks before teachers are allowed to start Economy checking.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from services.exampen_openai_batch import (  # noqa: E402
    OpenAIBatchClient,
    provider_batch_failure,
)


TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _line() -> bytes:
    return (
        json.dumps(
            {
                "custom_id": "stoody-economy-transport-preflight",
                "method": "POST",
                "url": "/v1/moderations",
                "body": {
                    "model": "omni-moderation-latest",
                    "input": "Stoody Economy Batch transport preflight.",
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


async def run() -> int:
    client = OpenAIBatchClient()
    input_file_id = ""
    provider_batch_id = ""
    provider_state: Mapping[str, Any] = {}
    cleanup_file_ids: set[str] = set()
    try:
        uploaded = await client.upload_jsonl("stoody-batch-preflight.jsonl", _line())
        input_file_id = str(uploaded.get("id") or "")
        cleanup_file_ids.add(input_file_id)
        await client.wait_for_file_ready(input_file_id)
        created = await client.create_batch(
            input_file_id=input_file_id,
            endpoint="/v1/moderations",
            metadata={"stoody_probe": "economy_transport"},
        )
        provider_batch_id = str(created.get("id") or "")
        provider_state = created
        for _ in range(36):
            status = str(provider_state.get("status") or "")
            if status in TERMINAL:
                break
            await asyncio.sleep(5)
            provider_state = await client.retrieve_batch(provider_batch_id)
        else:
            await client.cancel_batch(provider_batch_id)
            print("FAIL: OpenAI Batch preflight did not finish validation within 3 minutes.")
            return 2

        for field in ("output_file_id", "error_file_id"):
            file_id = str(provider_state.get(field) or "")
            if file_id:
                cleanup_file_ids.add(file_id)
        if str(provider_state.get("status") or "") != "completed":
            reason = provider_batch_failure(provider_state) or "OpenAI Batch preflight failed"
            print(f"FAIL: {reason}")
            return 2
        print("PASS: OpenAI Files and Batch are accessible in the same project scope.")
        return 0
    except Exception as exc:
        print(f"FAIL: {str(exc).replace(chr(10), ' ')[:600]}")
        return 2
    finally:
        for file_id in cleanup_file_ids:
            if file_id:
                await client.delete_file(file_id)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
