"""
Async Mistral OCR client for PDF processing.
"""

import logging
import os
import asyncio
from typing import Any, Dict

from fastapi import HTTPException, status

from config_async import OCR_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"


async def call_mistral_ocr(pdf_base64: str) -> Dict[str, Any]:
    """Call Mistral OCR API with base64 PDF data."""
    import aiohttp

    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY is not configured in environment variables")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Mistral API key is not configured. Please set MISTRAL_API_KEY in environment variables."
        )

    try:
        document_url = f"data:application/pdf;base64,{pdf_base64}"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": document_url
            },
            "include_image_base64": True
        }

        logger.info("Calling Mistral OCR API (PDF size: %s chars)", len(pdf_base64))

        async with aiohttp.ClientSession(trace_configs=[]) as session:
            async with session.post(
                MISTRAL_OCR_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=OCR_TIMEOUT_SECONDS)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("Mistral OCR API error: %s - %s", response.status, error_text)
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Mistral OCR API error: {error_text}"
                    )

                return await response.json()

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Mistral OCR API timeout")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="OCR processing timeout"
        )
    except aiohttp.ClientError as exc:
        logger.error("Mistral OCR API client error: %s - %s", type(exc).__name__, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR API connection error: {str(exc)}"
        )
    except Exception as exc:
        logger.error("Mistral OCR API unexpected error: %s - %s", type(exc).__name__, str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(exc)}"
        )
