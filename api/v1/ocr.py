"""
OCR API Router

Provides endpoints for OCR (Optical Character Recognition) operations.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])


class OCRAnalyzeRequest(BaseModel):
    """Request body for OCR analysis."""
    image_b64: str  # Base64-encoded image (with or without data URI prefix)
    prompt: Optional[str] = None  # Custom prompt for analysis


class OCRAnalyzeResponse(BaseModel):
    """Response from OCR analysis."""
    success: bool
    text: str
    provider: Optional[str] = None
    error: Optional[str] = None


@router.post("/analyze", response_model=OCRAnalyzeResponse)
async def analyze_image(
    payload: OCRAnalyzeRequest,
    token: Optional[str] = None,
):
    """
    Analyze an image and extract text content.
    
    Accepts a base64-encoded image and returns extracted text including:
    - Mathematical equations and formulas
    - Handwritten text
    - Printed text
    - Numbers and symbols
    """
    from main_async import app
    state = app.state
    expected = state.dashboard_token
    if expected and token != expected:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    if not payload.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")
    
    # Validate image size (max ~10MB base64 = ~7.5MB image)
    if len(payload.image_b64) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    
    ocr_service = get_ocr_service()
    result = await ocr_service.analyze_image(
        image_b64=payload.image_b64,
        prompt=payload.prompt,
    )
    
    if not result["success"]:
        logger.warning(f"OCR analysis failed: {result.get('error')}")
    else:
        logger.info(f"OCR analysis successful via {result.get('provider')}")
    
    return OCRAnalyzeResponse(
        success=result["success"],
        text=result.get("text", ""),
        provider=result.get("provider"),
        error=result.get("error"),
    )


@router.get("/status")
async def ocr_status():
    """Check OCR service availability."""
    ocr_service = get_ocr_service()
    return {
        "mistral_available": ocr_service.mistral_available,
        "openai_available": ocr_service.openai_available,
    }
