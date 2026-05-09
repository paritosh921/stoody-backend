"""
OCR API Router

Provides endpoints for OCR (Optical Character Recognition) operations.

Auth: Accepts SmartBoard cloud JWT via Authorization: Bearer header.
Tenant isolation enforced via JWT claims.
Feature gating: smartboard_cloud_access required.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.ocr_service import get_ocr_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])


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


async def _require_smartboard_auth(request: Request) -> dict:
    from api.v1.auth_async import get_current_user
    from core.tenant_features import is_feature_enabled
    from fastapi.security import HTTPBearer

    security = HTTPBearer()
    try:
        from main_async import app
        credentials = await security(request)
        user = await get_current_user(request, credentials, app.state.auth)
    except Exception:
        raise HTTPException(status_code=401, detail="SmartBoard JWT required")

    if not is_feature_enabled(
        user.get("enabled_features"),
        "smartboard_cloud_access",
        user.get("enabled_features_v2"),
    ):
        raise HTTPException(
            status_code=403,
            detail="Smartboard cloud access not enabled for this institution",
        )

    return user


@router.post("/analyze", response_model=OCRAnalyzeResponse)
async def analyze_image(
    request: Request,
    payload: OCRAnalyzeRequest,
):
    """
    Analyze an image and extract text content.

    Accepts a base64-encoded image and returns extracted text including:
    - Mathematical equations and formulas
    - Handwritten text
    - Printed text
    - Numbers and symbols
    """
    user = await _require_smartboard_auth(request)

    if not payload.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")

    # Validate image size (max ~10MB base64 = ~7.5MB image)
    if len(payload.image_b64) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    ocr_service = get_ocr_service()

    # Resolve tenant_db for LLM gate routing
    tenant_db = None
    try:
        from main_async import app
        db_manager = getattr(app.state, "db", None)
        db_name = user.get("db_name") or user.get("tenant_id")
        if db_manager and db_name:
            tenant_db = await db_manager.get_tenant_db(db_name)
    except Exception as exc:
        logger.warning("OCR tenant DB resolution failed: %s", exc)

    result = await ocr_service.analyze_image(
        image_b64=payload.image_b64,
        prompt=payload.prompt,
        tenant_db=tenant_db,
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
