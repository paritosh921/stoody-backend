"""
GDPR consent endpoints for B2C users.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.b2c_auth_dependencies import get_current_b2c_user, get_database
from api.v1.b2c_auth_schemas import AdultConsentRequest, ParentalConsentRequest
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/consent/adult")
@limiter.limit("5/minute")
async def submit_adult_consent(
    request: Request,
    consent_data: AdultConsentRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Submit GDPR consent for adult users (16+).

    - Records user's consent for privacy policy and AI processing
    - Stores consent timestamp and version for audit trail
    - Logs IP address for compliance
    """
    try:
        user_id = current_user.get("user_id")

        if not consent_data.gdpr_consent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Privacy policy consent is required",
            )
        if not consent_data.ai_personalization_consent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="AI personalization consent is required for our service",
            )

        consent_record = {
            "consent_completed": True,
            "is_minor": False,
            "has_parental_consent": False,
            "gdpr_consent": consent_data.gdpr_consent,
            "ai_personalization_consent": consent_data.ai_personalization_consent,
            "marketing_consent": consent_data.marketing_consent,
            "consent_timestamp": consent_data.consent_timestamp,
            "consent_version": consent_data.consent_version,
            "consent_ip": request.client.host if request.client else "unknown",
            "consent_user_agent": request.headers.get("user-agent", "unknown"),
            "updated_at": datetime.utcnow(),
        }

        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": consent_record},
        )

        await db.b2c_insert_one(
            "consent_audit_log",
            {
                "user_id": ObjectId(user_id),
                "consent_type": "adult",
                "action": "consent_granted",
                "timestamp": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "consent_version": consent_data.consent_version,
                "consent_details": {
                    "gdpr_consent": consent_data.gdpr_consent,
                    "ai_personalization_consent": consent_data.ai_personalization_consent,
                    "marketing_consent": consent_data.marketing_consent,
                },
            },
        )

        logger.info(f"Adult GDPR consent recorded for user: {user_id}")

        return {
            "success": True,
            "message": "Consent recorded successfully",
            "data": {"consent_completed": True, "is_minor": False},
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adult consent error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record consent",
        )


@router.post("/consent/parental")
@limiter.limit("5/minute")
async def submit_parental_consent(
    request: Request,
    consent_data: ParentalConsentRequest,
    current_user: Dict[str, Any] = Depends(get_current_b2c_user),
    db: DatabaseManager = Depends(get_database),
):
    """
    Submit parental consent for minors (under 16) under GDPR Article 8.

    - Records parent/guardian information
    - Stores all consent checkboxes
    - Records digital signature and audit trail
    - Compliant with EU SCC requirements
    """
    try:
        user_id = current_user.get("user_id")

        pc = consent_data.parental_consent
        if not all(
            [
                pc.is_legal_guardian,
                pc.consent_data_processing,
                pc.consent_ai_analysis,
                pc.consent_international_transfers,
            ]
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All parental consent checkboxes must be checked",
            )

        if (
            consent_data.digital_signature.lower().strip()
            != consent_data.parent_info.full_name.lower().strip()
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Digital signature must match parent's full name",
            )

        parent_info = {
            "full_name": consent_data.parent_info.full_name,
            "email": consent_data.parent_info.email,
            "phone": consent_data.parent_info.phone,
            "country": consent_data.parent_info.country,
            "relationship": consent_data.parent_info.relationship,
        }

        consent_record = {
            "consent_completed": True,
            "is_minor": True,
            "has_parental_consent": True,
            "parent_info": parent_info,
            "parental_consent": {
                "is_legal_guardian": pc.is_legal_guardian,
                "consent_data_processing": pc.consent_data_processing,
                "consent_ai_analysis": pc.consent_ai_analysis,
                "consent_international_transfers": pc.consent_international_transfers,
            },
            "digital_signature": consent_data.digital_signature,
            "signature_date": consent_data.signature_date,
            "consent_timestamp": consent_data.consent_timestamp,
            "consent_version": consent_data.consent_version,
            "scc_version": consent_data.scc_version,
            "consent_ip": request.client.host if request.client else "unknown",
            "consent_user_agent": request.headers.get("user-agent", "unknown"),
            "updated_at": datetime.utcnow(),
        }

        await db.b2c_update_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"$set": consent_record},
        )

        await db.b2c_insert_one(
            "consent_audit_log",
            {
                "user_id": ObjectId(user_id),
                "consent_type": "parental",
                "action": "consent_granted",
                "timestamp": datetime.utcnow(),
                "ip_address": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "consent_version": consent_data.consent_version,
                "scc_version": consent_data.scc_version,
                "parent_info": parent_info,
                "consent_details": {
                    "is_legal_guardian": pc.is_legal_guardian,
                    "consent_data_processing": pc.consent_data_processing,
                    "consent_ai_analysis": pc.consent_ai_analysis,
                    "consent_international_transfers": pc.consent_international_transfers,
                },
                "digital_signature": consent_data.digital_signature,
                "signature_date": consent_data.signature_date,
            },
        )

        logger.info(f"Parental consent recorded for minor user: {user_id}")

        return {
            "success": True,
            "message": "Parental consent recorded successfully",
            "data": {
                "consent_completed": True,
                "is_minor": True,
                "has_parental_consent": True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parental consent error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record parental consent",
        )
