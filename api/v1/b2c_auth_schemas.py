"""
Pydantic models for B2C authentication and profile endpoints.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, EmailStr


class GoogleLoginRequest(BaseModel):
    """Request model for Google OAuth login."""
    credential: str = Field(..., description="Google OAuth ID token")


class B2CUserResponse(BaseModel):
    """Response model for B2C user data."""
    user_id: str
    email: str
    full_name: str
    picture: Optional[str] = None
    user_type: str = "b2c_user"


class B2CTokenResponse(BaseModel):
    """Response model for B2C authentication."""
    success: bool = True
    data: Dict[str, Any]


class B2CAdminLoginRequest(BaseModel):
    """Request model for B2C Admin login."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class B2CAdminResponse(BaseModel):
    """Response model for B2C Admin data."""
    admin_id: str
    username: str
    email: Optional[str] = None
    full_name: str
    user_type: str = "b2c_admin"


class B2CAdminSetupRequest(BaseModel):
    """Request model for initial B2C Admin setup."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: Optional[EmailStr] = None
    full_name: str = Field(..., min_length=2, max_length=100)
    setup_key: str = Field(..., description="Secret key to authorize admin creation")


class B2COnboardingRequest(BaseModel):
    """Request model for B2C user onboarding."""
    exam_type: str = Field(..., description="JEE or NEET")
    class_level: str = Field(..., description="9, 10, 11, 12, or Dropper")
    full_name: str = Field(..., min_length=2, max_length=100)
    phone: str = Field(..., min_length=10, max_length=15)
    school_name: Optional[str] = None
    city: Optional[str] = None


class B2CProfileUpdateRequest(BaseModel):
    """Request model for B2C profile update."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = Field(None, min_length=10, max_length=15)
    school_name: Optional[str] = None
    city: Optional[str] = None
    exam_type: Optional[str] = None
    class_level: Optional[str] = None


class AdultConsentRequest(BaseModel):
    """Request model for adult GDPR consent."""
    is_minor: bool = False
    gdpr_consent: bool = Field(..., description="Privacy policy consent")
    ai_personalization_consent: bool = Field(..., description="AI processing consent")
    marketing_consent: bool = Field(False, description="Marketing communications consent")
    consent_timestamp: str = Field(..., description="ISO timestamp of consent")
    consent_version: str = Field("1.0", description="Version of consent document")


class ParentInfoModel(BaseModel):
    """Model for parent/guardian information."""
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., description="Parent's email address")
    phone: str = Field(..., min_length=10, max_length=20)
    country: str = Field(..., description="Country of residence")
    relationship: str = Field(..., description="Relationship to child")


class ParentalConsentModel(BaseModel):
    """Model for parental consent checkboxes."""
    is_legal_guardian: bool
    consent_data_processing: bool
    consent_ai_analysis: bool
    consent_international_transfers: bool


class ParentalConsentRequest(BaseModel):
    """Request model for parental GDPR consent."""
    is_minor: bool = True
    parent_info: ParentInfoModel
    parental_consent: ParentalConsentModel
    digital_signature: str = Field(..., description="Parent's full name as digital signature")
    signature_date: str = Field(..., description="ISO timestamp of signature")
    consent_timestamp: str = Field(..., description="ISO timestamp of consent")
    consent_version: str = Field("1.0", description="Version of consent document")
    scc_version: str = Field("2021/914", description="EU SCC version")
