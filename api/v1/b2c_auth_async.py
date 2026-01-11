"""
B2C authentication API router aggregator for Stoody.
"""

from fastapi import APIRouter

from api.v1.b2c_admin_async import router as admin_router
from api.v1.b2c_admin_leaderboard_async import router as leaderboard_router
from api.v1.b2c_auth_google_async import router as google_router
from api.v1.b2c_auth_user_async import router as user_router
from api.v1.b2c_consent_async import router as consent_router
from api.v1.b2c_parent_async import router as parent_router
from api.v1.b2c_profile_async import router as profile_router

router = APIRouter()

router.include_router(google_router)
router.include_router(user_router)
router.include_router(admin_router)
router.include_router(profile_router)
router.include_router(leaderboard_router)
router.include_router(consent_router)
router.include_router(parent_router)
