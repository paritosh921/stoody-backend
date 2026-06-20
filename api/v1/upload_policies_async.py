"""Public upload policy metadata for client-side UX hints."""

from fastapi import APIRouter

from core.upload_security.policies import all_public_upload_policies


router = APIRouter()


@router.get("/upload-policies/public")
async def get_public_upload_policies():
    return {
        "success": True,
        "policies": all_public_upload_policies(),
    }
