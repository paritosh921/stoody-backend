"""
Bypass authentication endpoint for testing when MongoDB is unavailable
WARNING: Remove this file in production!
"""
from fastapi import APIRouter
from pydantic import BaseModel, EmailStr
from core.auth import AuthManager
from datetime import datetime

router = APIRouter()

class BypassLoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/bypass/admin/login")
async def bypass_admin_login(login_data: BypassLoginRequest):
    """
    Bypass login endpoint - Returns token without MongoDB check
    WARNING: Only for testing! Remove in production!
    """
    if login_data.email != "admin@skillbot.app" or login_data.password != "admin123":
        return {"success": False, "message": "Invalid credentials"}

    # Create auth manager
    auth_manager = AuthManager()

    # Create admin data with real demo admin ID
    admin_data = {
        "id": "68e8d0d9a78ac3146233970f",  # Real demo admin ObjectId from database
        "email": "admin@skillbot.app",
        "full_name": "Demo Administrator",
        "user_type": "admin",
        "subdomain": "demo"
    }

    # Create session (this creates JWT token)
    session_data = await auth_manager.create_user_session(admin_data)

    return {
        "success": True,
        "data": {
            "access_token": session_data["access_token"],
            "user_type": "admin",
            "user": session_data["user"]
        }
    }
