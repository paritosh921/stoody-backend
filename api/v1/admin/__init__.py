from fastapi import APIRouter
from .students import router as students_router
from .dashboard import router as dashboard_router
from .monitoring import router as monitoring_router
from .promotion import router as promotion_router

admin_router = APIRouter()

admin_router.include_router(students_router, tags=["Admin - Students"])
admin_router.include_router(dashboard_router, tags=["Admin - Dashboard"])
admin_router.include_router(monitoring_router, tags=["Admin - Monitoring"])
admin_router.include_router(promotion_router, tags=["Admin - Session"])
