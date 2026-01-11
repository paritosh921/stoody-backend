from fastapi import APIRouter, Depends
from core.database import DatabaseManager
from api.v1.auth_async import get_database

from .structure import router as structure_router
from .content import router as content_router

learning_router = APIRouter()

@learning_router.get("/health", tags=["Learning"])
async def health_check(db: DatabaseManager = Depends(get_database)):
    """Health check endpoint"""
    try:
        docs = await db.mongo_find("documents", {"document_type": "Chapter Notes"}, limit=1)
        doc_count = len(docs) if docs else 0
        return {
            "success": True,
            "message": "Learning Mode API is operational",
            "has_documents": doc_count > 0
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Health check failed: {str(e)}"
        }

learning_router.include_router(structure_router)
learning_router.include_router(content_router)
