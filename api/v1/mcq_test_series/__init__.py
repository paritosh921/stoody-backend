from fastapi import APIRouter
from .list import router as list_router
from .questions import router as questions_router
from .attempts import router as attempts_router

mcq_test_series_router = APIRouter()

mcq_test_series_router.include_router(list_router)
mcq_test_series_router.include_router(questions_router)
mcq_test_series_router.include_router(attempts_router)
