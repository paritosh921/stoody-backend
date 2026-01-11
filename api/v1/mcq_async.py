"""
Async MCQ API aggregator for SkillBot.
"""

import logging

from fastapi import APIRouter

from api.v1.mcq_attempts_async import router as attempts_router
from api.v1.mcq_questions_async import router as questions_router
from api.v1.mcq_test_series_async import router as test_series_router

logger = logging.getLogger(__name__)

router = APIRouter()

router.include_router(questions_router)
router.include_router(test_series_router)
router.include_router(attempts_router)
