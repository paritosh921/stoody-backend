"""
Async PDF Processing API for SkillBot.
"""

import logging

from fastapi import APIRouter

from api.v1.pdf_documents_async import router as documents_router
from api.v1.pdf_images_async import router as images_router
from api.v1.pdf_ocr_async import router as ocr_router
from api.v1.pdf_questions_async import router as questions_router
from api.v1.pdf_student_async import router as student_router
from api.v1.pdf_upload_async import router as upload_router

# Suppress verbose aiohttp logging
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)
logging.getLogger("aiohttp.server").setLevel(logging.WARNING)

router = APIRouter()

router.include_router(upload_router)
router.include_router(ocr_router)
router.include_router(documents_router)
router.include_router(student_router)
router.include_router(questions_router)
router.include_router(images_router)
