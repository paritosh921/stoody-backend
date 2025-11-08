"""
SkillBot Async Backend - FastAPI Application
Designed for high concurrency (1000+ concurrent users)
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import configuration
from config_async import (
    CORS_ORIGINS,
    API_V1_PREFIX,
    RATE_LIMIT_DEFAULT,
    REDIS_URL,
    DEBUG_MODE,
    MAX_WORKERS,
    WORKER_CONNECTIONS,
    OCR_CONCURRENCY_LIMIT
)

# Import async database clients
from core.database import DatabaseManager
from core.cache import CacheManager
from core.auth import AuthManager

# Import middleware
from middleware.subdomain import subdomain_middleware

# Import async route modules
from api.v1.chat_async import router as chat_router
from api.v1.auth_async import router as auth_router
from api.v1.auth_bypass import router as auth_bypass_router
from api.v1.admin_async import router as admin_router
from api.v1.student_async import router as student_router
from api.v1.questions_async import router as questions_router
from api.v1.images_async import router as images_router
from api.v1.practice_async import router as practice_router
from api.v1.mcq_async import router as mcq_router
from api.v1.tutot_async import router as tutor_router

from api.v1.learning_async import router as learning_router

# Optional debugger routes (require LangChain stack). Gate import to avoid hard dependency.
try:
    from api.v1.debugger_async import router as debugger_router  # type: ignore
    _debugger_available = True
except Exception:
    debugger_router = None  # type: ignore
    _debugger_available = False

from api.v1.pdf_async import router as pdf_router


# Configure logging
logging.basicConfig(
    level=logging.INFO if DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
db_manager = None
cache_manager = None
auth_manager = None
session_timeout_task = None

async def check_inactive_sessions():
    """Background task to mark students offline if inactive for >5 minutes"""
    while True:
        try:
            if db_manager:
                # Find students who are online but haven't had activity in 5 minutes
                timeout_threshold = datetime.utcnow() - timedelta(minutes=5)

                # Get all online students
                online_students = await db_manager.mongo_find(
                    "students",
                    {"is_online": True}
                )

                for student in online_students:
                    student_id = student["_id"]

                    # Check last activity (get most recent)
                    activities = await db_manager.mongo_find(
                        "student_activity_log",
                        {"student_id": student_id},
                        sort=[("timestamp", -1)],
                        limit=1
                    )
                    last_activity = activities[0] if activities else None

                    # If last activity was more than 5 minutes ago, mark offline
                    if last_activity:
                        last_time = last_activity.get("timestamp")
                        if last_time and last_time < timeout_threshold:
                            await db_manager.mongo_update_one(
                                "students",
                                {"_id": student_id},
                                {"$set": {"is_online": False}}
                            )

                            # Log auto-logout
                            await db_manager.mongo_insert_one("student_activity_log", {
                                "student_id": student_id,
                                "action": "auto_logout",
                                "timestamp": datetime.utcnow(),
                                "metadata": {
                                    "reason": "inactivity_timeout"
                                }
                            })
                            logger.info(f"Auto-logged out student {student_id} due to inactivity")

        except Exception as e:
            logger.error(f"Session timeout check error: {str(e)}")

        # Run every 2 minutes
        await asyncio.sleep(120)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app"""
    global db_manager, cache_manager, auth_manager, session_timeout_task

    logger.info("🚀 Starting SkillBot Async Backend...")

    try:
        # Initialize database connections
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Initialize cache (optional - continue without it if unavailable)
        cache_manager = CacheManager(REDIS_URL)
        try:
            cache_init_success = await cache_manager.initialize()
            if not cache_init_success and not DEBUG_MODE:
                logger.warning("⚠️ Cache initialization returned False in production mode")
        except Exception as cache_error:
            logger.error(f"⚠️ Cache initialization failed: {str(cache_error)}")
            if DEBUG_MODE:
                logger.info("🔧 Continuing without cache in development mode")
                cache_manager = None
            else:
                # In production, cache is required
                raise

        # Initialize auth manager
        auth_manager = AuthManager()
        await auth_manager.initialize()

        # Inject cache dependency so sessions and rate limits hit Redis
        if cache_manager:
            auth_manager.set_cache_manager(cache_manager)

        # Store in app state
        app.state.db = db_manager
        app.state.cache = cache_manager
        app.state.auth = auth_manager
        app.state.ocr_semaphore = asyncio.Semaphore(max(1, OCR_CONCURRENCY_LIMIT))
        app.state.ocr_tasks = {}

        # Start background task for session timeout
        session_timeout_task = asyncio.create_task(check_inactive_sessions())
        logger.info("✅ Session timeout monitor started")

        logger.info("✅ All services initialized successfully")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("🛑 Shutting down services...")

        # Cancel background task
        if session_timeout_task:
            session_timeout_task.cancel()
            try:
                await session_timeout_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ Session timeout monitor stopped")

        if cache_manager:
            await cache_manager.close()
        if app.state and getattr(app.state, "ocr_tasks", None):
            for task_id, task in list(app.state.ocr_tasks.items()):
                if not task.done():
                    task.cancel()
            app.state.ocr_tasks.clear()
        if db_manager:
            await db_manager.close()
        logger.info("✅ Cleanup completed")

# Rate limiting setup (in-memory for development if Redis unavailable)
try:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[RATE_LIMIT_DEFAULT],
        storage_uri=REDIS_URL if not DEBUG_MODE else "memory://",
        strategy="fixed-window"
    )
except Exception as e:
    logger.warning(f"⚠️ Rate limiter using in-memory storage: {str(e)}")
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[RATE_LIMIT_DEFAULT],
        storage_uri="memory://",
        strategy="fixed-window"
    )

# Create FastAPI app with lifespan
app = FastAPI(
    title="SkillBot Async API",
    description="High-performance async API for SkillBot educational platform",
    version="2.0.0",
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None,
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Subdomain extraction middleware (MUST be before other middlewares)
app.middleware("http")(subdomain_middleware)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if DEBUG_MODE:
        # Don't log request body for OCR endpoints to avoid base64 spam
        if "/process-ocr" not in request.url.path:
            logger.info(f"📨 {request.method} {request.url.path}")
        else:
            logger.info(f"📨 {request.method} {request.url.path} (OCR processing - body not logged)")
    response = await call_next(request)
    if DEBUG_MODE:
        logger.info(f"📤 {response.status_code}")
    return response

# Include routers
app.include_router(
    chat_router,
    prefix=f"{API_V1_PREFIX}/chat",
    tags=["Chat"]
)

app.include_router(
    auth_router,
    prefix=f"{API_V1_PREFIX}/auth",
    tags=["Authentication"]
)

# Also include auth routes at /auth for frontend compatibility
app.include_router(
    auth_router,
    prefix="/auth",
    tags=["Authentication (Legacy)"]
)

# Bypass auth for testing when MongoDB is unavailable
if DEBUG_MODE:
    app.include_router(
        auth_bypass_router,
        prefix=f"{API_V1_PREFIX}/auth",
        tags=["Authentication Bypass (DEV ONLY)"]
    )

app.include_router(
    admin_router,
    prefix=f"{API_V1_PREFIX}/admin",
    tags=["Admin"]
)

app.include_router(
    student_router,
    prefix=f"{API_V1_PREFIX}/student",
    tags=["Student"]
)

app.include_router(
    questions_router,
    prefix=f"{API_V1_PREFIX}/questions",
    tags=["Questions"]
)

app.include_router(
    images_router,
    prefix=f"{API_V1_PREFIX}/images",
    tags=["Images"]
)

app.include_router(
    practice_router,
    prefix=f"{API_V1_PREFIX}/practice",
    tags=["Practice"]
)

app.include_router(
    mcq_router,
    prefix=f"{API_V1_PREFIX}/mcq",
    tags=["MCQ"]
)

# Tutor routes (admin can manage tutors; tutors can view their own students)
app.include_router(
    tutor_router,
    prefix=f"{API_V1_PREFIX}",
    tags=["Tutor"]
)

# Also mount tutor routes under /api/v1/tutor for frontend compatibility
app.include_router(
    tutor_router,
    prefix=f"{API_V1_PREFIX}/tutor",
    tags=["Tutor (Prefixed)"]
)

app.include_router(
    pdf_router,
    prefix=f"{API_V1_PREFIX}/pdf",
    tags=["PDF Processing"]
)

if _debugger_available and debugger_router:
    app.include_router(
        debugger_router,
        prefix=f"{API_V1_PREFIX}/debugger",
        tags=["Debugger"]
    )

    # Also include debugger routes at /api/debugger for frontend compatibility
    app.include_router(
        debugger_router,
        prefix="/api/debugger",
        tags=["Debugger (Legacy)"]
    )
else:
    logger.warning("Debugger routes disabled (optional dependencies missing)")

app.include_router(
    learning_router,
    prefix=f"{API_V1_PREFIX}/learning",
    tags=["Learning"]
)

# Also include learning routes at /api/learning for frontend compatibility
app.include_router(
    learning_router,
    prefix="/api/learning",
    tags=["Learning (Legacy)"]
)

# Also include MCQ routes at /api/mcq for frontend compatibility
app.include_router(
    mcq_router,
    prefix="/api/mcq",
    tags=["MCQ (Legacy)"]
)

# Static file serving
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Health check endpoint
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Comprehensive health check"""
    try:
        # Check database connection
        db_healthy = await app.state.db.health_check() if app.state.db else False

        # Check cache connection
        cache_healthy = await app.state.cache.health_check() if app.state.cache else False

        # Cache is optional in development mode
        cache_required = not DEBUG_MODE
        cache_status = "healthy" if cache_healthy else ("optional" if DEBUG_MODE else "unhealthy")

        # Check ChromaDB connection and count (non-fatal)
        chroma_healthy = False
        chroma_count = 0
        try:
            if app.state.db:
                chroma_count = await app.state.db.chroma_count()
                chroma_healthy = chroma_count is not None
        except Exception as _chroma_err:
            logger.warning(f"ChromaDB health probe failed: {str(_chroma_err)}")

        # Treat cache as optional in all environments; report overall service as running
        # even if one or more dependencies are degraded. Frontend uses this flag to
        # detect whether the backend is reachable.
        overall_healthy = True

        status_str = "healthy" if db_healthy else "degraded"
        # Include success/healthy booleans and common legacy keys for frontend compatibility
        payload = {
            "success": overall_healthy,
            "healthy": overall_healthy,
            "ok": overall_healthy,
            "status": status_str,
            "message": "Backend server is running" if overall_healthy else "Backend is degraded",
            "timestamp": time.time(),
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": cache_status,
                "chromadb": {
                    "connected": chroma_healthy,
                    "status": "online" if chroma_healthy else "offline",
                    "questions_count": chroma_count
                }
            },
            "chromaConnected": chroma_healthy,
            "chromadb": {
                "connected": chroma_healthy,
                "status": "online" if chroma_healthy else "offline",
                "questions_count": chroma_count
            },
            "version": "2.0.0",
            "mode": "development" if DEBUG_MODE else "production"
        }

        # Explicitly disable caching at any proxies/CDNs/browsers
        return JSONResponse(
            status_code=200,
            content=payload,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Backend-Server": "fastapi-async"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

# Compatibility health endpoints for frontend/CDN path expectations
@app.get("/api/health")
@limiter.limit("60/minute")
async def health_check_api_alias(request: Request):
    return await health_check(request)

@app.get("/api/v1/health")
@limiter.limit("60/minute")
async def health_check_v1_alias(request: Request):
    return await health_check(request)

# Healthz alias used by some load balancers/CDNs
@app.get("/healthz")
@limiter.limit("60/minute")
async def healthz_alias(request: Request):
    return await health_check(request)

# Legacy compatibility endpoint for token verification
@app.get("/verify")
async def verify_token_legacy(request: Request):
    """Legacy /verify endpoint for frontend compatibility"""
    try:
        # Import the auth manager
        from api.v1.auth_async import get_current_user, get_auth_manager
        from fastapi import Depends

        auth_manager = request.app.state.auth
        # This will raise an exception if not authenticated
        current_user = await get_current_user(request, auth_manager)

        return {
            "success": True,
            "data": {
                "user_id": current_user.get("user_id"),
                "user_type": current_user.get("user_type"),
                "email": current_user.get("email"),
                "username": current_user.get("username"),
                "full_name": current_user.get("full_name")
            }
        }
    except Exception as e:
        logger.error(f"Legacy verify endpoint error: {str(e)}")
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "Not authenticated",
                "message": "Invalid or missing authentication token"
            }
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SkillBot Async Backend API",
        "version": "2.0.0",
        "docs": "/docs" if DEBUG_MODE else "disabled",
        "health": "/health",
        "features": [
            "High concurrency support (1000+ users)",
            "Async database operations",
            "Redis caching",
            "Rate limiting",
            "Background task processing",
            "Comprehensive monitoring"
        ],
        "endpoints": {
            "chat": f"{API_V1_PREFIX}/chat",
            "auth": f"{API_V1_PREFIX}/auth",
            "admin": f"{API_V1_PREFIX}/admin",
            "student": f"{API_V1_PREFIX}/student",
            "tutor": f"{API_V1_PREFIX}/tutors",
            "questions": f"{API_V1_PREFIX}/questions",
            "images": f"{API_V1_PREFIX}/images",
            "practice": f"{API_V1_PREFIX}/practice",
            "mcq": f"{API_V1_PREFIX}/mcq",
            "debugger": f"{API_V1_PREFIX}/debugger"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

# 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not found",
            "message": "The requested endpoint does not exist"
        }
    )

if __name__ == "__main__":
    import uvicorn

    # Run with Uvicorn
    uvicorn.run(
        "main_async:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5001)),
        workers=1 if DEBUG_MODE else MAX_WORKERS,
        limit_concurrency=WORKER_CONNECTIONS,
        reload=DEBUG_MODE,
        access_log=DEBUG_MODE,
        log_level="info" if DEBUG_MODE else "warning"
    )
