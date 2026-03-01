"""
SkillBot Async Backend - FastAPI Application
Designed for high concurrency (1000+ concurrent users)
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pyinstrument import Profiler
try:
    from prometheus_fastapi_instrumentator import Instrumentator
except Exception:
    Instrumentator = None
try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
except Exception:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None

# Import configuration
from config_async import (
    CORS_ORIGINS,
    API_V1_PREFIX,
    RATE_LIMIT_DEFAULT,
    REDIS_URL,
    REDIS_REQUIRED,
    DEBUG_MODE,
    MAX_WORKERS,
    WORKER_CONNECTIONS,
    OCR_CONCURRENCY_LIMIT,
    ENABLE_METRICS,
)

# Import async database clients
from core.database import DatabaseManager
from core.cache import CacheManager
from core.auth import AuthManager
from core.observability import set_dependency_health

# Import middleware
from middleware.subdomain import subdomain_middleware
from middleware.tenant_middleware import TenantMiddleware
from middleware.security_headers import SecurityHeadersMiddleware

# Import async route modules
from api.v1.chat_async import router as chat_router
from api.v1.auth_async import router as auth_router
from api.v1.auth_cookie import router as auth_cookie_router
from api.v1.admin_async import router as admin_router
from api.v1.student_bulk_upload import router as student_bulk_upload_router
from api.v1.tutor_bulk_upload import router as tutor_bulk_upload_router
from api.v1.student_async import router as student_router
from api.v1.questions_async import router as questions_router
from api.v1.images_async import router as images_router
from api.v1.practice_async import router as practice_router
from api.v1.mcq_async import router as mcq_router
from api.v1.tutor_async import router as tutor_router

# Diagram generation pipeline routes (Kimi 2.5 + Gemini Nano Banana)
try:
    from api.v1.diagram_async import router as diagram_router
    _diagram_available = True
except Exception as e:
    diagram_router = None
    _diagram_available = False
    logging.warning(f"Diagram routes disabled: {str(e)}")

# Paper Builder routes (Question Paper Generation)
try:
    from api.v1.paper_builder_async import router as paper_builder_router
    _paper_builder_available = True
except Exception as e:
    paper_builder_router = None
    _paper_builder_available = False
    logging.warning(f"Paper Builder routes disabled: {str(e)}")

from api.v1.learning_async import router as learning_router
from api.v1.strokes_async import router as strokes_router

# Student Copies routes (pen stroke pages and pinned PDFs)
try:
    from api.v1.copies_async import router as copies_router
    _copies_available = True
except Exception as e:
    copies_router = None
    _copies_available = False
    logging.warning(f"Copies routes disabled: {str(e)}")

# Optional debugger routes (require LangChain stack). Gate import to avoid hard dependency.
try:
    from api.v1.debugger_async import router as debugger_router  # type: ignore
    _debugger_available = True
except Exception:
    debugger_router = None  # type: ignore
    _debugger_available = False

from api.v1.pdf_async import router as pdf_router
from api.v1.language_async import router as language_router
from api.v1.settings_async import router as settings_router
from api.v1.totp_2fa import router as totp_2fa_router

# B2C Authentication routes (Google OAuth for B2C users using stoody-b2c database)
try:
    from api.v1.b2c_auth_async import router as b2c_auth_router
    _b2c_auth_available = True
except Exception as e:
    b2c_auth_router = None
    _b2c_auth_available = False
    logging.warning(f"B2C auth routes disabled: {str(e)}")

# Online Class routes (WebSocket for remote teaching)
try:
    from api.v1.online_class_async import router as online_class_router
    _online_class_available = True
except Exception as e:
    online_class_router = None
    _online_class_available = False
    logging.warning(f"Online class routes disabled: {str(e)}")

# Classroom Management routes (Online Class with Meet links)
try:
    from api.v1.classroom_async import router as classroom_router
    _classroom_available = True
except Exception as e:
    classroom_router = None
    _classroom_available = False
    logging.warning(f"Classroom routes disabled: {str(e)}")

# Meeting Management routes (Google Meet integration)
try:
    from api.v1.meeting_async import router as meeting_router
    _meeting_available = True
except Exception as e:
    meeting_router = None
    _meeting_available = False
    logging.warning(f"Meeting routes disabled: {str(e)}")

# SmartBoard routes (real-time pen monitoring for teaching)
try:
    from api.v1.smartboard_async import router as smartboard_router
    _smartboard_available = True
except Exception as e:
    smartboard_router = None
    _smartboard_available = False
    logging.warning(f"SmartBoard routes disabled: {str(e)}")

# SmartBoard Token routes (secure JWT token issuance for SmartBoard access)
try:
    from api.v1.smartboard_token import router as smartboard_token_router
    _smartboard_token_available = True
except Exception as e:
    smartboard_token_router = None
    _smartboard_token_available = False
    logging.warning(f"SmartBoard Token routes disabled: {str(e)}")

# Dashboard routes (pen monitoring from stoody-pen-multi)
try:
    from api.v1.dashboard import router as dashboard_router
    _dashboard_available = True
except Exception as e:
    dashboard_router = None
    _dashboard_available = False
    logging.warning(f"Dashboard routes disabled: {str(e)}")

# Hub routes (BLE hub registration)
try:
    from api.v1.hub import router as hub_router
    _hub_available = True
except Exception as e:
    hub_router = None
    _hub_available = False
    logging.warning(f"Hub routes disabled: {str(e)}")

# Notes routes (teacher canvas notes)
try:
    from api.v1.notes import router as notes_router
    _notes_available = True
except Exception as e:
    notes_router = None
    _notes_available = False
    logging.warning(f"Notes routes disabled: {str(e)}")

# Pen Notes routes (student handwritten pen notes — read-only, data written by pen server)
try:
    from api.v1.pen_notes_async import router as pen_notes_router
    _pen_notes_available = True
except Exception as e:
    pen_notes_router = None
    _pen_notes_available = False
    logging.warning(f"Pen Notes routes disabled: {str(e)}")

# OCR routes (handwriting recognition)
try:
    from api.v1.ocr import router as ocr_router
    _ocr_available = True
except Exception as e:
    ocr_router = None
    _ocr_available = False
    logging.warning(f"OCR routes disabled: {str(e)}")

# Question Attempts routes (question locking and evaluation)
try:
    from api.v1.question_attempts import router as question_attempts_router
    _question_attempts_available = True
except Exception as e:
    question_attempts_router = None
    _question_attempts_available = False
    logging.warning(f"Question Attempts routes disabled: {str(e)}")

# Super Admin routes (tenant management for super admins)
try:
    from api.v1.superadmin_async import router as superadmin_router
    _superadmin_available = True
except Exception as e:
    superadmin_router = None
    _superadmin_available = False
    logging.warning(f"Super Admin routes disabled: {str(e)}")

# Smartboard Sessions routes (tutor session management)
try:
    from api.v1.smartboard_sessions_async import router as smartboard_sessions_router
    _smartboard_sessions_available = True
except Exception as e:
    smartboard_sessions_router = None
    _smartboard_sessions_available = False
    logging.warning(f"Smartboard Sessions routes disabled: {str(e)}")

# Timetable routes (class timetable, teacher calendar, pre-reads)
try:
    from api.v1.timetable_async import router as timetable_router
    from api.v1.timetable_bulk_upload import router as timetable_bulk_upload_router
    _timetable_available = True
except Exception as e:
    timetable_router = None
    timetable_bulk_upload_router = None
    _timetable_available = False
    logging.warning(f"Timetable routes disabled: {str(e)}")


# Configure logging
logging.basicConfig(
    level=logging.INFO if DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add rolling file log so Promtail can scrape backend runtime logs.
try:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    app_log_file = logs_dir / "app.log"
    file_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger = logging.getLogger()
    has_app_log_handler = any(
        isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("app.log")
        for h in root_logger.handlers
    )
    if not has_app_log_handler:
        root_logger.addHandler(file_handler)
except Exception as exc:
    logger.warning(f"Failed to configure app.log file handler: {exc}")

# Suppress httpx/httpcore logging to prevent token leakage in URLs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
                master_db = await db_manager.get_master_db()
                if master_db is None:
                    await asyncio.sleep(120)
                    continue

                tenants = await master_db["tenants"].find(
                    {"status": "active"},
                    {"db_name": 1},
                ).to_list(length=None)

                for tenant in tenants:
                    db_name = tenant.get("db_name")
                    if not db_name:
                        continue
                    tenant_db = await db_manager.get_tenant_db(db_name)
                    if tenant_db is None:
                        continue

                    online_students = await tenant_db["students"].find(
                        {"is_online": True}
                    ).to_list(length=10000)

                    for student in online_students:
                        student_id = student["_id"]

                        # Check last activity (get most recent)
                        activities = await tenant_db["student_activity_log"].find(
                            {"student_id": student_id}
                        ).sort("timestamp", -1).limit(1).to_list(length=1)
                        last_activity = activities[0] if activities else None

                        # If last activity was more than 5 minutes ago, mark offline
                        if last_activity:
                            last_time = last_activity.get("timestamp")
                            if last_time and last_time < timeout_threshold:
                                await tenant_db["students"].update_one(
                                    {"_id": student_id},
                                    {"$set": {"is_online": False}},
                                )

                                # Log auto-logout
                                await tenant_db["student_activity_log"].insert_one({
                                    "student_id": student_id,
                                    "action": "auto_logout",
                                    "timestamp": datetime.utcnow(),
                                    "metadata": {
                                        "reason": "inactivity_timeout"
                                    }
                                })
                                logger.info(
                                    "Auto-logged out student %s due to inactivity (tenant=%s)",
                                    student_id,
                                    db_name,
                                )

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

# =============================================================================
# RATE LIMITING SETUP (SECURITY-CRITICAL)
# =============================================================================
#
# Rate limiting prevents brute-force attacks. In production, Redis is REQUIRED
# for distributed rate limiting across multiple workers/instances.
#
# In-memory rate limiting is only acceptable in development because:
# - It doesn't persist across restarts
# - It doesn't work across multiple workers/processes
# - It can be easily bypassed during Redis outages
#

def _test_redis_connection(redis_url: str) -> bool:
    """Test if Redis is reachable"""
    try:
        import redis
        r = redis.from_url(redis_url, socket_connect_timeout=5)
        r.ping()
        return True
    except Exception as e:
        logger.error(f"Redis connection test failed: {str(e)}")
        return False


def _setup_rate_limiter() -> Limiter:
    """
    Set up rate limiter with proper security checks.

    Production: Requires Redis - fails startup if unavailable
    Development: Falls back to in-memory (with warning)
    """
    if DEBUG_MODE:
        # Development mode: try Redis, fall back to memory
        try:
            if _test_redis_connection(REDIS_URL):
                logger.info("✅ Rate limiter using Redis (development)")
                return Limiter(
                    key_func=get_remote_address,
                    default_limits=[RATE_LIMIT_DEFAULT],
                    storage_uri=REDIS_URL,
                    strategy="fixed-window"
                )
        except Exception as e:
            logger.warning(f"Redis unavailable in dev mode: {str(e)}")

        logger.warning(
            "⚠️ Rate limiter using IN-MEMORY storage (development only).\n"
            "   This does NOT work across multiple workers and resets on restart!"
        )
        return Limiter(
            key_func=get_remote_address,
            default_limits=[RATE_LIMIT_DEFAULT],
            storage_uri="memory://",
            strategy="fixed-window"
        )

    # Production mode: Redis is required for proper rate limiting
    if not REDIS_URL:
        if REDIS_REQUIRED:
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                "🚨 CRITICAL: Redis URL is not configured!\n"
                "=" * 70 + "\n"
                "Rate limiting without Redis is a security vulnerability.\n\n"
                "Please configure REDIS_URL in your environment:\n"
                "  REDIS_URL=redis://your-redis-host:6379/0\n\n"
                "Or set REDIS_REQUIRED=false to disable this check (NOT recommended).\n"
                "=" * 70
            )
        else:
            logger.warning(
                "⚠️ SECURITY WARNING: Using in-memory rate limiting in production.\n"
                "   This is NOT recommended and should only be temporary!"
            )
            return Limiter(
                key_func=get_remote_address,
                default_limits=[RATE_LIMIT_DEFAULT],
                storage_uri="memory://",
                strategy="fixed-window"
            )

    # Test Redis connection (works for localhost or remote)
    if not _test_redis_connection(REDIS_URL):
        if REDIS_REQUIRED:
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                "🚨 CRITICAL: Cannot connect to Redis!\n"
                "=" * 70 + "\n"
                f"Redis URL: {REDIS_URL}\n\n"
                "Rate limiting requires a working Redis connection in production.\n"
                "Please check your Redis configuration and ensure Redis is running.\n"
                "=" * 70
            )
        else:
            logger.error(
                "⚠️ SECURITY WARNING: Redis connection failed in production!\n"
                "   Falling back to in-memory rate limiting (NOT SECURE)."
            )
            return Limiter(
                key_func=get_remote_address,
                default_limits=[RATE_LIMIT_DEFAULT],
                storage_uri="memory://",
                strategy="fixed-window"
            )

    logger.info("✅ Rate limiter using Redis (production)")
    return Limiter(
        key_func=get_remote_address,
        default_limits=[RATE_LIMIT_DEFAULT],
        storage_uri=REDIS_URL,
        strategy="fixed-window"
    )


# Initialize rate limiter
limiter = _setup_rate_limiter()

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

# Prometheus metrics endpoint
if ENABLE_METRICS and Instrumentator is not None:
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/api/metrics", "/api/v1/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=DEBUG_MODE)
    logger.info("✅ Prometheus metrics enabled at /metrics")
elif ENABLE_METRICS:
    logger.warning("⚠️ Metrics enabled but prometheus-fastapi-instrumentator is unavailable")

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

# Tenant context middleware - sets admin_id for data isolation
app.add_middleware(TenantMiddleware)

# Security headers middleware - adds CSP, HSTS, X-Frame-Options, etc.
app.add_middleware(SecurityHeadersMiddleware)

@app.middleware("http")
async def profile_request(request: Request, call_next):
    """Profiler middleware"""
    if request.query_params.get("profile", "false").lower() == "true":
        profiler = Profiler(interval=0.001, async_mode="enabled")
        profiler.start()
        try:
            response = await call_next(request)
            return response
        finally:
            profiler.stop()
            output_dir = "logs/profiles"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            html_path = f"{output_dir}/profile_{timestamp}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(profiler.output_html())
            logger.info(f"Profiler saved to {html_path}")
            logger.info("\n" + profiler.output_text(unicode=True, color=True, show_all=True))
    else:
        return await call_next(request)

# Request timing middleware
def _is_client_disconnect_runtime_error(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and str(exc) == "No response returned."


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except RuntimeError as exc:
        if _is_client_disconnect_runtime_error(exc):
            if DEBUG_MODE:
                logger.debug(
                    "Client disconnected before response was sent: %s %s",
                    request.method,
                    request.url.path,
                )
            # Non-standard but widely used for client-aborted requests.
            return Response(status_code=499)
        raise
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
    try:
        response = await call_next(request)
    except RuntimeError as exc:
        if _is_client_disconnect_runtime_error(exc):
            if DEBUG_MODE:
                logger.debug(
                    "Request aborted by client before response: %s %s",
                    request.method,
                    request.url.path,
                )
            return Response(status_code=499)
        raise
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

# Cookie-based authentication routes (httpOnly cookies for security)
app.include_router(
    auth_cookie_router,
    prefix=f"{API_V1_PREFIX}/auth",
    tags=["Cookie Authentication"]
)

# Also include cookie auth routes at /auth for frontend compatibility
app.include_router(
    auth_cookie_router,
    prefix="/auth",
    tags=["Cookie Authentication (Legacy)"]
)
logger.info("✅ Cookie-based authentication routes enabled")

app.include_router(
    admin_router,
    prefix=f"{API_V1_PREFIX}/admin",
    tags=["Admin"]
)

# Student Bulk Upload routes (under admin)
app.include_router(
    student_bulk_upload_router,
    prefix=f"{API_V1_PREFIX}/admin",
    tags=["Student Bulk Upload"]
)
logger.info("✅ Student Bulk Upload routes enabled")

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

# Diagram generation pipeline routes
if _diagram_available:
    app.include_router(
        diagram_router,
        prefix=API_V1_PREFIX,
        tags=["Diagrams"]
    )
    logger.info("✅ Diagram generation pipeline routes enabled")

# Paper Builder routes
if _paper_builder_available:
    app.include_router(
        paper_builder_router,
        prefix=API_V1_PREFIX,
        tags=["Paper Builder"]
    )
    logger.info("✅ Paper Builder routes enabled")

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

# Teacher Bulk Upload routes (under /api/v1/tutor)
app.include_router(
    tutor_bulk_upload_router,
    prefix=f"{API_V1_PREFIX}/tutor",
    tags=["Teacher Bulk Upload"]
)
logger.info("✅ Teacher Bulk Upload routes enabled")

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

# Strokes from Stoody BLE agent (shared Mongo)
app.include_router(
    strokes_router,
    prefix=f"{API_V1_PREFIX}",
    tags=["Strokes"]
)

# Student Copies routes (pen stroke pages and pinned PDFs)
if _copies_available and copies_router:
    app.include_router(
        copies_router,
        prefix=f"{API_V1_PREFIX}/learning",
        tags=["Student Copies"]
    )
    logger.info("✅ Student Copies routes enabled")
else:
    logger.warning("⚠️ Student Copies routes disabled")

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

# Language Lab SSO routes
app.include_router(
    language_router,
    prefix=f"{API_V1_PREFIX}/language",
    tags=["Language Lab"]
)

# Video Content routes
try:
    from api.v1.video_async import router as video_router
    app.include_router(
        video_router,
        prefix=f"{API_V1_PREFIX}/video",
        tags=["Video"]
    )
    logger.info("✅ Video routes enabled")
except Exception as e:
    logger.warning(f"Video routes disabled: {str(e)}")


# School Settings routes
app.include_router(
    settings_router,
    prefix=f"{API_V1_PREFIX}/admin",
    tags=["School Settings"]
)

# B2C Authentication routes (Google OAuth for B2C users)
if _b2c_auth_available and b2c_auth_router:
    app.include_router(
        b2c_auth_router,
        prefix=f"{API_V1_PREFIX}/b2c",
        tags=["B2C Authentication"]
    )
    # Also mount at /auth/b2c for frontend convenience
    app.include_router(
        b2c_auth_router,
        prefix="/auth/b2c",
        tags=["B2C Authentication (Legacy)"]
    )
    logger.info("✅ B2C authentication routes enabled")
else:
    logger.warning("⚠️ B2C authentication routes disabled (missing dependencies)")

# TOTP 2FA routes (Google Authenticator for admins and tutors)
app.include_router(
    totp_2fa_router,
    prefix=f"{API_V1_PREFIX}/auth/2fa",
    tags=["Two-Factor Authentication"]
)
# Also mount at /auth/2fa for frontend convenience
app.include_router(
    totp_2fa_router,
    prefix="/auth/2fa",
    tags=["Two-Factor Authentication (Legacy)"]
)
logger.info("✅ TOTP 2FA authentication routes enabled")

# Classroom Management routes (Online Class with Meet links)
if _classroom_available and classroom_router:
    app.include_router(
        classroom_router,
        prefix=f"{API_V1_PREFIX}/classroom",
        tags=["Classroom"]
    )
    logger.info("✅ Classroom routes enabled")
else:
    logger.warning("⚠️ Classroom routes disabled")

# Meeting Management routes (Google Meet integration for online classes)
if _meeting_available and meeting_router:
    app.include_router(
        meeting_router,
        prefix=f"{API_V1_PREFIX}/meeting",
        tags=["Meeting"]
    )
    logger.info("✅ Meeting routes enabled")
else:
    logger.warning("⚠️ Meeting routes disabled")

# SmartBoard routes (real-time pen monitoring, question attempts)
if _smartboard_available and smartboard_router:
    app.include_router(
        smartboard_router,
        prefix=f"{API_V1_PREFIX}/smartboard",
        tags=["SmartBoard"]
    )
    logger.info("✅ SmartBoard routes enabled")
else:
    logger.warning("⚠️ SmartBoard routes disabled")

# SmartBoard Token routes (secure JWT token issuance)
if _smartboard_token_available and smartboard_token_router:
    app.include_router(
        smartboard_token_router,
        prefix=f"{API_V1_PREFIX}",
        tags=["SmartBoard Auth"]
    )
    logger.info("✅ SmartBoard Token routes enabled")
else:
    logger.warning("⚠️ SmartBoard Token routes disabled")

# Dashboard routes (pen monitoring from stoody-pen-multi)
if _dashboard_available and dashboard_router:
    app.include_router(
        dashboard_router,
        tags=["Dashboard"]
    )
    logger.info("✅ Dashboard routes enabled")
else:
    logger.warning("⚠️ Dashboard routes disabled")

# Hub routes (BLE hub registration)
if _hub_available and hub_router:
    app.include_router(
        hub_router,
        tags=["Hub"]
    )
    logger.info("✅ Hub routes enabled")
else:
    logger.warning("⚠️ Hub routes disabled")

# Notes routes (teacher canvas notes)
if _notes_available and notes_router:
    app.include_router(
        notes_router,
        tags=["Notes"]
    )
    logger.info("✅ Notes routes enabled")
else:
    logger.warning("⚠️ Notes routes disabled")

# Pen Notes routes (student handwritten pen notes)
if _pen_notes_available and pen_notes_router:
    app.include_router(
        pen_notes_router,
        prefix=f"{API_V1_PREFIX}",
        tags=["Pen Notes"]
    )
    logger.info("✅ Pen Notes routes enabled")
else:
    logger.warning("⚠️ Pen Notes routes disabled")

# OCR routes (handwriting recognition)
if _ocr_available and ocr_router:
    app.include_router(
        ocr_router,
        tags=["OCR"]
    )
    logger.info("✅ OCR routes enabled")
else:
    logger.warning("⚠️ OCR routes disabled")

# Question Attempts routes (question locking and evaluation)
if _question_attempts_available and question_attempts_router:
    app.include_router(
        question_attempts_router,
        tags=["Question Attempts"]
    )
    logger.info("✅ Question Attempts routes enabled")
else:
    logger.warning("⚠️ Question Attempts routes disabled")

# Super Admin routes (tenant management)
if _superadmin_available and superadmin_router:
    app.include_router(
        superadmin_router,
        prefix=f"{API_V1_PREFIX}/superadmin",
        tags=["Super Admin"]
    )
    logger.info("✅ Super Admin routes enabled")
else:
    logger.warning("⚠️ Super Admin routes disabled")

# Smartboard Sessions routes (tutor session management)
if _smartboard_sessions_available and smartboard_sessions_router:
    app.include_router(
        smartboard_sessions_router,
        tags=["Smartboard Sessions"]
    )
    logger.info("✅ Smartboard Sessions routes enabled")
else:
    logger.warning("⚠️ Smartboard Sessions routes disabled")

# Timetable routes (class timetable, teacher calendar, academic calendar, pre-reads)
if _timetable_available and timetable_router:
    app.include_router(
        timetable_router,
        prefix=f"{API_V1_PREFIX}/timetable",
        tags=["Timetable"]
    )
    app.include_router(
        timetable_bulk_upload_router,
        prefix=f"{API_V1_PREFIX}/admin",
        tags=["Timetable Bulk Upload"]
    )
    logger.info("✅ Timetable routes enabled")
else:
    logger.warning("⚠️ Timetable routes disabled")


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
        set_dependency_health("database", db_healthy)
        set_dependency_health("cache", cache_healthy)

        # Cache is optional in development mode
        cache_required = not DEBUG_MODE
        cache_status = "healthy" if cache_healthy else ("optional" if DEBUG_MODE else "unhealthy")

        # Get active tenant count from master DB (non-fatal, no tenant context needed)
        active_tenants_count = 0
        try:
            master_db = await app.state.db.get_master_db() if app.state.db else None
            if master_db is not None:
                active_tenants_count = await master_db["tenants"].count_documents({"status": "active"})
        except Exception as _master_err:
            logger.warning(f"Master DB tenant count failed: {str(_master_err)}")

        # Check S3 storage status
        try:
            from utils.s3_storage import is_s3_enabled, S3_BUCKET_NAME, USE_S3_STORAGE
            s3_enabled = is_s3_enabled()
            s3_status = {
                "enabled": s3_enabled,
                "use_s3_storage_env": USE_S3_STORAGE,
                "bucket": S3_BUCKET_NAME if s3_enabled else None
            }
            set_dependency_health("s3_storage", s3_enabled)
        except Exception as _s3_err:
            s3_status = {"enabled": False, "error": str(_s3_err)}
            set_dependency_health("s3_storage", False)

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
                "s3_storage": s3_status
            },
            "mongodb": {
                "connected": db_healthy,
                "status": "online" if db_healthy else "offline",
                "active_tenants": active_tenants_count
            },
            "s3_storage": s3_status,
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

# Compatibility metrics endpoints for CDN/proxy path routing.
if ENABLE_METRICS:
    def _render_metrics() -> Response:
        if generate_latest is None:
            raise HTTPException(status_code=503, detail="prometheus_client unavailable")
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Backend-Server": "fastapi-async",
            },
        )

    @app.get("/api/metrics", include_in_schema=DEBUG_MODE)
    @limiter.limit("120/minute")
    async def metrics_api_alias(request: Request):
        return _render_metrics()

    @app.get("/api/v1/metrics", include_in_schema=DEBUG_MODE)
    @limiter.limit("120/minute")
    async def metrics_v1_alias(request: Request):
        return _render_metrics()

# Legacy compatibility endpoint for token verification
@app.get("/verify")
async def verify_token_legacy(request: Request):
    """Legacy /verify endpoint for frontend compatibility"""
    try:
        # Import the auth manager
        from api.v1.auth_async import get_current_user, get_auth_manager
        from fastapi import Depends, HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        auth_manager = request.app.state.auth
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        token = auth_header.split(" ", 1)[1]
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        current_user = await get_current_user(request, credentials, auth_manager)

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
