"""
Async Configuration for SkillBot Backend
Optimized for high concurrency and performance

Security-hardened for production deployment.
"""

import os
import secrets
import logging
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Configure logging early
logger = logging.getLogger(__name__)

# Load environment variables (prefer backend/.env explicitly)
BASE_DIR_PATH = Path(__file__).parent.absolute()
load_dotenv(BASE_DIR_PATH / ".env")
load_dotenv()


# =============================================================================
# SECURITY HELPER FUNCTIONS
# =============================================================================

def _get_cors_origins() -> List[str]:
    """
    Build CORS origins list based on environment.

    Security considerations:
    - In development: Allow localhost origins for local testing
    - In production: Only allow explicitly configured HTTPS origins
    - Never allow wildcards (*) or placeholder domains
    """
    is_dev = os.getenv("NODE_ENV", "production") == "development"

    # Production origins (always included when valid)
    production_origins: List[str] = []

    # Main production frontend
    main_frontend = "https://app.stoody.in"
    production_origins.append(main_frontend)

    # Mobile app origins (Capacitor/Cordova) - always allowed
    mobile_origins = [
        "capacitor://localhost",
        "ionic://localhost",
        "http://localhost",   # Android WebView
        "https://localhost",  # Capacitor Android with androidScheme: 'https'
    ]
    production_origins.extend(mobile_origins)

    # Add any additional production origins from environment (comma-separated)
    additional_origins = os.getenv("CORS_ADDITIONAL_ORIGINS", "")
    if additional_origins:
        for origin in additional_origins.split(","):
            origin = origin.strip()
            # Only allow HTTPS origins in production list
            if origin and origin.startswith("https://") and "your-" not in origin.lower():
                production_origins.append(origin)

    # Frontend URL from environment (for staging/preview deployments)
    frontend_url = os.getenv("FRONTEND_URL", "")
    if frontend_url:
        frontend_url = frontend_url.strip()
        # Reject placeholder URLs
        if "your-" in frontend_url.lower() or "example.com" in frontend_url.lower():
            logger.warning(f"⚠️ FRONTEND_URL contains placeholder value: {frontend_url}")
        elif frontend_url.startswith("https://"):
            production_origins.append(frontend_url)
        elif is_dev and frontend_url.startswith("http://"):
            # Allow HTTP only in development
            pass  # Will be added to dev_origins below

    if is_dev:
        # Development origins - only in dev mode
        dev_origins = [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",  # Vite dev server
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5001",  # Backend dev
            "http://127.0.0.1:5001",
        ]
        # Add frontend URL for dev (can be http)
        if frontend_url and frontend_url.startswith("http://") and frontend_url not in dev_origins:
            dev_origins.append(frontend_url)

        all_origins = list(set(dev_origins + production_origins))
        logger.info(f"🔧 Development CORS origins: {len(all_origins)} origins allowed")
        return all_origins

    # Production: Only HTTPS origins
    final_origins = list(set(production_origins))
    logger.info(f"🔒 Production CORS origins: {final_origins}")
    return final_origins


def _validate_jwt_secret(secret: Optional[str], is_production: bool) -> str:
    """
    Validate JWT secret key meets security requirements.

    Requirements:
    - Must be set (no fallback in production)
    - Must be at least 32 characters (256 bits of entropy)
    - Must not be a known weak/default value

    In development: Generates a random secret if not set (with warning)
    In production: Raises ValueError if requirements not met
    """
    WEAK_SECRETS = {
        "fallback-secret-key",
        "secret",
        "secret-key",
        "jwt-secret",
        "changeme",
        "your-secret-key",
        "super-secret",
        "development-secret",
        "test-secret",
        "dev-secret",
        "placeholder",
    }

    if not secret or secret.strip() == "":
        if is_production:
            raise ValueError(
                "\n" + "=" * 70 + "\n"
                "🚨 CRITICAL SECURITY ERROR: JWT_SECRET_KEY is not set!\n"
                "=" * 70 + "\n"
                "You MUST set a secure JWT_SECRET_KEY in production.\n\n"
                "Generate a secure secret with:\n"
                "  python -c \"import secrets; print(secrets.token_urlsafe(64))\"\n\n"
                "Then add to your .env file:\n"
                "  JWT_SECRET_KEY=<generated_secret>\n"
                "=" * 70
            )
        # Generate a random secret for development (not persistent across restarts)
        generated = secrets.token_urlsafe(64)
        logger.warning(
            "⚠️ JWT_SECRET_KEY not set - generating random secret for development.\n"
            "   This will invalidate ALL tokens on restart!\n"
            "   Set JWT_SECRET_KEY in .env for persistent sessions."
        )
        return generated

    secret = secret.strip()

    # Check for known weak secrets
    if secret.lower() in WEAK_SECRETS or len(secret) < 16:
        if is_production:
            raise ValueError(
                "\n" + "=" * 70 + "\n"
                f"🚨 CRITICAL SECURITY ERROR: JWT_SECRET_KEY is insecure!\n"
                "=" * 70 + "\n"
                f"Current value: '{secret[:10]}...' (length: {len(secret)})\n\n"
                "This secret is either:\n"
                "  - A known weak/default value, OR\n"
                "  - Too short (must be at least 32 characters)\n\n"
                "Generate a secure secret with:\n"
                "  python -c \"import secrets; print(secrets.token_urlsafe(64))\"\n"
                "=" * 70
            )
        logger.warning(
            f"⚠️ JWT_SECRET_KEY appears weak (value: '{secret[:5]}...', length: {len(secret)}).\n"
            "   Use a stronger secret in production!"
        )

    # Check minimum length (32 chars = ~192 bits with base64)
    if len(secret) < 32:
        if is_production:
            raise ValueError(
                "\n" + "=" * 70 + "\n"
                f"🚨 CRITICAL SECURITY ERROR: JWT_SECRET_KEY is too short!\n"
                "=" * 70 + "\n"
                f"Current length: {len(secret)} characters\n"
                "Required: At least 32 characters\n\n"
                "Generate a secure secret with:\n"
                "  python -c \"import secrets; print(secrets.token_urlsafe(64))\"\n"
                "=" * 70
            )
        logger.warning(
            f"⚠️ JWT_SECRET_KEY is short ({len(secret)} chars). "
            "Use at least 32 characters in production!"
        )

    return secret


def _validate_redis_url(redis_url: str, is_production: bool) -> str:
    """
    Validate Redis URL configuration.
    In production, Redis is required for rate limiting and caching.
    """
    if not redis_url or redis_url == "redis://localhost:6379/0":
        if is_production:
            logger.warning(
                "⚠️ REDIS_URL is using default localhost configuration.\n"
                "   Ensure Redis is properly configured for production!"
            )
    return redis_url


# =============================================================================
# SETTINGS CLASS
# =============================================================================

class AsyncSettings(BaseSettings):
    """Settings for async backend configuration"""

    # Base paths
    BASE_DIR: Path = Path(__file__).parent.absolute()

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 5001))
    DEBUG_MODE: bool = os.getenv("NODE_ENV", "production") == "development"

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "SkillBot Async API"
    VERSION: str = "2.0.0"

    # CORS settings - dynamically generated based on environment
    CORS_ORIGINS: List[str] = _get_cors_origins()

    # Database URLs
    MONGODB_URL: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "skillbot_db")
    MONGODB_DB_MASTER: str = os.getenv("MONGODB_DB_MASTER", "skb_master")
    # B2C Database (Stoody B2C) - Completely separate from skillbot_db
    # Note: MongoDB is case-sensitive - use exact case of existing database
    MONGODB_DB_STOODY: str = os.getenv("MONGODB_DB_STOODY", "STOODY-b2c")

    # Google OAuth Configuration for B2C Users
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")

    # Allow disabling MongoDB in dev, but default to enabled if URI is present
    _dev_default = "true" if os.getenv("NODE_ENV", "production") == "development" else "false"
    _default_disable = "false" if os.getenv("MONGODB_URI") else _dev_default
    DISABLE_MONGODB: bool = os.getenv("DISABLE_MONGODB", _default_disable).lower() == "true"

    # Redis configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    # Whether Redis is required (fail if unavailable). True in production.
    REDIS_REQUIRED: bool = os.getenv("REDIS_REQUIRED", "true" if os.getenv("NODE_ENV", "production") != "development" else "false").lower() == "true"

    # Connection pools
    MONGODB_MIN_POOL_SIZE: int = int(os.getenv("MONGODB_MIN_POOL_SIZE", 50))
    MONGODB_MAX_POOL_SIZE: int = int(os.getenv("MONGODB_MAX_POOL_SIZE", 500))

    # Cache configuration
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    CACHE_MAX_CONNECTIONS: int = int(os.getenv("CACHE_MAX_CONNECTIONS", 500))

    # Rate limiting
    RATE_LIMIT_DEFAULT: str = os.getenv("RATE_LIMIT_DEFAULT", "600/minute")
    RATE_LIMIT_AUTH: str = os.getenv("RATE_LIMIT_AUTH", "120/minute")
    RATE_LIMIT_UPLOAD: str = os.getenv("RATE_LIMIT_UPLOAD", "120/minute")

    # JWT Configuration - SECURE: No fallback, validated at runtime
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    JWT_ALGORITHM: str = "HS256"

    # Password Configuration
    PASSWORD_MIN_LENGTH: int = int(os.getenv("PASSWORD_MIN_LENGTH", 8))
    PASSWORD_MAX_LENGTH: int = 72  # bcrypt limitation - do not change
    PASSWORD_REQUIRE_UPPERCASE: bool = os.getenv("PASSWORD_REQUIRE_UPPERCASE", "false").lower() == "true"
    PASSWORD_REQUIRE_LOWERCASE: bool = os.getenv("PASSWORD_REQUIRE_LOWERCASE", "false").lower() == "true"
    PASSWORD_REQUIRE_DIGIT: bool = os.getenv("PASSWORD_REQUIRE_DIGIT", "false").lower() == "true"
    PASSWORD_REQUIRE_SPECIAL: bool = os.getenv("PASSWORD_REQUIRE_SPECIAL", "false").lower() == "true"

    # Password Reset Configuration
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("PASSWORD_RESET_TOKEN_EXPIRE_MINUTES", 30))
    PASSWORD_RESET_RATE_LIMIT: str = os.getenv("PASSWORD_RESET_RATE_LIMIT", "3/hour")

    # Super Admin Configuration
    SUPERADMIN_SETUP_KEY: str = os.getenv("SUPERADMIN_SETUP_KEY", "")

    # Email Configuration (for password reset, notifications)
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "noreply@stoody.in")
    SMTP_FROM_NAME: str = os.getenv("SMTP_FROM_NAME", "Stoody")
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

    # File storage
    IMAGES_DIR: Path = BASE_DIR / "images"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: set = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", 2000))
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", 0.7))

    # Qdrant Vector Database Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "https://941b0324-5307-4db1-ba43-bddfeba2a93a.sa-east-1-0.aws.cloud.qdrant.io:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION_PREFIX: str = os.getenv("QDRANT_COLLECTION_PREFIX", "stoody_notes")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1536))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
    
    # Text Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Question Generation Configuration
    QUESTION_GEN_MODEL: str = os.getenv("QUESTION_GEN_MODEL", "gpt-4o")
    QUESTION_GEN_TEMPERATURE: float = float(os.getenv("QUESTION_GEN_TEMPERATURE", 0.7))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", 10))
    
    # Kimi/Moonshot AI Configuration (for question generation)
    KIMI_API_KEY: str = os.getenv("KIMI_API_KEY", os.getenv("MOONSHOT_API_KEY", ""))
    KIMI_BASE_URL: str = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/v1")
    KIMI_MODEL: str = os.getenv("KIMI_MODEL", "kimi-k2.5")
    KIMI_VISION_MODEL: str = os.getenv("KIMI_VISION_MODEL", "moonshot-v1-32k-vision-preview")
    USE_KIMI_FOR_QUESTIONS: bool = os.getenv("USE_KIMI_FOR_QUESTIONS", "true").lower() == "true"
    USE_KIMI_FOR_DIAGRAMS: bool = os.getenv("USE_KIMI_FOR_DIAGRAMS", "true").lower() == "true"

    # Async configuration
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 8))
    WORKER_CONNECTIONS: int = int(os.getenv("WORKER_CONNECTIONS", 2000))
    OPENAI_CONCURRENCY_LIMIT: int = int(os.getenv("OPENAI_CONCURRENCY_LIMIT", 200))
    OCR_TIMEOUT_SECONDS: int = int(os.getenv("OCR_TIMEOUT_SECONDS", 180))
    OCR_CONCURRENCY_LIMIT: int = int(os.getenv("OCR_CONCURRENCY_LIMIT", 8))

    # Background task configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # Performance tuning
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", 3600))
    DB_POOL_PRE_PING: bool = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"

    # Monitoring and logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if os.getenv("NODE_ENV", "production") == "development" else "WARNING")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"

    # Security Logging
    SECURITY_LOG_ENABLED: bool = os.getenv("SECURITY_LOG_ENABLED", "true").lower() == "true"
    SECURITY_LOG_FILE: str = os.getenv("SECURITY_LOG_FILE", "logs/security.log")

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in .env file


# =============================================================================
# CREATE AND VALIDATE SETTINGS
# =============================================================================

# Create settings instance
settings = AsyncSettings()

# Validate JWT secret key (this will raise if invalid in production)
_validated_jwt_secret = _validate_jwt_secret(
    os.getenv("JWT_SECRET_KEY", ""),
    not settings.DEBUG_MODE
)

# Validate superadmin JWT secret (use main JWT secret as fallback)
_validated_superadmin_secret = _validate_jwt_secret(
    os.getenv("SUPERADMIN_JWT_SECRET", _validated_jwt_secret),
    not settings.DEBUG_MODE
)

# Validate Redis URL
_validate_redis_url(settings.REDIS_URL, not settings.DEBUG_MODE)

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

BASE_DIR = settings.BASE_DIR
HOST = settings.HOST
PORT = settings.PORT
DEBUG_MODE = settings.DEBUG_MODE
API_V1_PREFIX = settings.API_V1_PREFIX
CORS_ORIGINS = settings.CORS_ORIGINS
MONGODB_URL = settings.MONGODB_URL
MONGODB_DB_NAME = settings.MONGODB_DB_NAME
MONGODB_DB_MASTER = settings.MONGODB_DB_MASTER
MONGODB_DB_STOODY = settings.MONGODB_DB_STOODY
GOOGLE_CLIENT_ID = settings.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
DISABLE_MONGODB = settings.DISABLE_MONGODB
REDIS_URL = settings.REDIS_URL
REDIS_REQUIRED = settings.REDIS_REQUIRED
RATE_LIMIT_DEFAULT = settings.RATE_LIMIT_DEFAULT
RATE_LIMIT_AUTH = settings.RATE_LIMIT_AUTH
RATE_LIMIT_UPLOAD = settings.RATE_LIMIT_UPLOAD

# JWT secrets - use validated values
JWT_SECRET_KEY = _validated_jwt_secret
SUPERADMIN_JWT_SECRET = _validated_superadmin_secret
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
JWT_ALGORITHM = settings.JWT_ALGORITHM

# Password configuration
PASSWORD_MIN_LENGTH = settings.PASSWORD_MIN_LENGTH
PASSWORD_MAX_LENGTH = settings.PASSWORD_MAX_LENGTH
PASSWORD_RESET_TOKEN_EXPIRE_MINUTES = settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES

# Email configuration
SMTP_HOST = settings.SMTP_HOST
SMTP_PORT = settings.SMTP_PORT
SMTP_USER = settings.SMTP_USER
SMTP_PASSWORD = settings.SMTP_PASSWORD
SMTP_FROM_EMAIL = settings.SMTP_FROM_EMAIL
SMTP_FROM_NAME = settings.SMTP_FROM_NAME
SMTP_USE_TLS = settings.SMTP_USE_TLS

IMAGES_DIR = settings.IMAGES_DIR
MAX_IMAGE_SIZE = settings.MAX_IMAGE_SIZE
ALLOWED_IMAGE_EXTENSIONS = settings.ALLOWED_IMAGE_EXTENSIONS
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_MODEL = settings.OPENAI_MODEL

# Kimi/Moonshot configuration
KIMI_API_KEY = settings.KIMI_API_KEY
KIMI_BASE_URL = settings.KIMI_BASE_URL
KIMI_MODEL = settings.KIMI_MODEL
USE_KIMI_FOR_QUESTIONS = settings.USE_KIMI_FOR_QUESTIONS
USE_KIMI_FOR_DIAGRAMS = settings.USE_KIMI_FOR_DIAGRAMS

# Qdrant configuration
QDRANT_URL = settings.QDRANT_URL
QDRANT_API_KEY = settings.QDRANT_API_KEY
QDRANT_COLLECTION_PREFIX = settings.QDRANT_COLLECTION_PREFIX

# Embedding configuration
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = settings.EMBEDDING_DIMENSIONS
EMBEDDING_BATCH_SIZE = settings.EMBEDDING_BATCH_SIZE

# Chunking configuration
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP

# Question generation configuration
QUESTION_GEN_MODEL = settings.QUESTION_GEN_MODEL
QUESTION_GEN_TEMPERATURE = settings.QUESTION_GEN_TEMPERATURE
MAX_CONTEXT_CHUNKS = settings.MAX_CONTEXT_CHUNKS

MAX_WORKERS = settings.MAX_WORKERS
WORKER_CONNECTIONS = settings.WORKER_CONNECTIONS
OPENAI_CONCURRENCY_LIMIT = settings.OPENAI_CONCURRENCY_LIMIT
OCR_TIMEOUT_SECONDS = settings.OCR_TIMEOUT_SECONDS
OCR_CONCURRENCY_LIMIT = settings.OCR_CONCURRENCY_LIMIT

# Security logging
SECURITY_LOG_ENABLED = settings.SECURITY_LOG_ENABLED
SECURITY_LOG_FILE = settings.SECURITY_LOG_FILE

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

# Ensure directories exist
IMAGES_DIR.mkdir(exist_ok=True)

# Create logs directory for security logging
logs_dir = BASE_DIR / "logs"
logs_dir.mkdir(exist_ok=True)

# =============================================================================
# PRODUCTION VALIDATION
# =============================================================================

if not DEBUG_MODE:
    # Production-only validations
    if not MONGODB_URL:
        raise ValueError("MONGODB_URL is required in production")

    if not OPENAI_API_KEY:
        logger.warning("⚠️ OPENAI_API_KEY not set - AI features will be disabled")

    # Log security status
    logger.info("🔒 Production security validations passed")
    logger.info(f"   - JWT secret: {'✓ Configured' if len(JWT_SECRET_KEY) >= 32 else '⚠️ Weak'}")
    logger.info(f"   - CORS origins: {len(CORS_ORIGINS)} allowed")
    logger.info(f"   - Redis required: {REDIS_REQUIRED}")
else:
    logger.info("🔧 Running in DEVELOPMENT mode - some security checks relaxed")
