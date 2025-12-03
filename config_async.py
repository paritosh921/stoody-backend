"""
Async Configuration for SkillBot Backend
Optimized for high concurrency and performance
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables (prefer backend/.env explicitly)
BASE_DIR_PATH = Path(__file__).parent.absolute()
load_dotenv(BASE_DIR_PATH / ".env")
load_dotenv()

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

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        os.getenv("FRONTEND_URL", "http://localhost:8080"),
        "https://app.stoody.in",  # Production frontend
        "https://your-frontend-domain.vercel.app"  # Add your Vercel URL
    ]

    # Database URLs
    MONGODB_URL: str = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "skillbot_db")
    # Allow disabling MongoDB in dev, but default to enabled if URI is present
    _dev_default = "true" if os.getenv("NODE_ENV", "production") == "development" else "false"
    _default_disable = "false" if os.getenv("MONGODB_URI") else _dev_default
    DISABLE_MONGODB: bool = os.getenv("DISABLE_MONGODB", _default_disable).lower() == "true"

    # Redis configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ChromaDB configuration
    CHROMADB_PATH: Path = BASE_DIR / "chromadb_data"
    CHROMADB_COLLECTION_NAME: str = "questions"

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

    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "fallback-secret-key")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    JWT_ALGORITHM: str = "HS256"

    # File storage
    IMAGES_DIR: Path = BASE_DIR / "images"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_EXTENSIONS: set = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5")  # GPT-5 for enhanced reasoning and multimodal processing
    OPENAI_MAX_TOKENS: int = int(os.getenv("OPENAI_MAX_TOKENS", 2000))  # Increased for detailed explanations
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", 0.7))

    # Async configuration
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 8))
    WORKER_CONNECTIONS: int = int(os.getenv("WORKER_CONNECTIONS", 2000))
    OPENAI_CONCURRENCY_LIMIT: int = int(os.getenv("OPENAI_CONCURRENCY_LIMIT", 200))
    OCR_TIMEOUT_SECONDS: int = int(os.getenv("OCR_TIMEOUT_SECONDS", 180))
    OCR_CONCURRENCY_LIMIT: int = int(os.getenv("OCR_CONCURRENCY_LIMIT", 8))

    # Background task configuration
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

    # Performance tuning
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", 3600))
    DB_POOL_PRE_PING: bool = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"

    # Monitoring and logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if DEBUG_MODE else "WARNING")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "false").lower() == "true"

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"  # Allow extra fields in .env file

# Create settings instance
settings = AsyncSettings()

# Export commonly used settings
BASE_DIR = settings.BASE_DIR
HOST = settings.HOST
PORT = settings.PORT
DEBUG_MODE = settings.DEBUG_MODE
API_V1_PREFIX = settings.API_V1_PREFIX
CORS_ORIGINS = settings.CORS_ORIGINS
MONGODB_URL = settings.MONGODB_URL
MONGODB_DB_NAME = settings.MONGODB_DB_NAME
DISABLE_MONGODB = settings.DISABLE_MONGODB
REDIS_URL = settings.REDIS_URL
CHROMADB_PATH = settings.CHROMADB_PATH
CHROMADB_COLLECTION_NAME = settings.CHROMADB_COLLECTION_NAME
RATE_LIMIT_DEFAULT = settings.RATE_LIMIT_DEFAULT
RATE_LIMIT_AUTH = settings.RATE_LIMIT_AUTH
RATE_LIMIT_UPLOAD = settings.RATE_LIMIT_UPLOAD
JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
JWT_ALGORITHM = settings.JWT_ALGORITHM
IMAGES_DIR = settings.IMAGES_DIR
MAX_IMAGE_SIZE = settings.MAX_IMAGE_SIZE
ALLOWED_IMAGE_EXTENSIONS = settings.ALLOWED_IMAGE_EXTENSIONS
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_MODEL = settings.OPENAI_MODEL
MAX_WORKERS = settings.MAX_WORKERS
WORKER_CONNECTIONS = settings.WORKER_CONNECTIONS
OPENAI_CONCURRENCY_LIMIT = settings.OPENAI_CONCURRENCY_LIMIT
OCR_TIMEOUT_SECONDS = settings.OCR_TIMEOUT_SECONDS
OCR_CONCURRENCY_LIMIT = settings.OCR_CONCURRENCY_LIMIT

# Ensure directories exist
CHROMADB_PATH.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Validation
if not MONGODB_URL and not DEBUG_MODE:
    raise ValueError("MONGODB_URL is required in production")

if not OPENAI_API_KEY and not DEBUG_MODE:
    raise ValueError("OPENAI_API_KEY is required for AI functionality")
