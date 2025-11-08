import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import timedelta

# Base directory for the backend
BASE_DIR = Path(__file__).parent.absolute()

# Load environment variables
# Ensure we load the .env that lives in the backend directory
# even if the server is started from the repo root.
load_dotenv(BASE_DIR / ".env")
# Also load any process-level or root-level .env if present
load_dotenv()

# Database configuration
CHROMADB_PATH = BASE_DIR / "chromadb_data"
CHROMADB_COLLECTION_NAME = "questions"

# MongoDB configuration
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'skillbot_db')

# Authentication configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'fallback-secret-key')
JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600)))

# Image storage configuration
IMAGES_DIR = BASE_DIR / "images"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Flask configuration
FLASK_PORT = int(os.getenv('PORT', 5001))
FLASK_DEBUG = os.getenv('NODE_ENV', 'production') == 'development'
CORS_ORIGINS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    os.getenv('FRONTEND_URL', 'http://localhost:8080')
]

# Create directories if they don't exist
CHROMADB_PATH.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
