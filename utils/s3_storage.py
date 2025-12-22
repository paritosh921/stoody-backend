"""
S3 Storage Service for SkillBot
Provides drop-in replacement for local file storage with Amazon S3

Features:
- Async upload/download operations
- Automatic content-type detection
- Presigned URL generation for secure access
- Fallback to local storage for development
- Compatible with existing file paths and URLs

Environment Variables:
- AWS_ACCESS_KEY_ID: AWS access key
- AWS_SECRET_ACCESS_KEY: AWS secret key
- AWS_REGION: AWS region (default: ap-south-1)
- S3_BUCKET_NAME: Your S3 bucket name
- USE_S3_STORAGE: Set to 'true' to enable S3 (default: 'false' for local)
- CLOUDFRONT_DOMAIN: Optional CloudFront domain for faster delivery
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import mimetypes

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, assume env vars are set externally

logger = logging.getLogger(__name__)

# Configuration from environment
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
USE_S3_STORAGE = os.getenv("USE_S3_STORAGE", "false").lower() == "true"
CLOUDFRONT_DOMAIN = os.getenv("CLOUDFRONT_DOMAIN", "")

# S3 client (lazy initialization)
_s3_client = None


def _get_s3_client():
    """Get or create S3 client (lazy initialization)"""
    global _s3_client
    
    if _s3_client is None and USE_S3_STORAGE:
        try:
            import boto3
            from botocore.config import Config
            
            config = Config(
                region_name=AWS_REGION,
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
            
            _s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                config=config
            )
            logger.info(f"✅ S3 client initialized for bucket: {S3_BUCKET_NAME}")
        except ImportError:
            logger.error("❌ boto3 not installed. Run: pip install boto3")
            _s3_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
            _s3_client = None
    
    return _s3_client


def is_s3_enabled() -> bool:
    """Check if S3 storage is enabled and properly configured"""
    if not USE_S3_STORAGE:
        return False
    
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
        logger.warning("S3_STORAGE is enabled but credentials are missing")
        return False
    
    return _get_s3_client() is not None


def _local_path_to_s3_key(local_path: str) -> str:
    """
    Convert local file path to S3 key.
    
    Examples:
    - uploads/documents/Test Series/doc001.pdf -> documents/Test Series/doc001.pdf
    - uploads/pdf_images/doc001/img-1.jpeg -> pdf_images/doc001/img-1.jpeg
    """
    # Normalize path separators
    path = local_path.replace("\\", "/")
    
    # Remove leading path components to get relative path
    if "uploads/" in path:
        return path.split("uploads/", 1)[1]
    
    return path


def _s3_key_to_local_path(s3_key: str) -> str:
    """Convert S3 key back to local path (for fallback)"""
    return os.path.join(os.getcwd(), "uploads", s3_key.replace("/", os.sep))


def get_content_type(filename: str) -> str:
    """Detect content type from filename"""
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


async def upload_file(
    file_data: bytes,
    local_path: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None
) -> Tuple[bool, str]:
    """
    Upload a file to S3 or local filesystem.
    
    Args:
        file_data: Binary file content
        local_path: Local file path (will be converted to S3 key)
        content_type: MIME type (auto-detected if not provided)
        metadata: Optional metadata to store with the file
    
    Returns:
        Tuple of (success: bool, storage_path: str)
    """
    if is_s3_enabled():
        return await _upload_to_s3(file_data, local_path, content_type, metadata)
    else:
        return await _upload_to_local(file_data, local_path)


async def _upload_to_s3(
    file_data: bytes,
    local_path: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None
) -> Tuple[bool, str]:
    """Upload file to S3"""
    try:
        s3_client = _get_s3_client()
        if not s3_client:
            logger.warning("S3 client not available, falling back to local storage")
            return await _upload_to_local(file_data, local_path)
        
        s3_key = _local_path_to_s3_key(local_path)
        
        # Detect content type if not provided
        if not content_type:
            content_type = get_content_type(local_path)
        
        # Prepare extra args
        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata
        
        # Run S3 upload in thread pool (boto3 is synchronous)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=file_data,
                **extra_args
            )
        )
        
        # Return S3 path in consistent format
        s3_path = f"s3://{S3_BUCKET_NAME}/{s3_key}"
        logger.info(f"✅ Uploaded to S3: {s3_key} ({len(file_data)} bytes)")
        return True, s3_path
        
    except Exception as e:
        logger.error(f"❌ S3 upload failed for {local_path}: {e}")
        # Fallback to local storage
        logger.info("Falling back to local storage...")
        return await _upload_to_local(file_data, local_path)


async def _upload_to_local(file_data: bytes, local_path: str) -> Tuple[bool, str]:
    """Upload file to local filesystem"""
    try:
        import aiofiles
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        async with aiofiles.open(local_path, "wb") as f:
            await f.write(file_data)
        
        logger.info(f"✅ Saved locally: {local_path} ({len(file_data)} bytes)")
        return True, local_path
        
    except Exception as e:
        logger.error(f"❌ Local upload failed for {local_path}: {e}")
        return False, ""


async def download_file(storage_path: str) -> Optional[bytes]:
    """
    Download a file from S3 or local filesystem.
    
    Args:
        storage_path: Either S3 path (s3://...) or local file path
    
    Returns:
        File content as bytes, or None if not found
    """
    if storage_path.startswith("s3://"):
        return await _download_from_s3(storage_path)
    else:
        return await _download_from_local(storage_path)


async def _download_from_s3(s3_path: str) -> Optional[bytes]:
    """Download file from S3"""
    try:
        s3_client = _get_s3_client()
        if not s3_client:
            # Try local fallback
            local_path = _s3_key_to_local_path(s3_path.replace(f"s3://{S3_BUCKET_NAME}/", ""))
            return await _download_from_local(local_path)
        
        # Parse S3 path
        s3_key = s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
        
        # Run S3 download in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        )
        
        file_data = response['Body'].read()
        logger.info(f"✅ Downloaded from S3: {s3_key} ({len(file_data)} bytes)")
        return file_data
        
    except Exception as e:
        logger.error(f"❌ S3 download failed for {s3_path}: {e}")
        return None


async def _download_from_local(local_path: str) -> Optional[bytes]:
    """Download file from local filesystem"""
    try:
        import aiofiles
        
        if not os.path.exists(local_path):
            logger.warning(f"Local file not found: {local_path}")
            return None
        
        async with aiofiles.open(local_path, "rb") as f:
            file_data = await f.read()
        
        return file_data
        
    except Exception as e:
        logger.error(f"❌ Local download failed for {local_path}: {e}")
        return None


def get_public_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Get a public URL for a file.
    
    For S3: Returns presigned URL or CloudFront URL
    For local: Returns API endpoint path
    
    Args:
        storage_path: Either S3 path or local file path
        expires_in: URL expiration in seconds (for presigned URLs)
    
    Returns:
        Public URL to access the file
    """
    if storage_path.startswith("s3://"):
        return _get_s3_url(storage_path, expires_in)
    else:
        return _get_local_url(storage_path)


def _get_s3_url(s3_path: str, expires_in: int = 3600) -> str:
    """Get presigned URL or CloudFront URL for S3 object"""
    s3_key = s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
    
    # Use CloudFront if configured (faster, cached)
    if CLOUDFRONT_DOMAIN:
        return f"https://{CLOUDFRONT_DOMAIN}/{s3_key}"
    
    # Generate presigned URL
    try:
        s3_client = _get_s3_client()
        if s3_client:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key},
                ExpiresIn=expires_in
            )
            return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
    
    # Fallback to direct S3 URL (requires public access)
    return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"


def _get_local_url(local_path: str) -> str:
    """Get API URL for local file"""
    # Convert local path to API endpoint
    relative_path = local_path.replace(os.getcwd(), "").replace("\\", "/").lstrip("/")
    
    # Check if it's an image
    if "pdf_images" in relative_path or "images" in relative_path:
        # Extract image ID from path
        filename = os.path.basename(local_path)
        image_id = os.path.splitext(filename)[0]
        return f"/api/v1/images/{image_id}"
    
    # For documents/PDFs
    if "documents" in relative_path:
        return f"/api/v1/pdf/{relative_path}"
    
    return f"/static/{relative_path}"


async def delete_file(storage_path: str) -> bool:
    """
    Delete a file from S3 or local filesystem.
    
    Args:
        storage_path: Either S3 path or local file path
    
    Returns:
        True if deletion was successful
    """
    if storage_path.startswith("s3://"):
        return await _delete_from_s3(storage_path)
    else:
        return await _delete_from_local(storage_path)


async def _delete_from_s3(s3_path: str) -> bool:
    """Delete file from S3"""
    try:
        s3_client = _get_s3_client()
        if not s3_client:
            return False
        
        s3_key = s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        )
        
        logger.info(f"✅ Deleted from S3: {s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 delete failed for {s3_path}: {e}")
        return False


async def _delete_from_local(local_path: str) -> bool:
    """Delete file from local filesystem"""
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info(f"✅ Deleted locally: {local_path}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"❌ Local delete failed for {local_path}: {e}")
        return False


async def file_exists(storage_path: str) -> bool:
    """Check if a file exists in storage"""
    if storage_path.startswith("s3://"):
        return await _s3_file_exists(storage_path)
    else:
        return os.path.exists(storage_path)


async def _s3_file_exists(s3_path: str) -> bool:
    """Check if file exists in S3"""
    try:
        s3_client = _get_s3_client()
        if not s3_client:
            return False
        
        s3_key = s3_path.replace(f"s3://{S3_BUCKET_NAME}/", "")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        )
        return True
        
    except Exception:
        return False


# ============================================================================
# MIGRATION HELPERS - For transitioning from local to S3
# ============================================================================

async def migrate_file_to_s3(local_path: str) -> Optional[str]:
    """
    Migrate a single file from local storage to S3.
    
    Args:
        local_path: Path to local file
    
    Returns:
        S3 path if successful, None if failed
    """
    if not is_s3_enabled():
        logger.warning("S3 storage is not enabled")
        return None
    
    file_data = await _download_from_local(local_path)
    if file_data is None:
        logger.error(f"File not found: {local_path}")
        return None
    
    success, s3_path = await _upload_to_s3(file_data, local_path)
    if success:
        return s3_path
    return None


async def migrate_directory_to_s3(local_dir: str, dry_run: bool = True) -> Dict[str, Any]:
    """
    Migrate an entire directory from local storage to S3.
    
    Args:
        local_dir: Local directory path (e.g., "uploads/documents")
        dry_run: If True, only report what would be migrated
    
    Returns:
        Migration report with counts and errors
    """
    report = {
        "total_files": 0,
        "migrated": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "dry_run": dry_run
    }
    
    if not os.path.exists(local_dir):
        report["errors"].append(f"Directory not found: {local_dir}")
        return report
    
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            report["total_files"] += 1
            
            if dry_run:
                s3_key = _local_path_to_s3_key(local_path)
                logger.info(f"[DRY RUN] Would migrate: {local_path} -> s3://{S3_BUCKET_NAME}/{s3_key}")
                continue
            
            try:
                s3_path = await migrate_file_to_s3(local_path)
                if s3_path:
                    report["migrated"] += 1
                else:
                    report["failed"] += 1
            except Exception as e:
                report["failed"] += 1
                report["errors"].append(f"{local_path}: {str(e)}")
    
    return report


# ============================================================================
# USAGE EXAMPLE (for reference)
# ============================================================================
"""
# In your pdf_async.py, replace:

# OLD (local storage):
async with aiofiles.open(original_file_path, "wb") as f:
    await f.write(image_data)

# NEW (S3-compatible):
from utils.s3_storage import upload_file
success, storage_path = await upload_file(image_data, original_file_path, content_type)

# The storage_path can be stored in MongoDB and used later with get_public_url()
"""
