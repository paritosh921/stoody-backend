"""
Path utilities for consistent file storage
Always stores relative paths to MongoDB for cross-platform compatibility
"""

import os
from pathlib import Path
from typing import Union

# Get backend directory (this file is in backend/utils/)
BACKEND_DIR = Path(__file__).parent.parent.absolute()


def get_relative_path(absolute_path: Union[str, Path]) -> str:
    """
    Convert absolute path to relative path from backend root

    Args:
        absolute_path: Absolute file path

    Returns:
        Relative path string (e.g., "uploads/pdf_images/doc1/img-1.jpg")
    """
    try:
        abs_path = Path(absolute_path).absolute()
        rel_path = abs_path.relative_to(BACKEND_DIR)
        # Always use forward slashes for consistency across platforms
        return str(rel_path).replace('\\', '/')
    except ValueError:
        # If path is not relative to backend dir, return as-is but log warning
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Path {absolute_path} is not relative to backend dir {BACKEND_DIR}")
        return str(absolute_path).replace('\\', '/')


def get_absolute_path(relative_path: Union[str, Path]) -> Path:
    """
    Convert relative path to absolute path

    Args:
        relative_path: Relative path from backend root

    Returns:
        Absolute Path object
    """
    # Handle both forward and backward slashes
    rel_path_str = str(relative_path).replace('\\', '/')
    return BACKEND_DIR / rel_path_str


def ensure_relative_storage(file_path: Union[str, Path]) -> str:
    """
    Ensure path is stored as relative for MongoDB
    If already relative, return as-is. If absolute, convert to relative.

    Args:
        file_path: File path (absolute or relative)

    Returns:
        Relative path string safe for MongoDB storage
    """
    path_str = str(file_path)

    # If path starts with /, C:, D:, etc., it's absolute
    if path_str.startswith('/') or (len(path_str) > 1 and path_str[1] == ':'):
        return get_relative_path(file_path)

    # Already relative
    return path_str.replace('\\', '/')


def file_exists(path: Union[str, Path]) -> bool:
    """
    Check if file exists (handles both relative and absolute paths)

    Args:
        path: File path (relative or absolute)

    Returns:
        True if file exists
    """
    if Path(path).is_absolute():
        return Path(path).exists()
    else:
        return get_absolute_path(path).exists()