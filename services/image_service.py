import os
import uuid
import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import io

from config import IMAGES_DIR, MAX_IMAGE_SIZE, ALLOWED_IMAGE_EXTENSIONS

class ImageService:
    """Service for managing question image storage and retrieval"""
    
    def __init__(self):
        self.images_dir = Path(IMAGES_DIR)
        self.images_dir.mkdir(exist_ok=True)
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    
    def validate_image_size(self, image_data: bytes) -> bool:
        """Check if image size is within limits"""
        return len(image_data) <= MAX_IMAGE_SIZE
    
    def save_base64_image(self, base64_data: str, original_filename: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save base64 image data to file system
        
        Returns:
            (success, file_path, error_message)
        """
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64 data
            try:
                image_data = base64.b64decode(base64_data)
            except Exception as e:
                return False, None, f"Invalid base64 data: {str(e)}"
            
            # Validate file size
            if not self.validate_image_size(image_data):
                return False, None, f"Image size exceeds maximum limit of {MAX_IMAGE_SIZE // (1024*1024)}MB"
            
            # Validate file extension
            if not self.is_allowed_file(original_filename):
                return False, None, f"File type not allowed. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            
            # Generate unique filename
            file_extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}_{original_filename}"
            file_path = self.images_dir / unique_filename
            
            # Validate image by opening with PIL
            try:
                with Image.open(io.BytesIO(image_data)) as img:
                    # Convert to RGB if necessary (for JPEG compatibility)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Save the image
                    img.save(file_path, format=file_extension.upper() if file_extension.upper() != 'JPG' else 'JPEG')
                    
            except Exception as e:
                return False, None, f"Invalid image data: {str(e)}"
            
            # Return relative path for storage in database
            relative_path = f"images/{unique_filename}"
            logging.info(f"Saved image: {relative_path}")
            
            return True, relative_path, None
            
        except Exception as e:
            logging.error(f"Failed to save image {original_filename}: {str(e)}")
            return False, None, f"Failed to save image: {str(e)}"
    
    def save_uploaded_file(self, file_data: bytes, filename: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Save uploaded file data to file system
        
        Returns:
            (success, file_path, error_message)
        """
        try:
            # Validate file size
            if not self.validate_image_size(file_data):
                return False, None, f"Image size exceeds maximum limit of {MAX_IMAGE_SIZE // (1024*1024)}MB"
            
            # Validate file extension
            if not self.is_allowed_file(filename):
                return False, None, f"File type not allowed. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            
            # Generate unique filename
            file_extension = filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = self.images_dir / unique_filename
            
            # Validate and save image
            try:
                with Image.open(io.BytesIO(file_data)) as img:
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Save the image
                    img.save(file_path, format=file_extension.upper() if file_extension.upper() != 'JPG' else 'JPEG')
                    
            except Exception as e:
                return False, None, f"Invalid image data: {str(e)}"
            
            # Return relative path for storage in database
            relative_path = f"images/{unique_filename}"
            logging.info(f"Saved uploaded image: {relative_path}")
            
            return True, relative_path, None
            
        except Exception as e:
            logging.error(f"Failed to save uploaded file {filename}: {str(e)}")
            return False, None, f"Failed to save image: {str(e)}"
    
    def get_image_path(self, relative_path: str) -> Optional[Path]:
        """Get absolute path for a stored image"""
        try:
            # Remove 'images/' prefix if present
            if relative_path.startswith('images/'):
                filename = relative_path[7:]  # Remove 'images/' prefix
            else:
                filename = relative_path
            
            file_path = self.images_dir / filename
            
            if file_path.exists() and file_path.is_file():
                return file_path
            else:
                logging.warning(f"Image not found: {relative_path}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to get image path for {relative_path}: {str(e)}")
            return None
    
    def get_image_base64(self, relative_path: str) -> Optional[str]:
        """Get base64 encoded image data"""
        try:
            file_path = self.get_image_path(relative_path)
            if not file_path:
                return None
            
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Determine MIME type from file extension
            file_extension = file_path.suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }.get(file_extension, 'image/jpeg')
            
            # Encode to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            logging.error(f"Failed to get base64 for {relative_path}: {str(e)}")
            return None
    
    def delete_image(self, relative_path: str) -> bool:
        """Delete an image file"""
        try:
            file_path = self.get_image_path(relative_path)
            if not file_path:
                return False
            
            file_path.unlink()
            logging.info(f"Deleted image: {relative_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete image {relative_path}: {str(e)}")
            return False
    
    def get_image_info(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a stored image"""
        try:
            file_path = self.get_image_path(relative_path)
            if not file_path:
                return None
            
            stat = file_path.stat()
            
            # Get image dimensions
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    format_name = img.format
            except Exception:
                width = height = format_name = None
            
            return {
                'filename': file_path.name,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'width': width,
                'height': height,
                'format': format_name
            }
            
        except Exception as e:
            logging.error(f"Failed to get image info for {relative_path}: {str(e)}")
            return None
    
    def cleanup_orphaned_images(self, valid_image_paths: set) -> int:
        """Remove images that are no longer referenced in the database"""
        try:
            deleted_count = 0
            
            for image_file in self.images_dir.iterdir():
                if image_file.is_file():
                    relative_path = f"images/{image_file.name}"
                    if relative_path not in valid_image_paths:
                        try:
                            image_file.unlink()
                            deleted_count += 1
                            logging.info(f"Deleted orphaned image: {relative_path}")
                        except Exception as e:
                            logging.error(f"Failed to delete orphaned image {relative_path}: {str(e)}")
            
            return deleted_count
            
        except Exception as e:
            logging.error(f"Failed to cleanup orphaned images: {str(e)}")
            return 0

# Global instance
image_service = ImageService()