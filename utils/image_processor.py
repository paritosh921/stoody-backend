"""
Image Processing Utilities for Canvas Answer Evaluation
Provides image enhancement, preprocessing, and format conversion for better OCR/LLM recognition.
"""

import base64
import io
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import PIL for image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not installed. Image enhancement will be skipped. Install with: pip install Pillow")


def decode_base64_image(data_url: str) -> Optional[bytes]:
    """Decode a base64 data URL to raw bytes."""
    try:
        if not data_url:
            return None
        
        # Handle data URL format
        if data_url.startswith('data:'):
            # Extract base64 portion after comma
            parts = data_url.split(',', 1)
            if len(parts) == 2:
                return base64.b64decode(parts[1])
        
        # Assume raw base64
        return base64.b64decode(data_url)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def encode_image_to_base64(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Encode image bytes to a data URL."""
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{b64}"


def enhance_canvas_image(data_url: str, target_width: int = 1500) -> str:
    """
    Enhance a canvas image for better OCR/LLM recognition.
    
    Improvements:
    1. Upscale to higher resolution
    2. Increase contrast
    3. Apply light sharpening
    4. Ensure solid white background
    5. Thicken strokes slightly
    
    Args:
        data_url: Base64 data URL of the canvas image
        target_width: Target width for the enhanced image
        
    Returns:
        Enhanced image as base64 data URL
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, returning original image")
        return data_url
    
    try:
        # Decode the image
        image_bytes = decode_base64_image(data_url)
        if not image_bytes:
            return data_url
        
        # Open with PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGBA to handle transparency
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Create white background and composite
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
        
        # Convert to RGB for processing
        img = img.convert('RGB')
        
        # Calculate new size maintaining aspect ratio
        original_width, original_height = img.size
        upscale_factor = 1.0
        if original_width < target_width:
            upscale_factor = target_width / max(original_width, 1)
            new_height = int(original_height * upscale_factor)
            img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        
        # Mild contrast boost — GPT-5.1 reads faint strokes well, avoid over-processing
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)

        # Apply slight sharpening
        img = img.filter(ImageFilter.SHARPEN)

        # Increase brightness slightly to ensure white background
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)

        # NOTE: Grayscale conversion and stroke dilation removed.
        # GPT-5.1 vision reads color images and thin strokes natively.
        # Grayscale was destroying student color-coded work (blue/red/green),
        # and dilation was merging closely-spaced lines in detailed diagrams.
        
        # Save to bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='PNG', quality=95)
        output_bytes = output_buffer.getvalue()
        
        # Encode back to data URL
        return encode_image_to_base64(output_bytes, "image/png")
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}", exc_info=True)
        return data_url


def enhance_canvas_images_batch(data_urls: List[str], target_width: int = 1500) -> List[str]:
    """
    Enhance multiple canvas images.
    
    Args:
        data_urls: List of base64 data URLs
        target_width: Target width for enhanced images
        
    Returns:
        List of enhanced images as base64 data URLs
    """
    enhanced = []
    for url in data_urls:
        if url:
            enhanced.append(enhance_canvas_image(url, target_width))
    return enhanced


def get_image_dimensions(data_url: str) -> Optional[Tuple[int, int]]:
    """Get the dimensions of a base64-encoded image."""
    if not PIL_AVAILABLE:
        return None
    
    try:
        img_bytes = decode_base64_image(data_url)
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            return img.size
    except Exception as e:
        logger.error(f"Failed to get image dimensions: {e}")
    
    return None


def is_canvas_empty(data_url: str, threshold: float = 0.99) -> bool:
    """
    Check if a canvas image is essentially empty (mostly white/transparent).
    
    Args:
        data_url: Base64 data URL of the canvas
        threshold: Percentage of white pixels to consider empty (0.99 = 99%)
        
    Returns:
        True if image is considered empty
    """
    if not PIL_AVAILABLE:
        return False
    
    try:
        img_bytes = decode_base64_image(data_url)
        if not img_bytes:
            return True
        
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to grayscale
        if img.mode == 'RGBA':
            # Check alpha channel first
            alpha = img.split()[3]
            # If mostly transparent, it's empty
            alpha_pixels = list(alpha.getdata())
            transparent_count = sum(1 for p in alpha_pixels if p < 128)
            if transparent_count / len(alpha_pixels) > 0.95:
                return True
        
        gray = img.convert('L')
        pixels = list(gray.getdata())
        
        # Count pixels that are near-white (> 240)
        white_count = sum(1 for p in pixels if p > 240)
        white_ratio = white_count / len(pixels)
        
        return white_ratio > threshold
        
    except Exception as e:
        logger.error(f"Failed to check if canvas is empty: {e}")
        return False


