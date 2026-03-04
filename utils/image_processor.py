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
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Apply slight sharpening
        img = img.filter(ImageFilter.SHARPEN)
        
        # Increase brightness slightly to ensure white background
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)
        
        # Convert to grayscale for adaptive stroke analysis
        gray = img.convert('L')

        # Adapt dilation strength to actual handwriting density and scaling.
        # This avoids over-thickening dense answers while still helping faint strokes.
        DARK_PIXEL_THRESHOLD = 200
        VERY_SPARSE_INK_RATIO = 0.012
        SPARSE_INK_RATIO = 0.03

        pixels = list(gray.getdata())
        dark_pixels = sum(1 for p in pixels if p < DARK_PIXEL_THRESHOLD)
        dark_ratio = (dark_pixels / len(pixels)) if pixels else 0.0

        if dark_ratio <= VERY_SPARSE_INK_RATIO:
            # Very faint/thin writing: stronger dilation only when source wasn't heavily upscaled.
            dilation_kernel = 5 if upscale_factor < 1.4 else 3
        elif dark_ratio <= SPARSE_INK_RATIO:
            # Medium-density writing: conservative thickening.
            dilation_kernel = 3 if upscale_factor < 2.2 else 1
        else:
            # Dense writing: avoid dilation to prevent character merging.
            dilation_kernel = 1

        if dilation_kernel > 1:
            inverted = ImageOps.invert(gray)
            dilated = inverted.filter(ImageFilter.MaxFilter(dilation_kernel))
            processed_gray = ImageOps.invert(dilated)
        else:
            processed_gray = gray

        # Convert back to RGB
        img = processed_gray.convert('RGB')
        
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


def merge_canvas_pages_vertical(data_urls: List[str], gap: int = 20, max_height: int = 4000) -> Optional[str]:
    """
    Merge multiple canvas pages into a single tall image.
    Useful for multi-page handwritten answers.
    
    Args:
        data_urls: List of base64 data URLs for each page
        gap: Pixel gap between pages
        max_height: Maximum height of merged image (to prevent memory issues)
        
    Returns:
        Merged image as base64 data URL, or None if failed
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available for image merging")
        return None
    
    if not data_urls:
        return None
    
    try:
        # Decode and open all images
        images = []
        for url in data_urls:
            img_bytes = decode_base64_image(url)
            if img_bytes:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    # Handle transparency
                    if img.mode == 'RGBA':
                        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
                        img = Image.alpha_composite(background, img)
                    img = img.convert('RGB')
                images.append(img)
        
        if not images:
            return None
        
        # Calculate dimensions
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images) + gap * (len(images) - 1)
        
        # Check if exceeds max height
        if total_height > max_height:
            # Scale down all images proportionally
            scale = max_height / total_height
            images = [img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS) 
                      for img in images]
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images) + gap * (len(images) - 1)
        
        # Create merged image
        merged = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in images:
            # Center horizontally
            x_offset = (max_width - img.width) // 2
            merged.paste(img, (x_offset, y_offset))
            y_offset += img.height + gap
        
        # Save to bytes
        output_buffer = io.BytesIO()
        merged.save(output_buffer, format='PNG', quality=95)
        output_bytes = output_buffer.getvalue()
        
        return encode_image_to_base64(output_bytes, "image/png")
        
    except Exception as e:
        logger.error(f"Image merging failed: {e}", exc_info=True)
        return None


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


def prepare_image_for_llm_evaluation(
    canvas_pages: List[str],
    enhance: bool = True,
    merge_pages: bool = True,
    target_width: int = 1500
) -> Tuple[List[str], Optional[str]]:
    """
    Prepare canvas images for LLM evaluation with full processing pipeline.
    
    Args:
        canvas_pages: List of canvas page data URLs
        enhance: Whether to enhance individual pages
        merge_pages: Whether to also create a merged view
        target_width: Target width for enhancement
        
    Returns:
        Tuple of (enhanced individual pages, merged image or None)
    """
    if not canvas_pages:
        return [], None
    
    # Filter out empty canvases
    non_empty_pages = [page for page in canvas_pages if page and not is_canvas_empty(page)]
    
    if not non_empty_pages:
        logger.warning("All canvas pages appear to be empty")
        return [], None
    
    # Enhance individual pages
    enhanced_pages = non_empty_pages
    if enhance:
        enhanced_pages = enhance_canvas_images_batch(non_empty_pages, target_width)
    
    # Create merged view for multi-page context
    merged_image = None
    if merge_pages and len(enhanced_pages) > 1:
        merged_image = merge_canvas_pages_vertical(enhanced_pages)
    elif len(enhanced_pages) == 1:
        merged_image = enhanced_pages[0]
    
    return enhanced_pages, merged_image
