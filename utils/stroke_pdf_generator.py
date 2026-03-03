"""
Stroke to PDF Generator for Stoody

Converts pen strokes (stored in MongoDB) to SVG and then to PDF.
Used for the "Pin Copy" feature in the Learning Mode.

Stroke Format (from BLE Agent):
- strokes: List of stroke objects
- Each stroke has:
  - points: List of {x, y, pressure} or [[x, y, pressure], ...]
  - svgPath: Pre-rendered SVG path string (V2 format from agent)
  - color: Stroke color (hex or named)
  - strokeWidth: Width of the stroke

Canvas Dimensions (from StoodyPenCanvas):
- A5: 592 x 840
- A4: 840 x 1188
- Default (pen coordinate space): 1480 x 2100
"""

import io
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Canonical paper dimensions in millimeters (must match BLE agent).
BOOK_DIMENSIONS_MM = {
    # Standard sizes
    'A6': (105, 148),
    'A5': (148, 210),
    'A4': (210, 297),
    'A3': (297, 420),

    # Book type codes from BLE pen
    'SS': (105, 148),
    'SN': (105, 148),
    'SM': (105, 148),
    'SL': (210, 148),  # A5 landscape
    'SW': (210, 148),
    'MS': (148, 210),  # A5 portrait
    'MN': (148, 210),
    'MM': (148, 210),
    'ML': (297, 210),  # A4 landscape
    'MW': (297, 210),
    'LS': (210, 297),  # A4 portrait
    'LN': (210, 297),
    'LM': (210, 297),
    'LL': (420, 297),  # A3 landscape
    'LW': (420, 297),
}

PIXELS_PER_MM = 4

BOOK_DIMENSIONS = {
    key: (round(width_mm * PIXELS_PER_MM), round(height_mm * PIXELS_PER_MM))
    for key, (width_mm, height_mm) in BOOK_DIMENSIONS_MM.items()
}
BOOK_DIMENSIONS['default'] = BOOK_DIMENSIONS['MS']

# Default stroke styling
DEFAULT_STROKE_COLOR = "#1a1a1a"
DEFAULT_STROKE_WIDTH = 2.0


def _is_closed_outline_path(path_data: str) -> bool:
    """
    Detect whether an SVG path is a closed outline (intended for fill rendering).

    Agent-generated perfect-freehand paths are closed and end with `Z`.
    """
    if not path_data:
        return False
    normalized = path_data.strip().lower()
    return normalized.endswith("z")


def get_canvas_dimensions(book_type: Optional[str]) -> Tuple[int, int]:
    """Get canvas dimensions for a book type."""
    if book_type and book_type.upper() in BOOK_DIMENSIONS:
        return BOOK_DIMENSIONS[book_type.upper()]
    return BOOK_DIMENSIONS['default']


def build_svg_path_from_points(points: List[Any]) -> str:
    """
    Build an SVG path from points array.
    
    Points can be in two formats:
    1. List of dicts: [{x: 10, y: 20, pressure: 0.5}, ...]
    2. List of arrays: [[10, 20, 0.5], ...]
    """
    if not points or len(points) < 2:
        return ""
    
    path_parts = []
    
    for i, point in enumerate(points):
        # Handle both dict and array format
        if isinstance(point, dict):
            x, y = point.get('x', 0), point.get('y', 0)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        
        if i == 0:
            path_parts.append(f"M {x:.2f} {y:.2f}")
        else:
            path_parts.append(f"L {x:.2f} {y:.2f}")
    
    return " ".join(path_parts) if path_parts else ""


def calculate_stroke_bounds(stroke_batches: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """
    Calculate the bounding box of all strokes.
    Returns (min_x, min_y, max_x, max_y).
    """
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for batch in stroke_batches:
        strokes = batch.get('strokes', [])
        
        for stroke in strokes:
            points = stroke.get('points', [])
            
            for point in points:
                if isinstance(point, dict):
                    x, y = point.get('x', 0), point.get('y', 0)
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    x, y = point[0], point[1]
                else:
                    continue
                
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            
            # Also parse svgPath if present
            svg_path = stroke.get('svgPath', '')
            if svg_path:
                # Extract coordinates from path
                import re
                coords = re.findall(r'([ML])\s*([\d.]+)\s+([\d.]+)', svg_path)
                for _, x_str, y_str in coords:
                    x, y = float(x_str), float(y_str)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
    
    # If no valid bounds found, return defaults
    if min_x == float('inf'):
        return (0, 0, 592, 840)
    
    return (min_x, min_y, max_x, max_y)


def build_svg_from_strokes(
    stroke_batches: List[Dict[str, Any]],
    book_type: Optional[str] = None,
    background_color: str = "#FFFBF0",
    include_background: bool = True,
    fit_to_content: bool = True
) -> str:
    """
    Build an SVG document from stroke batches.
    
    Args:
        stroke_batches: List of stroke batch documents from MongoDB
        book_type: Book type for canvas dimensions
        background_color: Background color for the SVG
        include_background: Whether to include background rectangle
        fit_to_content: If True, calculates viewBox to show all strokes
    
    Returns:
        SVG document as string
    """
    width, height = get_canvas_dimensions(book_type)
    
    # Calculate the bounding box of all strokes
    min_x, min_y, max_x, max_y = calculate_stroke_bounds(stroke_batches)
    
    # Calculate content dimensions
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    # Add padding (10% on each side)
    padding_x = content_width * 0.1 if content_width > 0 else 10
    padding_y = content_height * 0.1 if content_height > 0 else 10
    
    # Calculate viewBox that contains all strokes with padding
    view_min_x = min_x - padding_x
    view_min_y = min_y - padding_y
    view_width = content_width + (padding_x * 2)
    view_height = content_height + (padding_y * 2)
    
    # Ensure minimum viewBox size and maintain aspect ratio
    if view_width < 100:
        view_width = 592
        view_min_x = 0
    if view_height < 100:
        view_height = 840
        view_min_y = 0
    
    # Start SVG document with viewBox that fits content
    svg_parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{view_min_x:.2f} {view_min_y:.2f} {view_width:.2f} {view_height:.2f}" preserveAspectRatio="xMidYMid meet">',
    ]
    
    # Add background if requested
    if include_background:
        svg_parts.append(f'  <rect x="{view_min_x:.2f}" y="{view_min_y:.2f}" width="{view_width:.2f}" height="{view_height:.2f}" fill="{background_color}"/>')
    
    # Add styles
    svg_parts.append('  <style>')
    svg_parts.append('    path { stroke-linecap: round; stroke-linejoin: round; }')
    svg_parts.append('  </style>')
    
    # Process each stroke batch
    for batch in stroke_batches:
        strokes = batch.get('strokes', [])
        
        for stroke in strokes:
            # Get stroke styling
            color = stroke.get('color', DEFAULT_STROKE_COLOR)
            stroke_width = stroke.get('strokeWidth', DEFAULT_STROKE_WIDTH)
            
            # Scale stroke width proportionally
            # For large coordinate spaces, we need thicker strokes
            scale_factor = max(view_width, view_height) / 800
            adjusted_width = stroke_width * max(1, scale_factor)
            
            # Try to get pre-rendered SVG path first (V2 format)
            svg_path = stroke.get('svgPath', '')

            if svg_path:
                # Agent V2 paths are closed outlines from perfect-freehand.
                # They must be filled (not stroked) to avoid zig-zag artifacts.
                if _is_closed_outline_path(svg_path):
                    svg_parts.append(
                        f'  <path d="{svg_path}" fill="{color}" stroke="none"/>'
                    )
                else:
                    # Legacy/non-outline path: render as centerline stroke.
                    svg_parts.append(
                        f'  <path d="{svg_path}" fill="none" stroke="{color}" stroke-width="{adjusted_width:.1f}"/>'
                    )
                continue

            # Fall back to building centerline path from points
            points = stroke.get('points', [])
            svg_path = build_svg_path_from_points(points)
            if svg_path:
                svg_parts.append(
                    f'  <path d="{svg_path}" fill="none" stroke="{color}" stroke-width="{adjusted_width:.1f}"/>'
                )
    
    # Close SVG document
    svg_parts.append('</svg>')
    
    return "\n".join(svg_parts)


def svg_to_pdf_bytes(svg_content: str) -> Optional[bytes]:
    """
    Convert SVG content to PDF bytes.
    
    Uses reportlab with svglib for conversion.
    Falls back to cairosvg if available.
    """
    try:
        # Try svglib + reportlab first (most common in Python)
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        
        # Parse SVG
        drawing = svg2rlg(io.StringIO(svg_content))
        
        if drawing is None:
            logger.error("Failed to parse SVG with svglib")
            return None
        
        # Render to PDF
        pdf_buffer = io.BytesIO()
        renderPDF.drawToFile(drawing, pdf_buffer)
        pdf_buffer.seek(0)
        
        return pdf_buffer.read()
        
    except ImportError:
        logger.warning("svglib/reportlab not available, trying cairosvg...")
        
        try:
            import cairosvg
            return cairosvg.svg2pdf(bytestring=svg_content.encode('utf-8'))
        except ImportError:
            logger.error("Neither svglib nor cairosvg is installed. Please install: pip install svglib reportlab")
            return None
        except Exception as e:
            logger.error(f"cairosvg conversion failed: {e}")
            return None
    except Exception as e:
        logger.error(f"SVG to PDF conversion failed: {e}")
        return None


def svg_to_png_bytes(svg_content: str, scale: float = 1.0) -> Optional[bytes]:
    """
    Convert SVG content to PNG bytes for thumbnails.
    """
    try:
        import cairosvg
        return cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            scale=scale
        )
    except ImportError:
        logger.warning("cairosvg not available for PNG generation")
        return None
    except Exception as e:
        logger.error(f"SVG to PNG conversion failed: {e}")
        return None


async def generate_copy_pdf(
    stroke_batches: List[Dict[str, Any]],
    book_type: Optional[str] = None,
    background_color: str = "#FFFBF0"
) -> Optional[bytes]:
    """
    Generate a PDF from stroke batches.
    
    Args:
        stroke_batches: List of stroke batch documents from MongoDB
        book_type: Book type for canvas dimensions
        background_color: Background color
    
    Returns:
        PDF as bytes, or None if generation failed
    """
    if not stroke_batches:
        logger.warning("No stroke batches provided for PDF generation")
        return None
    
    # Build SVG
    svg_content = build_svg_from_strokes(
        stroke_batches,
        book_type=book_type,
        background_color=background_color
    )
    
    # Convert to PDF
    pdf_bytes = svg_to_pdf_bytes(svg_content)
    
    if pdf_bytes:
        logger.info(f"Generated PDF: {len(pdf_bytes)} bytes")
    
    return pdf_bytes


async def generate_copy_thumbnail(
    stroke_batches: List[Dict[str, Any]],
    book_type: Optional[str] = None,
    background_color: str = "#FFFBF0",
    scale: float = 0.2
) -> Optional[bytes]:
    """
    Generate a PNG thumbnail from stroke batches.
    
    Args:
        stroke_batches: List of stroke batch documents
        book_type: Book type for canvas dimensions
        background_color: Background color
        scale: Scale factor for the thumbnail (0.2 = 20% of original size)
    
    Returns:
        PNG as bytes, or None if generation failed
    """
    if not stroke_batches:
        return None
    
    # Build SVG
    svg_content = build_svg_from_strokes(
        stroke_batches,
        book_type=book_type,
        background_color=background_color
    )
    
    # Convert to PNG
    return svg_to_png_bytes(svg_content, scale=scale)


# Test helper
if __name__ == "__main__":
    # Test with sample strokes
    test_strokes = [
        {
            "strokes": [
                {
                    "points": [{"x": 100, "y": 100}, {"x": 150, "y": 120}, {"x": 200, "y": 100}],
                    "color": "#000000",
                    "strokeWidth": 2
                },
                {
                    "svgPath": "M 50 50 L 100 100 L 150 50",
                    "color": "#FF0000",
                    "strokeWidth": 3
                }
            ]
        }
    ]
    
    svg = build_svg_from_strokes(test_strokes, book_type="A5")
    print("Generated SVG:")
    print(svg)
