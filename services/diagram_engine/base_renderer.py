"""
Base Renderer Abstract Class

All subject-specific renderers inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import io
from datetime import datetime

from .specs.base_spec import (
    BaseDiagramSpec,
    DiagramSubject,
    OutputFormat,
)

logger = logging.getLogger(__name__)


class RenderResult:
    """Result of a render operation"""
    
    def __init__(
        self,
        image_data: bytes,
        format: OutputFormat,
        width: int,
        height: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.image_data = image_data
        self.format = format
        self.width = width
        self.height = height
        self.metadata = metadata or {}
        self.generated_at = datetime.utcnow()
    
    @property
    def file_size(self) -> int:
        return len(self.image_data)
    
    @property
    def content_type(self) -> str:
        content_types = {
            OutputFormat.PNG: "image/png",
            OutputFormat.SVG: "image/svg+xml",
            OutputFormat.PDF: "application/pdf",
        }
        return content_types.get(self.format, "application/octet-stream")
    
    @property
    def file_extension(self) -> str:
        return f".{self.format.value}"


class BaseRenderer(ABC):
    """
    Abstract base class for all diagram renderers.
    
    Each subject (maths, physics, chemistry, biology) has its own
    renderer that inherits from this class.
    
    Subclasses must implement:
        - subject: The subject this renderer handles
        - get_supported_types(): List of diagram types supported
        - render(): The main rendering method
    """
    
    @property
    @abstractmethod
    def subject(self) -> DiagramSubject:
        """The subject this renderer handles"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """
        Get list of diagram types this renderer supports.
        
        Returns:
            List of diagram type names (e.g., ['coordinate_graph', 'geometry'])
        """
        pass
    
    @abstractmethod
    async def render(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Render a diagram based on the specification.
        
        Args:
            spec: The diagram specification dictionary
            
        Returns:
            RenderResult containing the image data and metadata
            
        Raises:
            ValueError: If the spec is invalid
            RenderError: If rendering fails
        """
        pass
    
    def supports_type(self, diagram_type: str) -> bool:
        """
        Check if this renderer supports a specific diagram type.
        
        Args:
            diagram_type: The diagram type to check
            
        Returns:
            True if supported, False otherwise
        """
        return diagram_type in self.get_supported_types()
    
    def validate_spec(self, spec: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a diagram specification.
        
        Args:
            spec: The diagram specification to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if 'diagram_type' not in spec:
            return False, "Missing required field: diagram_type"
        
        diagram_type = spec.get('diagram_type')
        if not self.supports_type(diagram_type):
            supported = ', '.join(self.get_supported_types())
            return False, f"Unsupported diagram type '{diagram_type}'. Supported: {supported}"
        
        return True, None
    
    def get_default_dimensions(self, diagram_type: str) -> Tuple[int, int]:
        """
        Get default dimensions for a diagram type.
        
        Can be overridden by subclasses for type-specific defaults.
        
        Args:
            diagram_type: The diagram type
            
        Returns:
            Tuple of (width, height)
        """
        return (800, 600)
    
    def _apply_style(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default style values to spec.
        
        Args:
            spec: The diagram specification
            
        Returns:
            Spec with style defaults applied
        """
        default_style = {
            'background_color': '#ffffff',
            'line_color': '#000000',
            'font_family': 'Arial',
            'font_size': 12,
            'line_width': 1.5,
        }
        
        style = spec.get('style', {})
        merged_style = {**default_style, **style}
        
        result = spec.copy()
        result['style'] = merged_style
        return result
    
    def _get_dpi(self, spec: Dict[str, Any]) -> int:
        """Get DPI based on quality setting"""
        quality = spec.get('quality', 'high')
        dpi_map = {
            'low': 72,
            'medium': 150,
            'high': 300,
        }
        return dpi_map.get(quality, 300)


class RenderError(Exception):
    """Exception raised when diagram rendering fails"""
    
    def __init__(
        self,
        message: str,
        diagram_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.diagram_type = diagram_type
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error': self.message,
            'diagram_type': self.diagram_type,
            'details': self.details,
        }


class PlaceholderRenderer(BaseRenderer):
    """
    Placeholder renderer that generates simple placeholder images.
    
    Used when actual renderers are not yet implemented, allowing the
    system to function end-to-end while renderers are being developed.
    """
    
    def __init__(self, for_subject: DiagramSubject):
        self._subject = for_subject
    
    @property
    def subject(self) -> DiagramSubject:
        return self._subject
    
    def get_supported_types(self) -> List[str]:
        """Return all types as supported (placeholder accepts anything)"""
        from .specs.base_spec import SUPPORTED_DIAGRAM_TYPES
        return SUPPORTED_DIAGRAM_TYPES.get(self._subject, [])
    
    async def render(self, spec: Dict[str, Any]) -> RenderResult:
        """
        Generate a placeholder image with diagram type info.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            raise RenderError(
                "Pillow is required for placeholder rendering",
                spec.get('diagram_type', 'unknown'),
                {'missing_package': 'Pillow'}
            )
        
        # Get dimensions
        dimensions = spec.get('dimensions', {})
        width = dimensions.get('width', 800)
        height = dimensions.get('height', 600)
        
        # Get style
        style = spec.get('style', {})
        bg_color = style.get('background_color', '#ffffff')
        line_color = style.get('line_color', '#000000')
        
        # Create image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw border
        draw.rectangle(
            [(10, 10), (width - 10, height - 10)],
            outline=line_color,
            width=2
        )
        
        # Draw diagonal lines (placeholder pattern)
        draw.line([(10, 10), (width - 10, height - 10)], fill='#cccccc', width=1)
        draw.line([(width - 10, 10), (10, height - 10)], fill='#cccccc', width=1)
        
        # Add text
        diagram_type = spec.get('diagram_type', 'unknown')
        subject = spec.get('subject', self._subject.value)
        
        text_lines = [
            f"[PLACEHOLDER]",
            f"Subject: {subject}",
            f"Type: {diagram_type}",
            f"Size: {width}x{height}",
        ]
        
        # Use default font (no external font file needed)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Calculate text position (center)
        y_offset = height // 2 - (len(text_lines) * 25) // 2
        
        for i, line in enumerate(text_lines):
            # Get text bounding box
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            y = y_offset + i * 30
            
            # Draw text with background
            draw.rectangle(
                [(x - 5, y - 2), (x + text_width + 5, y + 25)],
                fill='#f0f0f0'
            )
            draw.text((x, y), line, fill=line_color, font=font)
        
        # Convert to bytes
        output_format = OutputFormat(spec.get('output_format', 'png'))
        buffer = io.BytesIO()
        
        if output_format == OutputFormat.PNG:
            img.save(buffer, format='PNG')
        elif output_format == OutputFormat.SVG:
            # For SVG, we still output PNG (placeholder limitation)
            img.save(buffer, format='PNG')
            output_format = OutputFormat.PNG
        elif output_format == OutputFormat.PDF:
            img.save(buffer, format='PDF')
        
        buffer.seek(0)
        image_data = buffer.read()
        
        logger.info(
            f"Generated placeholder for {subject}/{diagram_type} "
            f"({width}x{height}, {len(image_data)} bytes)"
        )
        
        return RenderResult(
            image_data=image_data,
            format=output_format,
            width=width,
            height=height,
            metadata={
                'placeholder': True,
                'diagram_type': diagram_type,
                'subject': subject,
            }
        )
