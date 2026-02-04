"""
Diagram Rendering Service for Question Generation.

Renders diagram_spec JSON to actual images using the appropriate tool:
- matplotlib: graphs, geometry, physics diagrams
- schemdraw: circuit diagrams
- rdkit: molecular structures

This service integrates with the existing diagram engine renderers.
"""

import logging
import base64
from typing import Any, Dict, List, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


# =============================================================================
# Renderer Imports (lazy loading)
# =============================================================================

_math_renderer = None
_physics_renderer = None
_chemistry_renderer = None

def get_math_renderer():
    """Get or create the MathRenderer instance."""
    global _math_renderer
    if _math_renderer is None:
        from services.diagram_engine.renderers.math_renderer import MathRenderer
        _math_renderer = MathRenderer()
    return _math_renderer

def get_physics_renderer():
    """Get or create the PhysicsRenderer instance."""
    global _physics_renderer
    if _physics_renderer is None:
        from services.diagram_engine.renderers.physics_renderer import PhysicsRenderer
        _physics_renderer = PhysicsRenderer()
    return _physics_renderer

def get_chemistry_renderer():
    """Get or create the ChemistryRenderer instance."""
    global _chemistry_renderer
    if _chemistry_renderer is None:
        from services.diagram_engine.renderers.chemistry_renderer import ChemistryRenderer
        _chemistry_renderer = ChemistryRenderer()
    return _chemistry_renderer


# =============================================================================
# Tool Selection
# =============================================================================

def select_renderer(subject: str, diagram_type: str, tool_hint: Optional[str] = None):
    """
    Select the appropriate renderer based on subject and diagram type.
    
    Args:
        subject: Subject (physics, chemistry, math, biology)
        diagram_type: Type of diagram
        tool_hint: Optional hint from LLM (matplotlib, schemdraw, rdkit)
        
    Returns:
        Renderer instance
    """
    subject_lower = subject.lower()
    diagram_type_lower = diagram_type.lower() if diagram_type else ""
    
    # Circuit diagrams -> Physics renderer (uses schemdraw)
    # Use word boundaries or exact matches to avoid false positives (e.g., "circle" matching "rc")
    circuit_types = ["circuit", "series_circuit", "parallel_circuit", "mixed_circuit", 
                     "rc_circuit", "rl_circuit", "rlc_circuit", "bridge", "potentiometer"]
    if any(diagram_type_lower == ct or diagram_type_lower.startswith(ct + "_") or 
           diagram_type_lower.endswith("_" + ct) for ct in circuit_types):
        return get_physics_renderer()
    
    # Also check for physics-specific types
    physics_types = ["ray_diagram", "free_body", "inclined_plane", "projectile", 
                     "wave_diagram", "electric_field", "magnetic_field"]
    if any(pt in diagram_type_lower for pt in physics_types):
        return get_physics_renderer()
    
    # Molecular structures -> Chemistry renderer (uses RDKit)
    molecule_types = ["molecule", "structure", "smiles", "isomer", "resonance", "stereoisomer", "orbital"]
    if any(mt in diagram_type_lower for mt in molecule_types):
        return get_chemistry_renderer()
    
    # Subject-based selection
    if "phys" in subject_lower:
        return get_physics_renderer()
    elif "chem" in subject_lower:
        return get_chemistry_renderer()
    elif "math" in subject_lower:
        return get_math_renderer()
    elif "bio" in subject_lower:
        # Biology uses chemistry renderer for some diagrams
        return get_chemistry_renderer()
    
    # Default to math renderer (matplotlib-based)
    return get_math_renderer()


# =============================================================================
# Diagram Rendering
# =============================================================================

async def render_diagram_async(
    diagram_spec: Dict[str, Any],
) -> Tuple[bool, Optional[bytes], Optional[str], Optional[str]]:
    """
    Render a diagram specification to image bytes.
    
    Args:
        diagram_spec: Diagram specification from LLM
        
    Returns:
        Tuple of (success, image_bytes, format, error_message)
    """
    if not diagram_spec:
        return False, None, None, "No diagram spec provided"
    
    try:
        subject = diagram_spec.get("subject", "math")
        diagram_type = diagram_spec.get("diagram_type", "")
        tool_hint = diagram_spec.get("tool")
        
        # Select renderer
        renderer = select_renderer(subject, diagram_type, tool_hint)
        
        logger.info(
            f"Rendering diagram: type={diagram_type}, subject={subject}, "
            f"renderer={renderer.__class__.__name__}"
        )
        
        # Render the diagram (renderer.render is async)
        # RenderResult is returned on success, RenderError is raised on failure
        result = await renderer.render(diagram_spec)
        
        # If we get here, rendering succeeded (RenderResult was returned)
        logger.info(
            f"Diagram rendered successfully: {len(result.image_data)} bytes, "
            f"format={result.format}"
        )
        # Get format as string (it's an enum)
        format_str = result.format.value if hasattr(result.format, 'value') else str(result.format)
        return True, result.image_data, format_str, None
            
    except Exception as e:
        logger.error(f"Diagram rendering error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, None, str(e)


def render_diagram_sync(
    diagram_spec: Dict[str, Any],
) -> Tuple[bool, Optional[bytes], Optional[str], Optional[str]]:
    """
    Synchronous version of render_diagram_async.
    
    Args:
        diagram_spec: Diagram specification from LLM
        
    Returns:
        Tuple of (success, image_bytes, format, error_message)
    """
    if not diagram_spec:
        return False, None, None, "No diagram spec provided"
    
    try:
        subject = diagram_spec.get("subject", "math")
        diagram_type = diagram_spec.get("diagram_type", "")
        tool_hint = diagram_spec.get("tool")
        
        # Select renderer
        renderer = select_renderer(subject, diagram_type, tool_hint)
        
        logger.info(
            f"Rendering diagram (sync): type={diagram_type}, subject={subject}, "
            f"renderer={renderer.__class__.__name__}"
        )
        
        # Render
        result = renderer.render(diagram_spec)
        
        if result.success:
            logger.info(
                f"Diagram rendered successfully: {len(result.image_data)} bytes, "
                f"format={result.format}"
            )
            return True, result.image_data, result.format, None
        else:
            logger.warning(f"Diagram rendering failed: {result.error}")
            return False, None, None, result.error
            
    except Exception as e:
        logger.error(f"Diagram rendering error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None, None, str(e)


def render_to_base64(
    diagram_spec: Dict[str, Any],
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Render diagram and return as base64-encoded data URL.
    
    Args:
        diagram_spec: Diagram specification
        
    Returns:
        Tuple of (success, data_url, error_message)
    """
    success, image_bytes, format_str, error = render_diagram_sync(diagram_spec)
    
    if not success or not image_bytes:
        return False, None, error
    
    # Determine MIME type
    if format_str == "svg":
        mime_type = "image/svg+xml"
    elif format_str == "png":
        mime_type = "image/png"
    elif format_str == "pdf":
        mime_type = "application/pdf"
    else:
        mime_type = f"image/{format_str}"
    
    # Encode to base64
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_data}"
    
    return True, data_url, None


# =============================================================================
# Diagram Validation
# =============================================================================

def validate_rendered_diagram(
    image_bytes: bytes,
    format_str: str,
    min_size_bytes: int = 500,
) -> Tuple[bool, List[str]]:
    """
    Validate a rendered diagram for quality.
    
    Args:
        image_bytes: Rendered diagram bytes
        format_str: Image format (svg, png)
        min_size_bytes: Minimum acceptable size
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    if len(image_bytes) < min_size_bytes:
        warnings.append(f"Image too small ({len(image_bytes)} bytes), may be empty or corrupted")
    
    # SVG-specific checks
    if format_str == "svg":
        try:
            svg_str = image_bytes.decode("utf-8")
            if "<svg" not in svg_str:
                warnings.append("SVG content does not contain <svg> tag")
            if len(svg_str) < 200:
                warnings.append("SVG content suspiciously short")
        except UnicodeDecodeError:
            warnings.append("SVG content is not valid UTF-8")
    
    # PNG-specific checks
    elif format_str == "png":
        # Check PNG magic bytes
        if not image_bytes.startswith(b'\x89PNG'):
            warnings.append("PNG content does not have valid header")
    
    return len(warnings) == 0, warnings


# =============================================================================
# Integration with Question Generation
# =============================================================================

async def render_question_diagram(
    question_draft: "QuestionDraft",
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Render the diagram for a question draft if required.
    
    Args:
        question_draft: QuestionDraft with diagram_spec
        
    Returns:
        Tuple of (success, image_url_or_data_url, error_message)
    """
    from .models.job import QuestionDraft
    from utils.s3_storage import upload_file, get_public_url, is_s3_enabled
    import uuid
    
    if not question_draft.diagram_required:
        return True, None, None
    
    if not question_draft.diagram_spec:
        return False, None, "diagram_required but no diagram_spec provided"
    
    # Render the diagram
    success, image_bytes, format_str, error = await render_diagram_async(
        question_draft.diagram_spec
    )
    
    if not success:
        return False, None, error
    
    # Validate
    is_valid, warnings = validate_rendered_diagram(image_bytes, format_str)
    if not is_valid:
        logger.warning(f"Diagram validation warnings: {warnings}")
    
    # Determine MIME type
    if format_str == "svg":
        mime_type = "image/svg+xml"
        extension = "svg"
    elif format_str == "png":
        mime_type = "image/png"
        extension = "png"
    else:
        mime_type = f"image/{format_str}"
        extension = format_str
    
    # Try to upload to S3 if enabled
    if is_s3_enabled():
        try:
            # Generate unique filename
            diagram_id = str(uuid.uuid4())
            s3_path = f"diagrams/{diagram_id}.{extension}"
            
            # Upload to S3
            upload_success, storage_path = await upload_file(
                file_data=image_bytes,
                local_path=s3_path,
                content_type=mime_type,
            )
            
            if upload_success:
                # Get public URL
                public_url = get_public_url(storage_path)
                logger.info(f"Diagram uploaded to S3: {public_url}")
                return True, public_url, None
            else:
                logger.warning(f"S3 upload failed, falling back to data URL")
        except Exception as e:
            logger.warning(f"S3 upload error: {e}, falling back to data URL")
    
    # Fallback: Convert to data URL
    b64_data = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_data}"
    
    return True, data_url, None


# Type hint for circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models.job import QuestionDraft
