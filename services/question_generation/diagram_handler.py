"""
Diagram Handler Service

Handles diagram generation, verification, and storage for question generation.
"""

import json
import logging
import base64
import os
import re
import aiohttp
from typing import Any, Dict, List, Optional, Tuple

from .models.question import GeneratedQuestion, DiagramSpec
from .constants import VALID_DIAGRAM_TYPES

logger = logging.getLogger(__name__)


# ============================================================================
# TYPE ALIASES FOR DIAGRAM VALIDATION
# ============================================================================

TYPE_ALIASES = {
    # Maths aliases
    "3d_coordinate_graph": "coordinate_graph",
    "3d_graph": "3d_plot",
    "graph": "coordinate_graph",
    "line_graph": "coordinate_graph",
    "function_graph": "coordinate_graph",
    "cartesian_graph": "coordinate_graph",
    "cartesian_plane": "coordinate_graph",
    "xy_graph": "coordinate_graph",
    "numberline": "number_line",
    "number-line": "number_line",
    "inequality_number_line": "number_line",
    "geometry": "geometry_construction",
    "geometric_construction": "geometry_construction",
    "triangle": "geometry_construction",
    "circle": "geometry_construction",
    "angle": "geometry_construction",
    "polygon": "geometry_construction",
    "venn": "venn_diagram",
    "set_diagram": "venn_diagram",
    "bar_graph": "bar_chart",
    "barchart": "bar_chart",
    "histogram": "bar_chart",
    "pie_graph": "pie_chart",
    "piechart": "pie_chart",
    "trig_circle": "trigonometric_circle",
    "unit_circle": "trigonometric_circle",
    
    # Physics aliases
    "circuit": "series_circuit",
    "electrical_circuit": "series_circuit",
    "circuit_diagram": "series_circuit",
    "lens_diagram": "ray_diagram_lens",
    "mirror_diagram": "ray_diagram_mirror",
    "optics": "ray_diagram_lens",
    "force_diagram": "free_body_diagram",
    "fbd": "free_body_diagram",
    "forces": "free_body_diagram",
    "projectile": "projectile_motion",
    "motion": "projectile_motion",
    "wave": "wave_diagram",
    "waves": "wave_diagram",
    "incline": "inclined_plane",
    "ramp": "inclined_plane",
    
    # Chemistry aliases
    "molecule": "molecule_2d",
    "molecular_structure": "molecule_2d",
    "chemical_structure": "molecule_2d",
    "reaction": "reaction_scheme",
    "chemical_reaction": "reaction_scheme",
    "titration": "lab_setup_titration",
    "distillation": "lab_setup_distillation",
    "electrolysis": "lab_setup_electrolysis",
    "periodic_table": "periodic_table_section",
    "orbital": "orbital_diagram",
    "electron_configuration": "orbital_diagram",
    
    # Biology aliases
    "heart": "human_heart",
    "brain": "human_brain",
    "cell": "animal_cell",
    "plant": "plant_cell",
    "dna": "dna_replication",
    "mitosis": "mitosis_stages",
    "meiosis": "meiosis_stages",
    "digestion": "digestive_system",
    "respiratory": "respiratory_system",
    "eye": "eye_structure",
    "ear": "ear_structure",
    "flower": "flower_structure",
}

DEFAULT_TYPES_BY_SUBJECT = {
    "maths": "coordinate_graph",
    "physics": "free_body_diagram",
    "chemistry": "molecule_2d",
    "biology": "animal_cell",
}


class DiagramHandler:
    """Handles diagram generation, verification, and storage."""
    
    def __init__(self, openai_service=None, verified_diagram_service=None):
        self._openai_service = openai_service
        self._verified_diagram_service = verified_diagram_service
    
    def set_services(self, openai_service, verified_diagram_service):
        """Set the required services after initialization."""
        self._openai_service = openai_service
        self._verified_diagram_service = verified_diagram_service
    
    async def generate_diagram_for_question(
        self,
        question: GeneratedQuestion,
    ) -> GeneratedQuestion:
        """
        Generate diagram for a specific question with verification loop.
        
        Args:
            question: The question to generate diagram for
            
        Returns:
            Question with diagram URL populated
        """
        if not self._verified_diagram_service:
            logger.warning("Verified diagram service not available")
            return question
        
        if not question.has_diagram and not question.diagram_spec:
            logger.debug(f"Question doesn't require a diagram: {question.question_text[:50]}...")
            return question
        
        try:
            # Build context for the diagram
            diagram_description = None
            if question.diagram_spec:
                spec_dict = question.diagram_spec.to_dict()
                # Flatten parameters into the main dict for DiagramPlan compatibility
                if 'parameters' in spec_dict and isinstance(spec_dict['parameters'], dict):
                    params = spec_dict.pop('parameters')
                    spec_dict.update(params)
                    
                diagram_description = spec_dict.get('description', spec_dict.get('title', ''))
            
            subject = question.subject.lower() if question.subject else 'maths'
            
            logger.info(f"   Generating verified diagram:")
            logger.info(f"      Subject: {subject}")
            logger.info(f"      Question: {question.question_text[:80]}...")
            if diagram_description:
                logger.info(f"      Description: {diagram_description}")
            
            # Generate diagram with verification loop
            result = await self._verified_diagram_service.generate_verified_diagram(
                question_text=question.question_text,
                subject=subject,
                diagram_description=diagram_description,
                diagram_spec=spec_dict if question.diagram_spec else None,
            )
            
            if result.image_bytes:
                # Save the image and get URL
                diagram_url = await self.save_diagram_image(result.image_bytes, question.question_id)
                
                if diagram_url:
                    question.diagram_url = diagram_url
                    question.has_diagram = True
                    
                    logger.info(f"   DIAGRAM GENERATED:")
                    logger.info(f"      Attempts: {result.attempts}")
                    if result.verification_result:
                        status_val = result.verification_result.status.value if hasattr(result.verification_result.status, 'value') else result.verification_result.status
                        logger.info(f"      Verification: {status_val}")
                        logger.info(f"      Confidence: {result.verification_result.confidence:.2f}")
                    logger.info(f"      URL: {diagram_url}")
                    
                    if not result.success:
                        logger.warning(f"   Note: Diagram may not be fully accurate (verification failed)")
                else:
                    logger.error("   FAILED - Could not save diagram image")
            else:
                error = result.error or 'Unknown error'
                logger.error(f"   FAILED - Diagram generation error: {error}")
                # Explicitly mark that we don't have a diagram
                question.has_diagram = False
                question.diagram_url = None
                
        except Exception as e:
            logger.error(f"   EXCEPTION generating diagram: {type(e).__name__}: {e}", exc_info=True)
        
        return question

    async def save_diagram_image(self, image_bytes: bytes, question_id: str) -> Optional[str]:
        """
        Save generated diagram image to storage and return URL.
        
        Args:
            image_bytes: The PNG image data
            question_id: The question ID for unique identification
            
        Returns:
            URL to the saved diagram, or None if saving failed
        """
        import hashlib
        
        try:
            # Generate unique filename using question_id
            filename = f"diagram_{question_id[:8]}_{hashlib.md5(image_bytes).hexdigest()[:8]}.png"
            
            # Try to use S3 storage if available
            try:
                from utils.s3_storage import upload_file, get_public_url
                
                local_path = f"uploads/diagrams/{filename}"
                
                success, storage_path = await upload_file(
                    file_data=image_bytes,
                    local_path=local_path,
                    content_type="image/png",
                    metadata={
                        'question_id': question_id,
                        'generated_by': 'tool_based_generator'
                    }
                )
                
                if success:
                    url = get_public_url(storage_path)
                    logger.info(f"Diagram saved to S3: {storage_path}")
                    return url
                else:
                    logger.warning("S3 upload failed, falling back to local")
                    
            except ImportError:
                logger.debug("S3 module not available, using local storage")
            
            # Fallback to local storage
            uploads_dir = os.path.join(os.getcwd(), "uploads", "diagrams")
            os.makedirs(uploads_dir, exist_ok=True)
            
            local_path = os.path.join(uploads_dir, filename)
            
            with open(local_path, 'wb') as f:
                f.write(image_bytes)
            
            logger.info(f"Diagram saved locally: {local_path}")
            
            # Return API endpoint URL for local files
            return f"/api/v1/diagrams/local/{filename}"
            
        except Exception as e:
            logger.error(f"Failed to save diagram: {e}")
            return None

    async def fetch_diagram_as_base64(self, url: str) -> Optional[str]:
        """
        Fetch diagram image and convert to base64.
        
        Args:
            url: Image URL or file path
            
        Returns:
            Base64 encoded image data or None if fetch fails
        """
        try:
            if url.startswith('http://') or url.startswith('https://'):
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                        if response.status == 200:
                            data = await response.read()
                            return base64.b64encode(data).decode('utf-8')
                        else:
                            logger.warning(f"Failed to fetch diagram: HTTP {response.status}")
            else:
                # Local file path
                if url.startswith('/') or os.path.isabs(url):
                    file_path = url
                elif url.startswith('uploads/'):
                    file_path = os.path.join(os.getcwd(), url)
                else:
                    file_path = os.path.join(os.getcwd(), url)
                
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        return base64.b64encode(f.read()).decode('utf-8')
                else:
                    logger.warning(f"Diagram file not found: {file_path}")
                    
        except Exception as e:
            logger.error(f"Failed to fetch diagram as base64: {e}")
        
        return None

    def validate_and_fix_diagram_spec(
        self,
        spec: Dict[str, Any],
        subject: str,
    ) -> Dict[str, Any]:
        """
        Validate and fix common issues with diagram specs from LLM.
        
        Args:
            spec: The diagram spec from LLM
            subject: The subject for context
            
        Returns:
            Fixed diagram spec
        """
        subject_lower = subject.lower()
        valid_types = VALID_DIAGRAM_TYPES.get(subject_lower, [])
        
        diagram_type = spec.get("diagram_type", "")
        
        # Check if type is valid
        if diagram_type not in valid_types:
            # Try to map to a valid type
            mapped_type = TYPE_ALIASES.get(diagram_type.lower())
            if mapped_type and mapped_type in valid_types:
                logger.info(f"Mapped invalid diagram type '{diagram_type}' -> '{mapped_type}'")
                spec["diagram_type"] = mapped_type
            else:
                default_type = DEFAULT_TYPES_BY_SUBJECT.get(subject_lower, "coordinate_graph")
                logger.warning(f"Unknown diagram type '{diagram_type}' for {subject}, defaulting to '{default_type}'")
                spec["diagram_type"] = default_type
        
        # Fix range fields
        spec = self._fix_range_fields(spec)
        
        # Fix points for 2D diagrams
        spec = self._fix_points(spec)
        
        # Fix lines/vectors
        spec = self._fix_lines(spec)
        
        return spec
    
    def _fix_range_fields(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Fix range, x_range, y_range fields in spec."""
        for key in ["range", "x_range", "y_range"]:
            if key in spec:
                val = spec[key]
                if isinstance(val, dict):
                    if 0 in val and 1 in val:
                        spec[key] = [val[0], val[1]]
                    elif "0" in val and "1" in val:
                        spec[key] = [val["0"], val["1"]]
                    elif "min" in val and "max" in val:
                        spec[key] = [val["min"], val["max"]]
                    elif "start" in val and "end" in val:
                        spec[key] = [val["start"], val["end"]]
                    else:
                        spec[key] = [-10, 10]
                elif isinstance(val, (int, float)):
                    spec[key] = [-abs(val), abs(val)]
                elif not isinstance(val, list) or len(val) < 2:
                    spec[key] = [-10, 10]
                else:
                    try:
                        spec[key] = [float(val[0]), float(val[1])]
                    except (TypeError, ValueError):
                        spec[key] = [-10, 10]
        return spec
    
    def _fix_points(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Fix points array in spec - ensure proper coordinates."""
        if "points" not in spec:
            return spec
            
        points = spec["points"]
        diagram_type = spec.get("diagram_type", "")
        is_2d = diagram_type in ["coordinate_graph", "geometry_construction", "3d_plot"]
        
        if not isinstance(points, list):
            return spec
            
        fixed_points = []
        for p in points:
            if isinstance(p, dict):
                label = str(p.get("label", ""))
                
                x_val = p.get("x", p.get("value", None))
                y_val = p.get("y", None)
                
                # Parse coordinates from label if needed
                if is_2d and (x_val is None or y_val is None):
                    coord_match = re.search(r'\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?', label)
                    if coord_match:
                        try:
                            if x_val is None:
                                x_val = float(coord_match.group(1))
                            if y_val is None:
                                y_val = float(coord_match.group(2))
                        except ValueError:
                            pass
                
                if x_val is None:
                    x_val = 0
                if y_val is None and is_2d:
                    y_val = 0
                
                fixed_point = {
                    "x": x_val,
                    "label": label,
                    "color": p.get("color", p.get("style", "#d62728")),
                }
                
                if is_2d:
                    fixed_point["y"] = y_val
                
                for key in ["marker", "size"]:
                    if key in p:
                        fixed_point[key] = p[key]
                
                fixed_points.append(fixed_point)
            elif isinstance(p, (int, float)):
                fixed_point = {"x": p, "label": str(p), "color": "#d62728"}
                if is_2d:
                    fixed_point["y"] = 0
                fixed_points.append(fixed_point)
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                fixed_point = {
                    "x": p[0],
                    "y": p[1] if is_2d else 0,
                    "label": f"({p[0]}, {p[1]})" if is_2d else str(p[0]),
                    "color": "#d62728"
                }
                fixed_points.append(fixed_point)
        
        spec["points"] = fixed_points
        return spec
    
    def _fix_lines(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Fix lines array in spec - ensure proper from/to coordinates."""
        if "lines" not in spec:
            return spec
            
        lines = spec["lines"]
        if not isinstance(lines, list):
            return spec
            
        fixed_lines = []
        for line in lines:
            if isinstance(line, dict):
                from_val = line.get("from", [0, 0])
                to_val = line.get("to", [1, 1])
                
                # Ensure from/to are lists of 2 numbers
                if isinstance(from_val, dict):
                    from_val = [from_val.get("x", 0), from_val.get("y", 0)]
                if isinstance(to_val, dict):
                    to_val = [to_val.get("x", 0), to_val.get("y", 0)]
                
                if not isinstance(from_val, list) or len(from_val) < 2:
                    from_val = [0, 0]
                if not isinstance(to_val, list) or len(to_val) < 2:
                    to_val = [1, 1]
                
                fixed_line = {
                    "from": from_val[:2],
                    "to": to_val[:2],
                }
                
                # Preserve optional properties
                for key in ["label", "color", "style", "arrow"]:
                    if key in line:
                        fixed_line[key] = line[key]
                
                fixed_lines.append(fixed_line)
        
        spec["lines"] = fixed_lines
        return spec


# Global instance
_diagram_handler: Optional[DiagramHandler] = None


def get_diagram_handler() -> DiagramHandler:
    """Get or create the diagram handler singleton."""
    global _diagram_handler
    if _diagram_handler is None:
        _diagram_handler = DiagramHandler()
    return _diagram_handler
