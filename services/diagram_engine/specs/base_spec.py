"""
Base Diagram Specification Models

Core Pydantic models that all diagram types inherit from.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import hashlib
import json


class DiagramSubject(str, Enum):
    """Supported diagram subjects"""
    MATHS = "maths"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"


class OutputFormat(str, Enum):
    """Supported output formats"""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"


class DiagramQuality(str, Enum):
    """Output quality levels"""
    LOW = "low"      # 72 DPI, smaller file size
    MEDIUM = "medium"  # 150 DPI, balanced
    HIGH = "high"    # 300 DPI, print quality


class DiagramDimensions(BaseModel):
    """Diagram dimensions specification"""
    width: int = Field(default=800, ge=100, le=4000, description="Width in pixels")
    height: int = Field(default=600, ge=100, le=4000, description="Height in pixels")
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v < 100:
            raise ValueError("Dimension must be at least 100 pixels")
        if v > 4000:
            raise ValueError("Dimension must not exceed 4000 pixels")
        return v


class DiagramStyle(BaseModel):
    """Diagram styling options"""
    background_color: str = Field(
        default="#ffffff",
        pattern=r"^#[0-9a-fA-F]{6}$",
        description="Background color in hex format"
    )
    line_color: str = Field(
        default="#000000",
        pattern=r"^#[0-9a-fA-F]{6}$",
        description="Default line color in hex format"
    )
    font_family: str = Field(
        default="Arial",
        description="Font family for labels"
    )
    font_size: int = Field(
        default=12,
        ge=8,
        le=48,
        description="Font size in points"
    )
    line_width: float = Field(
        default=1.5,
        ge=0.5,
        le=10.0,
        description="Default line width in points"
    )


class BaseDiagramSpec(BaseModel):
    """
    Base specification for all diagram types.
    
    All subject-specific specs inherit from this base class.
    """
    subject: DiagramSubject = Field(
        ...,
        description="Subject area (maths, physics, chemistry, biology)"
    )
    diagram_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Specific diagram type within the subject"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG,
        description="Output format (png, svg, pdf)"
    )
    quality: DiagramQuality = Field(
        default=DiagramQuality.HIGH,
        description="Output quality level"
    )
    dimensions: DiagramDimensions = Field(
        default_factory=DiagramDimensions,
        description="Diagram dimensions"
    )
    style: DiagramStyle = Field(
        default_factory=DiagramStyle,
        description="Styling options"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for tracking"
    )
    
    def get_spec_hash(self) -> str:
        """
        Generate a unique hash for this spec.
        Used for caching and deduplication.
        """
        # Create a deterministic JSON string
        spec_dict = self.model_dump(exclude={'metadata'})
        spec_json = json.dumps(spec_dict, sort_keys=True, default=str)
        return hashlib.sha256(spec_json.encode()).hexdigest()[:16]
    
    def get_dpi(self) -> int:
        """Get DPI based on quality setting"""
        dpi_map = {
            DiagramQuality.LOW: 72,
            DiagramQuality.MEDIUM: 150,
            DiagramQuality.HIGH: 300,
        }
        return dpi_map[self.quality]

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "maths",
                "diagram_type": "coordinate_graph",
                "output_format": "png",
                "quality": "high",
                "dimensions": {"width": 800, "height": 600},
                "style": {
                    "background_color": "#ffffff",
                    "line_color": "#000000",
                    "font_family": "Arial",
                    "font_size": 12
                }
            }
        }
    }


class DiagramResult(BaseModel):
    """Result of a successful diagram generation"""
    diagram_id: str = Field(..., description="Unique diagram identifier")
    url: str = Field(..., description="URL to access the diagram")
    storage_path: str = Field(..., description="Internal storage path (S3 or local)")
    format: OutputFormat = Field(..., description="Output format")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    file_size_bytes: int = Field(..., description="File size in bytes")
    spec_hash: str = Field(..., description="Hash of the input spec")
    cached: bool = Field(default=False, description="Whether result was from cache")
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Generation timestamp"
    )
    generation_time_ms: Optional[int] = Field(
        default=None,
        description="Time taken to generate in milliseconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "diagram_id": "diag_abc123",
                "url": "https://cdn.stoody.in/diagrams/diag_abc123.png",
                "storage_path": "s3://stoody-diagrams/diagrams/diag_abc123.png",
                "format": "png",
                "width": 800,
                "height": 600,
                "file_size_bytes": 45678,
                "spec_hash": "a1b2c3d4e5f6g7h8",
                "cached": False,
                "generated_at": "2025-01-31T10:30:00Z",
                "generation_time_ms": 1250
            }
        }
    }


class DiagramError(BaseModel):
    """Error response for diagram generation failures"""
    error_code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    spec_hash: Optional[str] = Field(
        default=None,
        description="Hash of the input spec that caused the error"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error_code": "INVALID_SPEC",
                "message": "Invalid diagram specification",
                "details": {"field": "functions", "error": "Expression syntax error"},
                "spec_hash": "a1b2c3d4e5f6g7h8",
                "timestamp": "2025-01-31T10:30:00Z"
            }
        }
    }


# ============================================================================
# Diagram Type Registry
# ============================================================================

# Mapping of subject to supported diagram types
SUPPORTED_DIAGRAM_TYPES: Dict[DiagramSubject, List[str]] = {
    DiagramSubject.MATHS: [
        "coordinate_graph",
        "geometry_construction", 
        "number_line",
        "venn_diagram",
        "bar_chart",
        "pie_chart",
        "trigonometric_circle",
        "3d_plot",
    ],
    DiagramSubject.PHYSICS: [
        "series_circuit",
        "parallel_circuit",
        "mixed_circuit",
        "ray_diagram_lens",
        "ray_diagram_mirror",
        "free_body_diagram",
        "inclined_plane",
        "projectile_motion",
        "wave_diagram",
        "electric_field",
        "magnetic_field",
    ],
    DiagramSubject.CHEMISTRY: [
        "molecule_2d",
        "molecule_3d",
        "reaction_scheme",
        "lab_setup_titration",
        "lab_setup_distillation",
        "lab_setup_electrolysis",
        "lab_setup_manometer",
        "periodic_table_section",
        "orbital_diagram",
        "crystal_structure",
    ],
    DiagramSubject.BIOLOGY: [
        "human_heart",
        "human_brain",
        "nephron",
        "neuron",
        "plant_cell",
        "animal_cell",
        "dna_replication",
        "mitosis_stages",
        "meiosis_stages",
        "digestive_system",
        "respiratory_system",
        "eye_structure",
        "ear_structure",
        "flower_structure",
    ],
}


def get_supported_types(subject: DiagramSubject) -> List[str]:
    """Get list of supported diagram types for a subject"""
    return SUPPORTED_DIAGRAM_TYPES.get(subject, [])


def is_valid_diagram_type(subject: DiagramSubject, diagram_type: str) -> bool:
    """Check if a diagram type is valid for a subject"""
    return diagram_type in SUPPORTED_DIAGRAM_TYPES.get(subject, [])
