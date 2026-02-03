"""
Diagram Generation Engine for Stoody

A modular diagram generation system that creates accurate, subject-specific
diagrams for educational content.

Supported subjects:
- Mathematics: Graphs, geometry, Venn diagrams
- Physics: Circuits, ray diagrams, FBDs
- Chemistry: Molecular structures, lab apparatus
- Biology: Anatomical diagrams with labels

Usage:
    from services.diagram_engine import DiagramEngine
    
    engine = DiagramEngine()
    result = await engine.generate(diagram_spec)
"""

from .engine import DiagramEngine, get_diagram_engine
from .base_renderer import BaseRenderer
from .output_handler import OutputHandler
from .spec_validator import SpecValidator

# New tool-based diagram generation with verification
from .diagram_toolkit import DiagramBuilder, get_diagram_tools
from .tool_based_generator import ToolBasedDiagramGenerator
from .diagram_verifier import DiagramVerifier, VerificationResult, VerificationStatus
from .verified_diagram_service import VerifiedDiagramService, get_verified_diagram_service

__all__ = [
    # Legacy diagram engine
    "DiagramEngine",
    "get_diagram_engine",
    "BaseRenderer", 
    "OutputHandler",
    "SpecValidator",
    # New tool-based generation
    "DiagramBuilder",
    "get_diagram_tools",
    "ToolBasedDiagramGenerator",
    "DiagramVerifier",
    "VerificationResult",
    "VerificationStatus",
    "VerifiedDiagramService",
    "get_verified_diagram_service",
]

__version__ = "1.0.0"
