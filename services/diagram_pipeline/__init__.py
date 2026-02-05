"""
Diagram Pipeline Service

A simple, robust pipeline for generating exam-quality diagrams:
- LLM1 (Kimi 2.5): Writes clean, precise text instructions for diagrams
- TikZ renderer: Generates images from Kimi-authored LaTeX/TikZ
- LLM2 (Kimi 2.5): Reviews images and refines instructions if needed

Constraints:
- Only Kimi 2.5 (temp=0.6) for LLM tasks
- No Gemini image generation dependency for diagrams
- Only OpenAI text-embedding-3-small for embeddings
- No schemdraw, RDKit, matplotlib, or other drawing libraries
"""

from .service import DiagramPipelineService
from .models import (
    DiagramGenerationRequest,
    DiagramGenerationResponse,
    DiagramReview,
    DiagramInstructions,
)

__all__ = [
    "DiagramPipelineService",
    "DiagramGenerationRequest",
    "DiagramGenerationResponse",
    "DiagramReview",
    "DiagramInstructions",
]
