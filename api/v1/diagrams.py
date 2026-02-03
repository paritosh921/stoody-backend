"""
Diagram Generation API Router

Provides endpoints for generating educational diagrams using
the diagram engine.

Endpoints:
- POST /generate: Generate a single diagram
- POST /generate-batch: Generate multiple diagrams
- GET /templates: List available diagram types
- GET /templates/{subject}: List types for a subject
- GET /templates/{subject}/{type}/schema: Get JSON schema
- GET /{diagram_id}: Get diagram metadata
- GET /{diagram_id}/download: Download diagram file
- DELETE /{diagram_id}: Delete a diagram
- GET /stats: Get engine statistics
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io

from services.diagram_engine import DiagramEngine
from services.diagram_engine.engine import get_diagram_engine
from services.diagram_engine.specs.base_spec import (
    DiagramSubject,
    OutputFormat,
    DiagramResult,
    DiagramError,
    SUPPORTED_DIAGRAM_TYPES,
)
from services.diagram_engine.base_renderer import RenderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/diagrams", tags=["diagrams"])


# ============================================================================
# Request/Response Models
# ============================================================================

class DiagramGenerateRequest(BaseModel):
    """Request body for diagram generation"""
    spec: Dict[str, Any] = Field(
        ...,
        description="The diagram specification JSON"
    )
    output_format: Optional[str] = Field(
        default="png",
        description="Output format: png, svg, pdf"
    )
    quality: Optional[str] = Field(
        default="high",
        description="Output quality: low, medium, high"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "spec": {
                    "subject": "physics",
                    "diagram_type": "series_circuit",
                    "components": [
                        {"type": "battery", "label": "12V"},
                        {"type": "resistor", "label": "4Ω"},
                        {"type": "resistor", "label": "6Ω"}
                    ]
                },
                "output_format": "png",
                "quality": "high"
            }
        }
    }


class DiagramGenerateResponse(BaseModel):
    """Response from diagram generation"""
    success: bool
    diagram_id: Optional[str] = None
    url: Optional[str] = None
    format: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size_bytes: Optional[int] = None
    cached: bool = False
    generation_time_ms: Optional[int] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class DiagramBatchRequest(BaseModel):
    """Request body for batch diagram generation"""
    diagrams: List[DiagramGenerateRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of diagram specifications (max 50)"
    )
    stop_on_error: bool = Field(
        default=False,
        description="Stop processing on first error"
    )


class DiagramBatchResponse(BaseModel):
    """Response from batch diagram generation"""
    total: int
    successful: int
    failed: int
    results: List[DiagramGenerateResponse]
    errors: List[Dict[str, Any]]


class DiagramTemplatesResponse(BaseModel):
    """Response listing available diagram types"""
    subjects: Dict[str, List[str]]


# ============================================================================
# Dependency for getting engine
# ============================================================================

async def get_engine(
    tenant_id: Optional[str] = Query(None, description="Tenant ID for multi-tenant isolation")
) -> DiagramEngine:
    """Get diagram engine instance"""
    return get_diagram_engine(tenant_id=tenant_id)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/generate", response_model=DiagramGenerateResponse)
async def generate_diagram(
    request: DiagramGenerateRequest,
    engine: DiagramEngine = Depends(get_engine),
) -> DiagramGenerateResponse:
    """
    Generate a diagram from a specification.
    
    The specification must include:
    - subject: maths, physics, chemistry, or biology
    - diagram_type: Specific type for the subject (e.g., series_circuit)
    - Additional fields specific to the diagram type
    
    Returns the generated diagram URL and metadata.
    """
    try:
        # Merge format/quality into spec if provided
        spec = request.spec.copy()
        if request.output_format:
            spec['output_format'] = request.output_format
        if request.quality:
            spec['quality'] = request.quality
        
        # Generate the diagram
        result = await engine.generate(spec)
        
        return DiagramGenerateResponse(
            success=True,
            diagram_id=result.diagram_id,
            url=result.url,
            format=result.format.value,
            width=result.width,
            height=result.height,
            file_size_bytes=result.file_size_bytes,
            cached=result.cached,
            generation_time_ms=result.generation_time_ms,
        )
        
    except ValueError as e:
        logger.warning(f"Diagram validation error: {e}")
        return DiagramGenerateResponse(
            success=False,
            error=str(e),
            error_details={'type': 'validation_error'}
        )
    except RenderError as e:
        logger.error(f"Diagram render error: {e}")
        return DiagramGenerateResponse(
            success=False,
            error=e.message,
            error_details=e.details
        )
    except Exception as e:
        logger.exception(f"Unexpected diagram generation error: {e}")
        return DiagramGenerateResponse(
            success=False,
            error="Internal server error",
            error_details={'type': type(e).__name__}
        )


@router.post("/generate-batch", response_model=DiagramBatchResponse)
async def generate_diagrams_batch(
    request: DiagramBatchRequest,
    engine: DiagramEngine = Depends(get_engine),
) -> DiagramBatchResponse:
    """
    Generate multiple diagrams in a single request.
    
    Useful for generating all diagrams for a question paper.
    Maximum 50 diagrams per request.
    """
    results = []
    errors = []
    
    for i, diagram_request in enumerate(request.diagrams):
        try:
            spec = diagram_request.spec.copy()
            if diagram_request.output_format:
                spec['output_format'] = diagram_request.output_format
            if diagram_request.quality:
                spec['quality'] = diagram_request.quality
            
            result = await engine.generate(spec)
            
            results.append(DiagramGenerateResponse(
                success=True,
                diagram_id=result.diagram_id,
                url=result.url,
                format=result.format.value,
                width=result.width,
                height=result.height,
                file_size_bytes=result.file_size_bytes,
                cached=result.cached,
                generation_time_ms=result.generation_time_ms,
            ))
            
        except Exception as e:
            error_response = DiagramGenerateResponse(
                success=False,
                error=str(e),
            )
            results.append(error_response)
            errors.append({
                'index': i,
                'error': str(e),
            })
            
            if request.stop_on_error:
                break
    
    successful = sum(1 for r in results if r.success)
    
    return DiagramBatchResponse(
        total=len(request.diagrams),
        successful=successful,
        failed=len(errors),
        results=results,
        errors=errors,
    )


@router.get("/templates", response_model=DiagramTemplatesResponse)
async def get_templates() -> DiagramTemplatesResponse:
    """
    Get all available diagram types organized by subject.
    """
    return DiagramTemplatesResponse(
        subjects={s.value: types for s, types in SUPPORTED_DIAGRAM_TYPES.items()}
    )


@router.get("/templates/{subject}")
async def get_subject_templates(
    subject: str,
) -> Dict[str, Any]:
    """
    Get available diagram types for a specific subject.
    """
    try:
        subj = DiagramSubject(subject.lower())
        types = SUPPORTED_DIAGRAM_TYPES.get(subj, [])
        return {
            "subject": subject,
            "diagram_types": types,
            "count": len(types),
        }
    except ValueError:
        valid = [s.value for s in DiagramSubject]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid subject '{subject}'. Must be one of: {valid}"
        )


@router.get("/templates/{subject}/{diagram_type}/schema")
async def get_diagram_schema(
    subject: str,
    diagram_type: str,
) -> Dict[str, Any]:
    """
    Get the JSON schema for a specific diagram type.
    
    This schema describes the required and optional fields
    for the diagram specification.
    """
    try:
        subj = DiagramSubject(subject.lower())
    except ValueError:
        valid = [s.value for s in DiagramSubject]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid subject '{subject}'. Must be one of: {valid}"
        )
    
    valid_types = SUPPORTED_DIAGRAM_TYPES.get(subj, [])
    if diagram_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diagram type '{diagram_type}' for subject '{subject}'. Valid types: {valid_types}"
        )
    
    # Return basic schema (specific schemas will be added per diagram type)
    from services.diagram_engine.spec_validator import get_validator
    validator = get_validator()
    base_schema = validator.get_schema(subj)
    
    return {
        "subject": subject,
        "diagram_type": diagram_type,
        "schema": base_schema,
    }


@router.get("/{diagram_id}")
async def get_diagram_metadata(
    diagram_id: str,
    engine: DiagramEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """
    Get metadata for a generated diagram.
    
    Returns information about the diagram including its URL,
    dimensions, and generation details.
    """
    # For now, return basic info
    # Full implementation would query MongoDB
    return {
        "diagram_id": diagram_id,
        "message": "Diagram metadata lookup not yet implemented. Use the URL from generation response.",
    }


@router.get("/{diagram_id}/download")
async def download_diagram(
    diagram_id: str,
    engine: DiagramEngine = Depends(get_engine),
):
    """
    Download a generated diagram file.
    
    Returns the diagram image/file directly.
    """
    # Try to get from output handler
    data = await engine.output_handler.get(diagram_id)
    
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Diagram not found: {diagram_id}"
        )
    
    # Determine content type from diagram_id (assumes format in id)
    if diagram_id.endswith('.svg') or '.svg' in diagram_id:
        content_type = "image/svg+xml"
        filename = f"{diagram_id}.svg"
    elif diagram_id.endswith('.pdf') or '.pdf' in diagram_id:
        content_type = "application/pdf"
        filename = f"{diagram_id}.pdf"
    else:
        content_type = "image/png"
        filename = f"{diagram_id}.png"
    
    return StreamingResponse(
        io.BytesIO(data),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(data)),
        }
    )


@router.delete("/{diagram_id}")
async def delete_diagram(
    diagram_id: str,
    engine: DiagramEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """
    Delete a generated diagram.
    
    Removes the diagram from storage and cache.
    """
    success = await engine.output_handler.delete(diagram_id)
    
    if success:
        return {
            "success": True,
            "message": f"Diagram {diagram_id} deleted",
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Diagram not found or already deleted: {diagram_id}"
        )


@router.get("/stats/engine")
async def get_engine_stats(
    engine: DiagramEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """
    Get diagram engine statistics.
    
    Returns information about registered renderers, cache status,
    and supported diagram types.
    """
    return engine.get_engine_stats()


@router.post("/cache/clear")
async def clear_cache(
    engine: DiagramEngine = Depends(get_engine),
) -> Dict[str, Any]:
    """
    Clear the diagram cache.
    
    Forces regeneration of all subsequent diagram requests.
    """
    count = engine.clear_cache()
    return {
        "success": True,
        "cleared_entries": count,
    }
