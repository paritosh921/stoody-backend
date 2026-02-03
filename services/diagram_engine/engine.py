"""
Diagram Generation Engine

The main orchestrator that routes diagram generation requests
to the appropriate subject-specific renderer.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Type
from datetime import datetime

from .specs.base_spec import (
    DiagramSubject,
    DiagramResult,
    DiagramError,
    SUPPORTED_DIAGRAM_TYPES,
)
from .base_renderer import BaseRenderer, RenderResult, RenderError, PlaceholderRenderer
from .output_handler import OutputHandler, get_output_handler
from .spec_validator import SpecValidator, ValidationResult, get_validator

logger = logging.getLogger(__name__)


class DiagramEngine:
    """
    Main diagram generation engine.
    
    Routes requests to appropriate renderers based on subject,
    handles validation, storage, and caching.
    
    Usage:
        engine = DiagramEngine()
        result = await engine.generate({
            "subject": "physics",
            "diagram_type": "series_circuit",
            "components": [...]
        })
    """
    
    def __init__(
        self,
        tenant_id: Optional[str] = None,
        use_cache: bool = True,
        use_placeholders: bool = True,
    ):
        """
        Initialize the diagram engine.
        
        Args:
            tenant_id: Optional tenant ID for multi-tenant storage
            use_cache: Whether to cache generated diagrams
            use_placeholders: Use placeholder renderers for unimplemented types
        """
        self.tenant_id = tenant_id
        self.use_cache = use_cache
        self.use_placeholders = use_placeholders
        
        # Components
        self.validator = get_validator()
        self.output_handler = get_output_handler(tenant_id=tenant_id, use_cache=use_cache)
        
        # Renderer registry
        self._renderers: Dict[DiagramSubject, BaseRenderer] = {}
        
        # Initialize placeholder renderers first (as fallback)
        if use_placeholders:
            self._init_placeholders()
        
        # Initialize actual renderers (replace placeholders where available)
        self._init_actual_renderers()
        
        logger.info(
            f"DiagramEngine initialized "
            f"(tenant={tenant_id}, cache={use_cache}, placeholders={use_placeholders})"
        )
    
    def _init_placeholders(self) -> None:
        """Initialize placeholder renderers for all subjects"""
        for subject in DiagramSubject:
            if subject not in self._renderers:
                self._renderers[subject] = PlaceholderRenderer(subject)
                logger.debug(f"Placeholder renderer created for {subject.value}")
    
    def _init_actual_renderers(self) -> None:
        """Initialize actual renderers for all subjects (replaces placeholders)"""
        try:
            from .renderers import MathRenderer, PhysicsRenderer, ChemistryRenderer, BiologyRenderer
            
            # Register MathRenderer
            try:
                self._renderers[DiagramSubject.MATHS] = MathRenderer()
                logger.info("MathRenderer registered successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MathRenderer: {e}")
            
            # Register PhysicsRenderer
            try:
                self._renderers[DiagramSubject.PHYSICS] = PhysicsRenderer()
                logger.info("PhysicsRenderer registered successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PhysicsRenderer: {e}")
            
            # Register ChemistryRenderer
            try:
                self._renderers[DiagramSubject.CHEMISTRY] = ChemistryRenderer()
                logger.info("ChemistryRenderer registered successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize ChemistryRenderer: {e}")
            
            # Register BiologyRenderer
            try:
                self._renderers[DiagramSubject.BIOLOGY] = BiologyRenderer()
                logger.info("BiologyRenderer registered successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize BiologyRenderer: {e}")
                
        except ImportError as e:
            logger.warning(f"Could not import renderers: {e}. Using placeholders.")
    
    def register_renderer(self, renderer: BaseRenderer) -> None:
        """
        Register a renderer for a subject.
        
        Args:
            renderer: The renderer instance to register
        """
        self._renderers[renderer.subject] = renderer
        logger.info(
            f"Registered {renderer.__class__.__name__} for {renderer.subject.value} "
            f"(types: {renderer.get_supported_types()})"
        )
    
    def get_renderer(self, subject: DiagramSubject) -> Optional[BaseRenderer]:
        """
        Get the renderer for a subject.
        
        Args:
            subject: The subject to get renderer for
            
        Returns:
            The renderer, or None if not registered
        """
        return self._renderers.get(subject)
    
    def has_renderer(self, subject: DiagramSubject) -> bool:
        """Check if a renderer is registered for a subject"""
        renderer = self._renderers.get(subject)
        if renderer is None:
            return False
        # Check if it's a real renderer (not placeholder)
        return not isinstance(renderer, PlaceholderRenderer)
    
    async def generate(
        self,
        spec: Dict[str, Any],
        tenant_id: Optional[str] = None,
        skip_cache: bool = False,
    ) -> DiagramResult:
        """
        Generate a diagram from a specification.
        
        Args:
            spec: The diagram specification
            tenant_id: Optional tenant ID override
            skip_cache: If True, bypass cache and always generate fresh diagram
            
        Returns:
            DiagramResult with URL and metadata
            
        Raises:
            ValueError: If spec is invalid
            RenderError: If rendering fails
        """
        start_time = time.time()
        
        # Validate the spec
        validation = self.validator.validate(spec)
        if not validation.is_valid:
            error = validation.to_error()
            logger.warning(f"Validation failed: {error.message}")
            raise ValueError(error.message)
        
        # Get validated and normalized spec
        validated_spec = validation.validated_spec
        spec_hash = validation.spec_hash
        
        # Check cache (unless skip_cache is True)
        if self.use_cache and not skip_cache:
            cached = await self._check_cache(spec_hash)
            if cached:
                logger.info(f"Cache hit for {spec_hash}")
                return cached
        elif skip_cache:
            logger.info(f"Skipping cache check (skip_cache=True)")
        
        # Get the appropriate renderer
        subject = DiagramSubject(validated_spec['subject'])
        renderer = self.get_renderer(subject)
        
        if renderer is None:
            raise ValueError(f"No renderer available for subject: {subject.value}")
        
        # Check if diagram type is supported
        diagram_type = validated_spec['diagram_type']
        if not renderer.supports_type(diagram_type):
            raise ValueError(
                f"Diagram type '{diagram_type}' not supported by {renderer.__class__.__name__}"
            )
        
        # Render the diagram
        try:
            logger.info(f"Rendering {subject.value}/{diagram_type}")
            render_result = await renderer.render(validated_spec)
        except RenderError as e:
            logger.error(f"Render failed: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected render error: {e}")
            raise RenderError(
                message=str(e),
                diagram_type=diagram_type,
                details={'exception': type(e).__name__}
            )
        
        # Save to storage (pass skip_cache to bypass output handler cache)
        result = await self.output_handler.save(
            render_result=render_result,
            spec_hash=spec_hash,
            tenant_id=tenant_id or self.tenant_id,
            metadata={
                'subject': subject.value,
                'diagram_type': diagram_type,
            },
            skip_cache=skip_cache,
        )
        
        # Log timing
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Generated {subject.value}/{diagram_type} in {elapsed_ms}ms "
            f"({render_result.file_size} bytes)"
        )
        
        return result
    
    async def generate_batch(
        self,
        specs: List[Dict[str, Any]],
        tenant_id: Optional[str] = None,
        stop_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate multiple diagrams.
        
        Args:
            specs: List of diagram specifications
            tenant_id: Optional tenant ID override
            stop_on_error: Stop processing on first error
            
        Returns:
            Dictionary with results and any errors
        """
        results = []
        errors = []
        
        for i, spec in enumerate(specs):
            try:
                result = await self.generate(spec, tenant_id=tenant_id)
                results.append({
                    'index': i,
                    'result': result,
                })
            except (ValueError, RenderError) as e:
                error_info = {
                    'index': i,
                    'error': str(e),
                    'spec': spec,
                }
                errors.append(error_info)
                
                if stop_on_error:
                    break
        
        return {
            'total': len(specs),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors,
        }
    
    async def _check_cache(self, spec_hash: str) -> Optional[DiagramResult]:
        """Check if a diagram is in cache"""
        # This is handled by OutputHandler
        # The cache check happens during save() which returns cached result
        return None  # For now, let OutputHandler handle caching
    
    def get_supported_subjects(self) -> List[str]:
        """Get list of subjects with registered renderers"""
        return [s.value for s in self._renderers.keys()]
    
    def get_supported_types(
        self,
        subject: Optional[DiagramSubject] = None
    ) -> Dict[str, List[str]]:
        """
        Get supported diagram types.
        
        Args:
            subject: Optional subject to filter by
            
        Returns:
            Dictionary of subject -> list of types
        """
        if subject:
            renderer = self.get_renderer(subject)
            if renderer:
                return {subject.value: renderer.get_supported_types()}
            return {subject.value: []}
        
        result = {}
        for s, renderer in self._renderers.items():
            result[s.value] = renderer.get_supported_types()
        return result
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'tenant_id': self.tenant_id,
            'use_cache': self.use_cache,
            'use_placeholders': self.use_placeholders,
            'registered_renderers': {
                s.value: {
                    'class': r.__class__.__name__,
                    'is_placeholder': isinstance(r, PlaceholderRenderer),
                    'supported_types': r.get_supported_types(),
                }
                for s, r in self._renderers.items()
            },
            'cache_stats': self.output_handler.get_cache_stats(),
        }
    
    def clear_cache(self) -> int:
        """Clear the diagram cache"""
        return self.output_handler.clear_cache()


# ============================================================================
# Singleton instance for convenience
# ============================================================================

_engine: Optional[DiagramEngine] = None


def get_diagram_engine(
    tenant_id: Optional[str] = None,
    **kwargs
) -> DiagramEngine:
    """
    Get or create the diagram engine instance.
    
    Args:
        tenant_id: Optional tenant ID
        **kwargs: Additional arguments for DiagramEngine
        
    Returns:
        DiagramEngine instance
    """
    global _engine
    
    if _engine is None:
        _engine = DiagramEngine(tenant_id=tenant_id, **kwargs)
    elif tenant_id and tenant_id != _engine.tenant_id:
        # Create new engine for different tenant
        _engine = DiagramEngine(tenant_id=tenant_id, **kwargs)
    
    return _engine


async def generate_diagram(
    spec: Dict[str, Any],
    tenant_id: Optional[str] = None,
) -> DiagramResult:
    """
    Convenience function to generate a diagram.
    
    Args:
        spec: The diagram specification
        tenant_id: Optional tenant ID
        
    Returns:
        DiagramResult
    """
    engine = get_diagram_engine(tenant_id=tenant_id)
    return await engine.generate(spec, tenant_id=tenant_id)
