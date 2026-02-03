"""
Specification Validator for Diagram Engine

Validates diagram specifications against JSON schemas and
provides helpful error messages.
"""

import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Type
from pydantic import ValidationError

from .specs.base_spec import (
    BaseDiagramSpec,
    DiagramSubject,
    DiagramError,
    SUPPORTED_DIAGRAM_TYPES,
    is_valid_diagram_type,
)

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation operation"""
    
    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[Dict[str, Any]]] = None,
        validated_spec: Optional[Dict[str, Any]] = None,
        spec_hash: Optional[str] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.validated_spec = validated_spec
        self.spec_hash = spec_hash
    
    def to_error(self) -> DiagramError:
        """Convert to DiagramError for API response"""
        if self.is_valid:
            raise ValueError("Cannot convert valid result to error")
        
        error_messages = [e.get('message', str(e)) for e in self.errors]
        return DiagramError(
            error_code="VALIDATION_ERROR",
            message="; ".join(error_messages) if error_messages else "Validation failed",
            details={'errors': self.errors},
            spec_hash=self.spec_hash,
        )


class SpecValidator:
    """
    Validates diagram specifications.
    
    Provides:
    - Schema validation using Pydantic
    - Subject/type validation
    - Helpful error messages
    - Spec normalization with defaults
    """
    
    # Required fields in any diagram spec
    REQUIRED_FIELDS = ['subject', 'diagram_type']
    
    # Subject name aliases (normalize to canonical names)
    SUBJECT_ALIASES = {
        'mathematics': 'maths',
        'math': 'maths',
        'phy': 'physics',
        'chem': 'chemistry',
        'bio': 'biology',
    }
    
    # Optional fields with defaults
    DEFAULT_VALUES = {
        'output_format': 'png',
        'quality': 'high',
        'dimensions': {'width': 800, 'height': 600},
        'style': {
            'background_color': '#ffffff',
            'line_color': '#000000',
            'font_family': 'Arial',
            'font_size': 12,
            'line_width': 1.5,
        },
    }
    
    def __init__(self):
        """Initialize the validator"""
        # Registry of subject-specific validators
        self._subject_validators: Dict[DiagramSubject, Type[BaseDiagramSpec]] = {}
    
    def register_spec_class(
        self,
        subject: DiagramSubject,
        spec_class: Type[BaseDiagramSpec]
    ) -> None:
        """
        Register a subject-specific spec class for validation.
        
        Args:
            subject: The subject to register for
            spec_class: The Pydantic model class
        """
        self._subject_validators[subject] = spec_class
        logger.info(f"Registered spec class for {subject.value}: {spec_class.__name__}")
    
    def validate(self, spec: Dict[str, Any]) -> ValidationResult:
        """
        Validate a diagram specification.
        
        Args:
            spec: The specification dictionary
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in spec:
                errors.append({
                    'field': field,
                    'message': f"Missing required field: {field}",
                    'type': 'missing_field',
                })
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate subject (with alias normalization)
        subject_str = spec.get('subject', '').lower()
        # Apply alias normalization
        subject_str = self.SUBJECT_ALIASES.get(subject_str, subject_str)
        try:
            subject = DiagramSubject(subject_str)
        except ValueError:
            valid_subjects = [s.value for s in DiagramSubject]
            errors.append({
                'field': 'subject',
                'message': f"Invalid subject '{subject_str}'. Must be one of: {valid_subjects}",
                'type': 'invalid_value',
                'valid_values': valid_subjects,
            })
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate diagram type
        diagram_type = spec.get('diagram_type', '')
        if not is_valid_diagram_type(subject, diagram_type):
            valid_types = SUPPORTED_DIAGRAM_TYPES.get(subject, [])
            errors.append({
                'field': 'diagram_type',
                'message': f"Invalid diagram type '{diagram_type}' for subject '{subject.value}'",
                'type': 'invalid_value',
                'valid_values': valid_types,
            })
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate output format
        output_format = spec.get('output_format', 'png').lower()
        valid_formats = ['png', 'svg', 'pdf']
        if output_format not in valid_formats:
            errors.append({
                'field': 'output_format',
                'message': f"Invalid output format '{output_format}'. Must be one of: {valid_formats}",
                'type': 'invalid_value',
                'valid_values': valid_formats,
            })
        
        # Validate quality
        quality = spec.get('quality', 'high').lower()
        valid_qualities = ['low', 'medium', 'high']
        if quality not in valid_qualities:
            errors.append({
                'field': 'quality',
                'message': f"Invalid quality '{quality}'. Must be one of: {valid_qualities}",
                'type': 'invalid_value',
                'valid_values': valid_qualities,
            })
        
        # Validate dimensions
        dimensions = spec.get('dimensions', {})
        if dimensions:
            width = dimensions.get('width', 800)
            height = dimensions.get('height', 600)
            
            if not isinstance(width, int) or width < 100 or width > 4000:
                errors.append({
                    'field': 'dimensions.width',
                    'message': f"Width must be an integer between 100 and 4000, got: {width}",
                    'type': 'invalid_value',
                })
            
            if not isinstance(height, int) or height < 100 or height > 4000:
                errors.append({
                    'field': 'dimensions.height',
                    'message': f"Height must be an integer between 100 and 4000, got: {height}",
                    'type': 'invalid_value',
                })
        
        # Validate style
        style = spec.get('style', {})
        if style:
            # Validate hex colors
            for color_field in ['background_color', 'line_color']:
                if color_field in style:
                    color = style[color_field]
                    if not self._is_valid_hex_color(color):
                        errors.append({
                            'field': f'style.{color_field}',
                            'message': f"Invalid hex color: {color}. Format: #RRGGBB",
                            'type': 'invalid_format',
                        })
            
            # Validate font size
            if 'font_size' in style:
                font_size = style['font_size']
                if not isinstance(font_size, int) or font_size < 8 or font_size > 48:
                    errors.append({
                        'field': 'style.font_size',
                        'message': f"Font size must be between 8 and 48, got: {font_size}",
                        'type': 'invalid_value',
                    })
        
        if errors:
            return ValidationResult(is_valid=False, errors=errors)
        
        # Normalize spec with defaults
        normalized = self.normalize(spec)
        
        # Generate spec hash from the FULL normalized spec (not just base fields)
        # This ensures that diagram-specific parameters (points, range, functions, etc.)
        # are included in the hash for proper cache differentiation
        try:
            # Create hash from the FULL normalized spec
            spec_json = json.dumps(normalized, sort_keys=True, default=str)
            spec_hash = hashlib.sha256(spec_json.encode()).hexdigest()[:16]
            
            # Still validate with BaseDiagramSpec for type checking
            base_spec = BaseDiagramSpec(**{
                k: v for k, v in normalized.items() 
                if k in BaseDiagramSpec.model_fields
            })
        except ValidationError as e:
            errors = [
                {
                    'field': '.'.join(str(loc) for loc in err['loc']),
                    'message': err['msg'],
                    'type': err['type'],
                }
                for err in e.errors()
            ]
            return ValidationResult(is_valid=False, errors=errors)
        
        return ValidationResult(
            is_valid=True,
            validated_spec=normalized,
            spec_hash=spec_hash,
        )
    
    def normalize(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a spec by applying defaults.
        
        Args:
            spec: The specification dictionary
            
        Returns:
            Normalized specification with defaults applied
        """
        result = {}
        
        # Copy required fields (with alias normalization)
        subject_str = spec['subject'].lower()
        result['subject'] = self.SUBJECT_ALIASES.get(subject_str, subject_str)
        result['diagram_type'] = spec['diagram_type']
        
        # Apply defaults for optional fields
        result['output_format'] = spec.get('output_format', self.DEFAULT_VALUES['output_format']).lower()
        result['quality'] = spec.get('quality', self.DEFAULT_VALUES['quality']).lower()
        
        # Merge dimensions with defaults
        default_dims = self.DEFAULT_VALUES['dimensions'].copy()
        spec_dims = spec.get('dimensions', {})
        result['dimensions'] = {**default_dims, **spec_dims}
        
        # Merge style with defaults
        default_style = self.DEFAULT_VALUES['style'].copy()
        spec_style = spec.get('style', {})
        result['style'] = {**default_style, **spec_style}
        
        # Copy any additional fields (subject-specific)
        known_fields = {'subject', 'diagram_type', 'output_format', 'quality', 'dimensions', 'style', 'metadata'}
        for key, value in spec.items():
            if key not in known_fields:
                result[key] = value
        
        # Copy metadata if present
        if 'metadata' in spec:
            result['metadata'] = spec['metadata']
        
        return result
    
    def _is_valid_hex_color(self, color: str) -> bool:
        """Check if a string is a valid hex color"""
        if not isinstance(color, str):
            return False
        if not color.startswith('#'):
            return False
        if len(color) != 7:
            return False
        try:
            int(color[1:], 16)
            return True
        except ValueError:
            return False
    
    def get_schema(self, subject: Optional[DiagramSubject] = None) -> Dict[str, Any]:
        """
        Get JSON schema for diagram specifications.
        
        Args:
            subject: Optional subject to get schema for
            
        Returns:
            JSON schema dictionary
        """
        base_schema = BaseDiagramSpec.model_json_schema()
        
        if subject:
            # Add subject-specific types
            base_schema['properties']['diagram_type']['enum'] = SUPPORTED_DIAGRAM_TYPES.get(subject, [])
        else:
            # Add all types
            all_types = []
            for types in SUPPORTED_DIAGRAM_TYPES.values():
                all_types.extend(types)
            base_schema['properties']['diagram_type']['enum'] = list(set(all_types))
        
        return base_schema
    
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
            return {subject.value: SUPPORTED_DIAGRAM_TYPES.get(subject, [])}
        return {s.value: types for s, types in SUPPORTED_DIAGRAM_TYPES.items()}


# ============================================================================
# Singleton instance
# ============================================================================

_validator: Optional[SpecValidator] = None


def get_validator() -> SpecValidator:
    """Get the singleton validator instance"""
    global _validator
    if _validator is None:
        _validator = SpecValidator()
    return _validator
