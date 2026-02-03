"""
Output Handler for Diagram Engine

Handles saving generated diagrams to S3 or local storage,
URL generation, and caching.
"""

import os
import logging
import hashlib
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from .base_renderer import RenderResult
from .specs.base_spec import OutputFormat, DiagramResult

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Handles diagram output storage and URL generation.
    
    Uses the existing S3 storage utility for consistent storage handling
    across the application.
    """
    
    # Default storage prefix for diagrams
    DIAGRAM_PREFIX = "diagrams"
    
    def __init__(
        self,
        tenant_id: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        """
        Initialize the output handler.
        
        Args:
            tenant_id: Optional tenant ID for multi-tenant isolation
            use_cache: Whether to use caching
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.tenant_id = tenant_id
        self.use_cache = use_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, DiagramResult] = {}
    
    def _generate_diagram_id(self) -> str:
        """Generate a unique diagram ID"""
        return f"diag_{uuid.uuid4().hex[:12]}"
    
    def _get_storage_key(
        self,
        diagram_id: str,
        output_format: OutputFormat,
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Generate the storage key (S3 key or local path).
        
        Format: diagrams/{tenant_id}/{diagram_id}.{ext}
        """
        tenant = tenant_id or self.tenant_id or "default"
        extension = output_format.value
        return f"{self.DIAGRAM_PREFIX}/{tenant}/{diagram_id}.{extension}"
    
    def _get_local_path(self, storage_key: str) -> str:
        """Convert storage key to local file path"""
        base_dir = os.path.join(os.getcwd(), "uploads")
        return os.path.join(base_dir, storage_key.replace("/", os.sep))
    
    async def save(
        self,
        render_result: RenderResult,
        spec_hash: str,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_cache: bool = False,
    ) -> DiagramResult:
        """
        Save a rendered diagram to storage.
        
        Args:
            render_result: The render result containing image data
            spec_hash: Hash of the specification (for caching)
            tenant_id: Optional tenant ID override
            metadata: Optional metadata to store
            skip_cache: If True, bypass cache and always save fresh diagram
            
        Returns:
            DiagramResult with storage info and URL
        """
        # Check cache first (unless skip_cache is True)
        if self.use_cache and not skip_cache and spec_hash in self._cache:
            cached = self._cache[spec_hash]
            logger.info(f"Cache hit for spec_hash: {spec_hash}")
            return DiagramResult(
                diagram_id=cached.diagram_id,
                url=cached.url,
                storage_path=cached.storage_path,
                format=cached.format,
                width=cached.width,
                height=cached.height,
                file_size_bytes=cached.file_size_bytes,
                spec_hash=spec_hash,
                cached=True,
                generated_at=cached.generated_at,
                generation_time_ms=cached.generation_time_ms,
            )
        elif skip_cache:
            logger.info(f"Skipping output cache (skip_cache=True) for spec_hash: {spec_hash}")
        
        start_time = datetime.utcnow()
        
        # Generate diagram ID and storage key
        diagram_id = self._generate_diagram_id()
        storage_key = self._get_storage_key(
            diagram_id,
            render_result.format,
            tenant_id or self.tenant_id
        )
        local_path = self._get_local_path(storage_key)
        
        # Try to use S3 storage
        try:
            from utils.s3_storage import upload_file, get_public_url, is_s3_enabled
            
            success, storage_path = await upload_file(
                file_data=render_result.image_data,
                local_path=local_path,
                content_type=render_result.content_type,
                metadata={
                    'diagram_id': diagram_id,
                    'spec_hash': spec_hash,
                    'format': render_result.format.value,
                    **(metadata or {})
                }
            )
            
            if success:
                # Get public URL
                url = get_public_url(storage_path)
                logger.info(f"Saved diagram to S3: {storage_path}")
            else:
                # Fallback failed, use local path
                storage_path = local_path
                url = f"/api/v1/diagrams/{diagram_id}/download"
                logger.warning(f"S3 upload failed, using local: {local_path}")
                
        except ImportError:
            # S3 module not available, use local storage
            storage_path = await self._save_local(
                render_result.image_data,
                local_path
            )
            url = f"/api/v1/diagrams/{diagram_id}/download"
            logger.info(f"Saved diagram locally: {local_path}")
        
        # Calculate generation time
        end_time = datetime.utcnow()
        generation_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Create result
        result = DiagramResult(
            diagram_id=diagram_id,
            url=url,
            storage_path=storage_path,
            format=render_result.format,
            width=render_result.width,
            height=render_result.height,
            file_size_bytes=render_result.file_size,
            spec_hash=spec_hash,
            cached=False,
            generated_at=render_result.generated_at,
            generation_time_ms=generation_time_ms,
        )
        
        # Cache the result
        if self.use_cache:
            self._cache[spec_hash] = result
        
        return result
    
    async def _save_local(self, data: bytes, path: str) -> str:
        """Save data to local filesystem"""
        import aiofiles
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)
        
        return path
    
    async def get(self, diagram_id: str) -> Optional[bytes]:
        """
        Retrieve a diagram by ID.
        
        Args:
            diagram_id: The diagram ID
            
        Returns:
            The diagram data as bytes, or None if not found
        """
        # First check cache
        for spec_hash, result in self._cache.items():
            if result.diagram_id == diagram_id:
                return await self._download(result.storage_path)
        
        # If not in cache, try to find the file in local storage
        # Try common formats
        base_dir = os.path.join(os.getcwd(), "uploads", self.DIAGRAM_PREFIX)
        for ext in ['png', 'svg', 'pdf']:
            # Check in all tenant directories
            if os.path.exists(base_dir):
                for tenant_dir in os.listdir(base_dir):
                    file_path = os.path.join(base_dir, tenant_dir, f"{diagram_id}.{ext}")
                    if os.path.exists(file_path):
                        try:
                            import aiofiles
                            async with aiofiles.open(file_path, 'rb') as f:
                                return await f.read()
                        except Exception as e:
                            logger.error(f"Error reading diagram file: {e}")
        
        return None
    
    async def _download(self, storage_path: str) -> Optional[bytes]:
        """Download a file from storage"""
        try:
            from utils.s3_storage import download_file
            return await download_file(storage_path)
        except ImportError:
            # Try local file
            import aiofiles
            if os.path.exists(storage_path):
                async with aiofiles.open(storage_path, 'rb') as f:
                    return await f.read()
            return None
    
    async def delete(self, diagram_id: str) -> bool:
        """
        Delete a diagram by ID.
        
        Args:
            diagram_id: The diagram ID
            
        Returns:
            True if deletion was successful
        """
        # Find in cache and remove
        to_remove = None
        for spec_hash, result in self._cache.items():
            if result.diagram_id == diagram_id:
                to_remove = spec_hash
                break
        
        if to_remove:
            result = self._cache.pop(to_remove)
            try:
                from utils.s3_storage import delete_file
                return await delete_file(result.storage_path)
            except ImportError:
                if os.path.exists(result.storage_path):
                    os.remove(result.storage_path)
                    return True
        
        return False
    
    def clear_cache(self) -> int:
        """
        Clear the in-memory cache.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} entries from diagram cache")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(r.file_size_bytes for r in self._cache.values())
        return {
            'entries': len(self._cache),
            'total_size_bytes': total_size,
            'use_cache': self.use_cache,
            'ttl_seconds': self.cache_ttl_seconds,
        }


# ============================================================================
# Singleton instance for convenience
# ============================================================================

_output_handler: Optional[OutputHandler] = None


def get_output_handler(
    tenant_id: Optional[str] = None,
    **kwargs
) -> OutputHandler:
    """
    Get or create the output handler instance.
    
    Args:
        tenant_id: Optional tenant ID
        **kwargs: Additional arguments for OutputHandler
        
    Returns:
        OutputHandler instance
    """
    global _output_handler
    
    if _output_handler is None:
        _output_handler = OutputHandler(tenant_id=tenant_id, **kwargs)
    elif tenant_id and tenant_id != _output_handler.tenant_id:
        # Create new handler for different tenant
        _output_handler = OutputHandler(tenant_id=tenant_id, **kwargs)
    
    return _output_handler
