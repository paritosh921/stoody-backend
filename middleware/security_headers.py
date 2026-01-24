"""
Security Headers Middleware for Stoody
Adds comprehensive security headers to all responses
"""

import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config_async import DEBUG_MODE

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all HTTP responses
    
    Headers Added:
    - Content-Security-Policy (CSP)
    - Strict-Transport-Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    """
    
    def __init__(self, app, csp_policy: str = None):
        super().__init__(app)
        self.csp_policy = csp_policy or self._get_default_csp()
        
    def _get_default_csp(self) -> str:
        """
        Get default Content Security Policy
        
        Returns:
            CSP policy string
        """
        if DEBUG_MODE:
            # Relaxed CSP for development
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://accounts.google.com https://accounts.google.com/gsi/client; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' data: https://fonts.gstatic.com; "
                "img-src 'self' data: https: blob:; "
                "connect-src 'self' http://localhost:5001 https://api.stoody.in https://*.stoody.in https://accounts.google.com; "
                "frame-src 'self' https://accounts.google.com; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'; "
                "object-src 'none';"
            )
        else:
            # Strict CSP for production
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://accounts.google.com https://accounts.google.com/gsi/client; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' data: https://fonts.gstatic.com; "
                "img-src 'self' data: https: blob:; "
                "connect-src 'self' https://api.stoody.in https://*.stoody.in https://accounts.google.com; "
                "frame-src 'self' https://accounts.google.com; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'; "
                "object-src 'none'; "
                "upgrade-insecure-requests;"
            )
        
        return csp
        
    async def dispatch(self, request: Request, call_next):
        """
        Process request and add security headers to response
        
        Args:
            request: FastAPI Request
            call_next: Next middleware/endpoint
            
        Returns:
            Response with security headers
        """
        # Call next middleware/endpoint
        response = await call_next(request)
        
        # Add Content Security Policy
        response.headers["Content-Security-Policy"] = self.csp_policy
        
        # Add Strict-Transport-Security (HSTS) - only in production
        if not DEBUG_MODE:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Prevent clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Enable XSS filtering (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Control browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        # Indicate that server follows security best practices
        if not DEBUG_MODE:
            response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Cache control for sensitive data
        if request.url.path.startswith("/api/v1/auth") or request.url.path.startswith("/auth"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # Add custom security header
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Backend-Server"] = "stoody-api"
        
        return response


def get_security_headers() -> dict:
    """
    Get security headers as dictionary (for manual response creation)
    
    Returns:
        Dictionary of security headers
    """
    headers = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        ),
    }
    
    if not DEBUG_MODE:
        headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    
    return headers
