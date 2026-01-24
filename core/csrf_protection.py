"""
CSRF Protection Module for Stoody
Implements double-submit cookie pattern for CSRF protection
"""

import secrets
import hashlib
import hmac
import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException, status

from config_async import JWT_SECRET_KEY, DEBUG_MODE

logger = logging.getLogger(__name__)

# CSRF configuration
CSRF_TOKEN_LENGTH = 32
CSRF_TOKEN_EXPIRY = 3600  # 1 hour in seconds
CSRF_COOKIE_NAME = "stoody_csrf_token"
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_COOKIE_HTTPONLY = False  # Must be readable by JavaScript
CSRF_COOKIE_SECURE = not DEBUG_MODE
CSRF_COOKIE_SAMESITE = "strict"

class CSRFProtection:
    """CSRF token generation and validation using double-submit cookie pattern"""
    
    def __init__(self):
        self.secret = JWT_SECRET_KEY
        self.token_length = CSRF_TOKEN_LENGTH
        self.expiry = CSRF_TOKEN_EXPIRY
        
    def generate_csrf_token(self) -> str:
        """
        Generate a secure random CSRF token
        
        Returns:
            URL-safe random token string
        """
        # Generate cryptographically secure random token
        token = secrets.token_urlsafe(self.token_length)
        
        # Create HMAC signature to prevent token forgery
        signature = self._create_signature(token)
        
        # Combine token and signature
        csrf_token = f"{token}.{signature}"
        
        logger.debug("Generated new CSRF token")
        
        return csrf_token
        
    def _create_signature(self, token: str) -> str:
        """
        Create HMAC signature for CSRF token
        
        Args:
            token: CSRF token to sign
            
        Returns:
            Hex-encoded HMAC signature
        """
        signature = hmac.new(
            self.secret.encode(),
            token.encode(),
            hashlib.sha256
        ).hexdigest()[:16]  # Truncate to 16 chars for brevity
        
        return signature
        
    def validate_csrf_token(self, token: str) -> bool:
        """
        Validate CSRF token signature
        
        Args:
            token: CSRF token to validate (format: "token.signature")
            
        Returns:
            True if valid, False otherwise
        """
        if not token:
            return False
            
        try:
            # Split token and signature
            parts = token.split(".")
            if len(parts) != 2:
                logger.warning("Invalid CSRF token format")
                return False
                
            token_part, signature_part = parts
            
            # Verify signature
            expected_signature = self._create_signature(token_part)
            
            if not hmac.compare_digest(signature_part, expected_signature):
                logger.warning("CSRF token signature mismatch")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"CSRF token validation error: {str(e)}")
            return False
            
    def set_csrf_cookie(self, response: Response, token: str) -> None:
        """
        Set CSRF token in cookie (double-submit pattern)
        
        Args:
            response: FastAPI Response object
            token: CSRF token to set
        """
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            max_age=self.expiry,
            path="/",
            domain=None,  # Set to ".stoody.in" for subdomain support
            secure=CSRF_COOKIE_SECURE,
            httponly=CSRF_COOKIE_HTTPONLY,  # Must be False for JS access
            samesite=CSRF_COOKIE_SAMESITE
        )
        
        logger.debug("Set CSRF cookie")
        
    def get_csrf_token_from_cookie(self, request: Request) -> Optional[str]:
        """
        Get CSRF token from cookie
        
        Args:
            request: FastAPI Request object
            
        Returns:
            CSRF token or None
        """
        return request.cookies.get(CSRF_COOKIE_NAME)
        
    def get_csrf_token_from_header(self, request: Request) -> Optional[str]:
        """
        Get CSRF token from request header
        
        Args:
            request: FastAPI Request object
            
        Returns:
            CSRF token or None
        """
        return request.headers.get(CSRF_HEADER_NAME)
        
    def verify_csrf_protection(self, request: Request) -> bool:
        """
        Verify CSRF protection using double-submit cookie pattern
        
        Args:
            request: FastAPI Request object
            
        Returns:
            True if CSRF verification passes
            
        Raises:
            HTTPException: If CSRF verification fails
        """
        # Skip CSRF check for safe methods
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return True
            
        # Skip CSRF check in development mode
        if DEBUG_MODE:
            logger.debug("CSRF check skipped in development mode")
            return True
            
        # Get token from cookie and header
        cookie_token = self.get_csrf_token_from_cookie(request)
        header_token = self.get_csrf_token_from_header(request)
        
        # Validate token exists
        if not cookie_token or not header_token:
            logger.warning("Missing CSRF token in request")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token missing"
            )
            
        # Validate tokens match (double-submit check)
        if not hmac.compare_digest(cookie_token, header_token):
            logger.warning("CSRF token mismatch")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token mismatch"
            )
            
        # Validate token signature
        if not self.validate_csrf_token(cookie_token):
            logger.warning("Invalid CSRF token")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token"
            )
            
        logger.debug("CSRF verification passed")
        return True


# Global instance
csrf_protection = CSRFProtection()


async def csrf_protect(request: Request) -> None:
    """
    CSRF protection middleware/dependency
    
    Args:
        request: FastAPI Request object
        
    Raises:
        HTTPException: If CSRF verification fails
    """
    csrf_protection.verify_csrf_protection(request)
