"""
Subdomain extraction middleware for multi-tenant authentication
Extracts subdomain from Host header and attaches to request.state
"""
from fastapi import Request
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def extract_subdomain(host: str) -> Optional[str]:
    """
    Extract subdomain from host header

    Examples:
        demo.skillbot.app → "demo"
        school123.skillbot.app → "school123"
        app.skillbot.app → None (main app)
        skillbot.app → None (main domain)
        localhost:5173 → None (development)
        demo.skillbot.local:5173 → "demo" (local testing)
    """
    if not host:
        return None

    # Remove port if present
    host = host.split(':')[0]

    # Development: localhost or 127.0.0.1
    if host in ['localhost', '127.0.0.1']:
        return None

    # Split by dots
    parts = host.split('.')

    # Need at least subdomain.domain.tld (3 parts) OR subdomain.domain.local (3 parts for local testing)
    if len(parts) < 3:
        return None

    # First part is subdomain
    subdomain = parts[0]

    # Exclude reserved subdomains
    if subdomain in ['www', 'app', 'admin', 'api']:
        return None

    return subdomain


async def subdomain_middleware(request: Request, call_next):
    """
    Middleware to extract and attach subdomain to request state
    """
    host = request.headers.get("host", "")
    subdomain = extract_subdomain(host)

    # Attach to request state
    request.state.subdomain = subdomain

    logger.debug(f"Request host: {host}, extracted subdomain: {subdomain}")

    response = await call_next(request)
    return response
