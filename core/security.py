"""
Security Utilities for SkillBot Backend

This module provides security-related utilities including:
- Password validation and hashing
- Input sanitization for MongoDB queries
- Security event logging
- Rate limiting helpers

Production-grade security implementations.
"""

import re
import hmac
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps

# Configure security logger
security_logger = logging.getLogger("security")


# =============================================================================
# PASSWORD VALIDATION
# =============================================================================

class PasswordValidationError(Exception):
    """Raised when password validation fails"""
    def __init__(self, message: str, errors: List[str]):
        self.message = message
        self.errors = errors
        super().__init__(self.message)


class PasswordValidator:
    """
    Validates password strength and compliance with security policies.

    Bcrypt has a 72-byte limit - passwords are automatically truncated.
    This validator warns users about this limitation.
    """

    # bcrypt limitation - passwords longer than 72 bytes are silently truncated
    BCRYPT_MAX_BYTES = 72

    def __init__(
        self,
        min_length: int = 8,
        max_length: int = 72,
        require_uppercase: bool = False,
        require_lowercase: bool = False,
        require_digit: bool = False,
        require_special: bool = False,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digit = require_digit
        self.require_special = require_special
        self.special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")

    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against security policy.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not password:
            errors.append("Password is required")
            return False, errors

        # Check minimum length
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters")

        # Check maximum length (bcrypt limitation)
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > self.BCRYPT_MAX_BYTES:
            errors.append(
                f"Password exceeds {self.BCRYPT_MAX_BYTES} bytes when encoded. "
                f"For security, please use a shorter password (current: {len(password_bytes)} bytes). "
                "Note: Non-ASCII characters may use multiple bytes."
            )
        elif len(password) > self.max_length:
            errors.append(f"Password must not exceed {self.max_length} characters")

        # Check for uppercase
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        # Check for lowercase
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        # Check for digit
        if self.require_digit and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")

        # Check for special character
        if self.require_special and not any(c in self.special_chars for c in password):
            errors.append("Password must contain at least one special character")

        # Check for common weak passwords
        weak_passwords = {
            "password", "password1", "password123", "123456", "12345678",
            "qwerty", "abc123", "letmein", "welcome", "admin", "login",
            "passw0rd", "master", "hello", "freedom", "whatever", "shadow",
            "qazwsx", "trustno1", "iloveyou"
        }
        if password.lower() in weak_passwords:
            errors.append("This password is too common. Please choose a stronger password.")

        return len(errors) == 0, errors

    def validate_or_raise(self, password: str) -> None:
        """Validate password and raise PasswordValidationError if invalid"""
        is_valid, errors = self.validate(password)
        if not is_valid:
            raise PasswordValidationError(
                "Password does not meet security requirements",
                errors
            )

    def get_strength_score(self, password: str) -> int:
        """
        Calculate password strength score (0-100).

        Scoring:
        - Length: up to 30 points
        - Character variety: up to 40 points
        - Bonus for mixed case: 10 points
        - Bonus for numbers: 10 points
        - Bonus for special chars: 10 points
        """
        if not password:
            return 0

        score = 0

        # Length score (up to 30 points)
        length = len(password)
        if length >= 16:
            score += 30
        elif length >= 12:
            score += 25
        elif length >= 8:
            score += 20
        elif length >= 6:
            score += 10
        else:
            score += 5

        # Character variety (up to 40 points)
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in self.special_chars for c in password)

        variety_count = sum([has_lower, has_upper, has_digit, has_special])
        score += variety_count * 10

        # Bonus for mixed case
        if has_lower and has_upper:
            score += 10

        # Bonus for numbers
        if has_digit:
            score += 10

        # Bonus for special chars
        if has_special:
            score += 10

        # Penalty for common patterns
        if password.lower() in {"password", "123456", "qwerty"}:
            score = max(0, score - 50)

        return min(100, score)


# Default validator instance - configure from settings
def get_password_validator() -> PasswordValidator:
    """Get configured password validator"""
    from config_async import settings
    return PasswordValidator(
        min_length=settings.PASSWORD_MIN_LENGTH,
        max_length=settings.PASSWORD_MAX_LENGTH,
        require_uppercase=settings.PASSWORD_REQUIRE_UPPERCASE,
        require_lowercase=settings.PASSWORD_REQUIRE_LOWERCASE,
        require_digit=settings.PASSWORD_REQUIRE_DIGIT,
        require_special=settings.PASSWORD_REQUIRE_SPECIAL,
    )


# =============================================================================
# MONGODB INPUT SANITIZATION
# =============================================================================

class MongoSanitizer:
    """
    Sanitizes user input for safe use in MongoDB queries.

    Prevents NoSQL injection attacks by escaping special characters
    and validating query operators.
    """

    # MongoDB operators that could be dangerous if injected
    DANGEROUS_OPERATORS = {
        "$where", "$function", "$accumulator", "$expr",
        "$jsonSchema", "$regex", "$text", "$mod"
    }

    # Regex special characters that need escaping
    REGEX_SPECIAL_CHARS = r'[\^\$\.\*\+\?\(\)\[\]\{\}\|\\]'

    @staticmethod
    def escape_regex(pattern: str) -> str:
        """
        Escape special regex characters in user input for safe use in $regex queries.

        This prevents regex injection attacks where malicious input could cause:
        - ReDoS (Regular Expression Denial of Service)
        - Unintended pattern matching
        - Information disclosure

        Args:
            pattern: User-provided search pattern

        Returns:
            Safely escaped pattern for use in MongoDB $regex
        """
        if not pattern:
            return ""

        # Escape all regex special characters
        return re.escape(pattern)

    @staticmethod
    def escape_for_like_search(pattern: str) -> str:
        """
        Escape pattern for case-insensitive substring search.

        Use this for search inputs where you want to find substrings.

        Args:
            pattern: User search input

        Returns:
            Escaped pattern safe for MongoDB $regex with $options: 'i'
        """
        if not pattern:
            return ""

        # Trim whitespace
        pattern = pattern.strip()

        # Escape special characters
        escaped = re.escape(pattern)

        return escaped

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """
        Sanitize a string value for safe database storage.

        - Trims whitespace
        - Limits length
        - Removes null bytes
        - Strips control characters (except newlines, tabs)

        Args:
            value: Input string
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        if not value:
            return ""

        if not isinstance(value, str):
            value = str(value)

        # Remove null bytes
        value = value.replace('\x00', '')

        # Remove control characters except \n, \r, \t
        value = ''.join(
            c for c in value
            if c in '\n\r\t' or (ord(c) >= 32 or ord(c) == 9)
        )

        # Trim and limit length
        value = value.strip()[:max_length]

        return value

    @staticmethod
    def validate_object_id(value: str) -> bool:
        """
        Validate that a string is a valid MongoDB ObjectId format.

        Args:
            value: String to validate

        Returns:
            True if valid ObjectId format, False otherwise
        """
        if not value or not isinstance(value, str):
            return False

        # ObjectId is 24 hex characters
        return bool(re.match(r'^[0-9a-fA-F]{24}$', value))

    @staticmethod
    def sanitize_query(query: Dict[str, Any], allow_operators: bool = False) -> Dict[str, Any]:
        """
        Sanitize a MongoDB query dictionary.

        Recursively checks for dangerous operators and sanitizes values.

        Args:
            query: Query dictionary
            allow_operators: Whether to allow MongoDB operators (for internal use only)

        Returns:
            Sanitized query dictionary

        Raises:
            ValueError: If dangerous operators detected and not allowed
        """
        if not query:
            return {}

        sanitized = {}

        for key, value in query.items():
            # Check for operator injection in keys
            if key.startswith('$'):
                if not allow_operators:
                    raise ValueError(f"MongoDB operator '{key}' not allowed in user input")
                if key.lower() in MongoSanitizer.DANGEROUS_OPERATORS:
                    raise ValueError(f"Dangerous MongoDB operator '{key}' detected")

            # Recursively sanitize nested dictionaries
            if isinstance(value, dict):
                sanitized[key] = MongoSanitizer.sanitize_query(value, allow_operators)
            elif isinstance(value, list):
                sanitized[key] = [
                    MongoSanitizer.sanitize_query(item, allow_operators)
                    if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, str):
                sanitized[key] = MongoSanitizer.sanitize_string(value)
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def build_safe_search_query(
        search_term: str,
        fields: List[str],
        case_insensitive: bool = True
    ) -> Dict[str, Any]:
        """
        Build a safe MongoDB search query for multiple fields.

        Creates an $or query that searches across specified fields
        using properly escaped regex patterns.

        Args:
            search_term: User search input
            fields: List of field names to search
            case_insensitive: Whether to use case-insensitive search

        Returns:
            MongoDB query dictionary
        """
        if not search_term or not fields:
            return {}

        # Escape the search term
        escaped_term = MongoSanitizer.escape_for_like_search(search_term)

        if not escaped_term:
            return {}

        # Build search conditions for each field
        conditions = []
        for field in fields:
            condition = {
                field: {
                    "$regex": escaped_term,
                    "$options": "i" if case_insensitive else ""
                }
            }
            conditions.append(condition)

        return {"$or": conditions} if conditions else {}


# =============================================================================
# SECURITY EVENT LOGGING
# =============================================================================

class SecurityEventType:
    """Security event type constants"""
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGIN_BLOCKED = "LOGIN_BLOCKED"
    LOGOUT = "LOGOUT"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    PASSWORD_RESET_REQUEST = "PASSWORD_RESET_REQUEST"
    PASSWORD_RESET_COMPLETE = "PASSWORD_RESET_COMPLETE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    AUTHORIZATION_FAILURE = "AUTHORIZATION_FAILURE"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    ACCOUNT_LOCKED = "ACCOUNT_LOCKED"
    ACCOUNT_UNLOCKED = "ACCOUNT_UNLOCKED"
    TWO_FA_ENABLED = "TWO_FA_ENABLED"
    TWO_FA_DISABLED = "TWO_FA_DISABLED"
    TWO_FA_FAILURE = "TWO_FA_FAILURE"
    INJECTION_ATTEMPT = "INJECTION_ATTEMPT"
    INVALID_INPUT = "INVALID_INPUT"


class SecurityLogger:
    """
    Centralized security event logging.

    Logs security-relevant events to both:
    - Application log (for real-time monitoring)
    - Security-specific log file (for auditing)

    Format includes:
    - Timestamp
    - Event type
    - User/session identifier
    - IP address
    - User agent
    - Additional context
    """

    def __init__(self, enabled: bool = True, log_file: str = "logs/security.log"):
        self.enabled = enabled
        self.log_file = log_file

        # Configure file handler if enabled
        if enabled:
            self._setup_file_handler()

    def _setup_file_handler(self):
        """Set up file handler for security log"""
        try:
            from pathlib import Path
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(self.log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            security_logger.addHandler(handler)
            security_logger.setLevel(logging.INFO)
        except Exception as e:
            logging.warning(f"Failed to set up security log file: {e}")

    def log_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        tenant_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a security event.

        Args:
            event_type: Type of security event (use SecurityEventType constants)
            user_id: User ID if known
            username: Username if known
            email: Email if known
            ip_address: Client IP address
            user_agent: Client user agent
            tenant_id: Tenant ID for multi-tenant context
            success: Whether the operation was successful
            details: Additional event-specific details
        """
        if not self.enabled:
            return

        # Build log entry
        entry = {
            "event": event_type,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add user identification
        if user_id:
            entry["user_id"] = user_id
        if username:
            entry["username"] = username
        if email:
            # Mask email for privacy
            entry["email"] = self._mask_email(email)

        # Add request context
        if ip_address:
            entry["ip"] = ip_address
        if user_agent:
            # Truncate user agent
            entry["user_agent"] = user_agent[:200] if len(user_agent) > 200 else user_agent
        if tenant_id:
            entry["tenant_id"] = tenant_id

        # Add details
        if details:
            entry["details"] = details

        # Log to security logger
        log_level = logging.INFO if success else logging.WARNING
        security_logger.log(log_level, f"{event_type} | {entry}")

        # Also log to main application logger for critical events
        if event_type in {
            SecurityEventType.LOGIN_BLOCKED,
            SecurityEventType.AUTHORIZATION_FAILURE,
            SecurityEventType.INJECTION_ATTEMPT,
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            SecurityEventType.ACCOUNT_LOCKED,
        }:
            logging.warning(f"SECURITY: {event_type} - {entry}")

    def _mask_email(self, email: str) -> str:
        """Mask email for privacy in logs"""
        if not email or '@' not in email:
            return "***"

        local, domain = email.rsplit('@', 1)
        if len(local) <= 2:
            masked_local = "*" * len(local)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]

        return f"{masked_local}@{domain}"

    def log_login_attempt(
        self,
        success: bool,
        username: Optional[str] = None,
        email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_type: str = "unknown",
        failure_reason: Optional[str] = None,
    ):
        """Log a login attempt"""
        event_type = SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE

        details = {"user_type": user_type}
        if not success and failure_reason:
            details["reason"] = failure_reason

        self.log_event(
            event_type=event_type,
            username=username,
            email=email,
            ip_address=ip_address,
            user_agent=user_agent,
            tenant_id=tenant_id,
            success=success,
            details=details,
        )

    def log_rate_limit(
        self,
        ip_address: str,
        endpoint: str,
        limit: str,
        user_id: Optional[str] = None,
    ):
        """Log rate limit exceeded"""
        self.log_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            user_id=user_id,
            ip_address=ip_address,
            success=False,
            details={"endpoint": endpoint, "limit": limit},
        )

    def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: Optional[str] = None,
    ):
        """Log authorization failure (IDOR attempt, etc.)"""
        self.log_event(
            event_type=SecurityEventType.AUTHORIZATION_FAILURE,
            user_id=user_id,
            ip_address=ip_address,
            success=False,
            details={"resource": resource, "action": action},
        )

    def log_injection_attempt(
        self,
        ip_address: str,
        input_field: str,
        malicious_input: str,
        user_id: Optional[str] = None,
    ):
        """Log potential injection attempt"""
        # Truncate malicious input for logging
        safe_input = malicious_input[:100] if len(malicious_input) > 100 else malicious_input

        self.log_event(
            event_type=SecurityEventType.INJECTION_ATTEMPT,
            user_id=user_id,
            ip_address=ip_address,
            success=False,
            details={"field": input_field, "input_preview": safe_input},
        )


# Global security logger instance
def get_security_logger() -> SecurityLogger:
    """Get configured security logger instance"""
    from config_async import SECURITY_LOG_ENABLED, SECURITY_LOG_FILE
    return SecurityLogger(enabled=SECURITY_LOG_ENABLED, log_file=SECURITY_LOG_FILE)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_client_ip(request) -> str:
    """
    Extract client IP address from request.

    Handles X-Forwarded-For header for proxied requests.
    """
    # Check X-Forwarded-For header (set by proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()

    # Check X-Real-IP header
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct client
    if hasattr(request, 'client') and request.client:
        return request.client.host

    return "unknown"


def get_user_agent(request) -> str:
    """Extract user agent from request"""
    return request.headers.get("User-Agent", "unknown")


def secure_compare(a: str, b: str) -> bool:
    """
    Constant-time string comparison to prevent timing attacks.
    """
    return hmac.compare_digest(a.encode(), b.encode())
