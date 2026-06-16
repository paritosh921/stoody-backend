"""Role-scoped password reset OTP helpers."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple


class PasswordResetOtpManager:
    """Create and validate short-lived password reset OTP records."""

    def __init__(self, length: int = 6, expire_minutes: int = 10, max_attempts: int = 3):
        self.length = length
        self.expire_minutes = expire_minutes
        self.max_attempts = max_attempts

    def generate_otp(self) -> str:
        start = 10 ** (self.length - 1)
        end = (10 ** self.length) - 1
        return str(secrets.randbelow(end - start + 1) + start)

    def hash_otp(self, otp: str) -> str:
        return hashlib.sha256(otp.encode("utf-8")).hexdigest()

    def verify_otp(self, otp: str, stored_hash: str) -> bool:
        return hmac.compare_digest(self.hash_otp(otp), stored_hash)

    def create_otp_record(
        self,
        *,
        user_id: str,
        email: str,
        role: str,
        tenant_id: Optional[str] = None,
        now: Optional[datetime] = None,
        otp: Optional[str] = None,
    ) -> Dict[str, Any]:
        issued_at = now or datetime.utcnow()
        plain_otp = otp or self.generate_otp()
        record = {
            "user_id": user_id,
            "email": email.lower().strip(),
            "role": role,
            "tenant_id": tenant_id,
            "otp_hash": self.hash_otp(plain_otp),
            "created_at": issued_at,
            "expires_at": issued_at + timedelta(minutes=self.expire_minutes),
            "attempts": 0,
            "max_attempts": self.max_attempts,
            "used": False,
            "used_at": None,
        }
        return {"otp": plain_otp, "record": record}

    def validate_record(
        self,
        record: Dict[str, Any],
        otp: str,
        *,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        if not record:
            return False, "not_found"
        if record.get("used"):
            return False, "used"
        if int(record.get("attempts", 0)) >= int(record.get("max_attempts", self.max_attempts)):
            return False, "attempts_exhausted"
        current_time = now or datetime.utcnow()
        expires_at = record.get("expires_at")
        if expires_at and current_time > expires_at:
            return False, "expired"
        if not self.verify_otp(otp, record.get("otp_hash", "")):
            return False, "invalid_otp"
        return True, "valid"
