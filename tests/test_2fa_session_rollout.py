from __future__ import annotations

import asyncio
from pathlib import Path

from core.auth import AuthManager


class _SystemConfigCollection:
    def __init__(self, value: float | None):
        self.value = value

    async def find_one(self, query):
        assert query == {"key": "min_token_issued_at"}
        if self.value is None:
            return None
        return {"key": "min_token_issued_at", "value": self.value}


class _MasterDb(dict):
    def __init__(self, min_token_issued_at: float | None):
        super().__init__()
        self["system_config"] = _SystemConfigCollection(min_token_issued_at)


class _DbManager:
    def __init__(self, min_token_issued_at: float | None):
        self.master_db = _MasterDb(min_token_issued_at)

    async def get_master_db(self):
        return self.master_db


def test_admin_tutor_rollout_cutoff_rejects_pre_existing_tokens():
    asyncio.run(_admin_tutor_rollout_cutoff_rejects_pre_existing_tokens())


async def _admin_tutor_rollout_cutoff_rejects_pre_existing_tokens():
    auth_manager = AuthManager()
    token = auth_manager.create_access_token(
        {
            "sub": "admin-1",
            "user_type": "admin",
            "db_name": "skb_abcd_1234",
            "tenant_id": "ABCD-1234",
        }
    )
    payload = auth_manager.decode_access_token(token)
    assert payload and payload["iat"]

    auth_manager.set_db_manager(_DbManager(float(payload["iat"]) + 1))

    assert await auth_manager.verify_token_and_get_user(token) is None


def test_prod_backend_deploy_invalidates_existing_sessions():
    workflow = Path(".github/workflows/deploy-prod-backend.yml").read_text()

    assert "PROD_DEPLOY_SESSION_SECRET" in workflow
    assert "/api/v1/auth/invalidate-all-sessions" in workflow
    assert "X-Deploy-Secret" in workflow
