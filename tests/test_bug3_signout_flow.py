"""
Bug #3 — Desktop app sign-out does not sign out web portal.

The flow:
  1. Desktop agent calls POST /auth/remote-logout with a user JWT.
  2. Backend sets user-level revocation in Redis for all known user IDs.
  3. Portal periodically calls GET /auth/verify with its own JWT.
  4. Backend checks user-level revocation → returns 401 → portal auto-logs out.

Root causes being tested:
  A. Token user_id resolution: the agent's JWT has user_id=username,
     but the portal JWT has sub=ObjectId.  The remote-logout must resolve
     username → ObjectId via DB lookup.  If db_name is missing from the
     token, the lookup is skipped and only the username is revoked,
     which doesn't match what /verify checks.
  B. Redis availability: revoke_user_session() silently degrades to
     no-op when cache_manager is None.
  C. Per-worker in-memory blacklist: token_blacklist.is_revoked() only
     works on the worker that processed the logout request; the Redis-
     backed is_user_session_revoked() is the cross-worker mechanism.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── A. Verify token decoding extracts user_id correctly ────────────

def test_remote_logout_extracts_user_id_from_payload():
    """
    The remote-logout code does:
        user_id = payload.get("user_id") or payload.get("sub")

    The agent's user token has:
        {"user_id": username, "sub": ObjectId, ...}

    So user_id = username (not ObjectId).  This is intentional — the code
    then does a DB lookup to resolve username → ObjectId.
    """
    # Simulate agent's user token payload
    payload = {
        "sub": "65abc123def456789012abcd",  # ObjectId
        "user_id": "john_doe",               # username
        "username": "john_doe",
        "db_name": "skb_indl-ciel-1001",
        "type": "access",
    }

    user_id = payload.get("user_id") or payload.get("sub")
    assert user_id == "john_doe", (
        f"Expected username 'john_doe', got '{user_id}'"
    )


def test_remote_logout_collects_all_id_variants():
    """
    The remote-logout builds ids_to_revoke from:
      1. user_id (from token — username)
      2. username (from token — same as user_id usually)
      3. ObjectId (from DB lookup)

    All must be revoked so the portal's verify (which checks ObjectId)
    finds the revocation flag.
    """
    # Simulate what remote-logout does
    payload = {
        "sub": "65abc123def456789012abcd",
        "user_id": "john_doe",
        "username": "john_doe",
        "db_name": "skb_indl-ciel-1001",
    }

    user_id = payload.get("user_id") or payload.get("sub")
    ids_to_revoke = set()
    ids_to_revoke.add(user_id)

    token_username = payload.get("username")
    if token_username:
        ids_to_revoke.add(token_username)

    # Simulate DB lookup finding the ObjectId
    student_oid = "65abc123def456789012abcd"
    ids_to_revoke.add(student_oid)

    assert "john_doe" in ids_to_revoke
    assert "65abc123def456789012abcd" in ids_to_revoke
    assert len(ids_to_revoke) == 2  # username + ObjectId


def test_remote_logout_without_db_name_misses_objectid():
    """
    BUG SCENARIO: If the agent's token doesn't have db_name, the DB
    lookup is skipped, and only the username is revoked.  The portal's
    verify checks by ObjectId, so it won't find the revocation.
    """
    payload = {
        "sub": "65abc123def456789012abcd",
        "user_id": "john_doe",
        "username": "john_doe",
        "db_name": None,  # Missing!
    }

    user_id = payload.get("user_id") or payload.get("sub")
    ids_to_revoke = set()
    ids_to_revoke.add(user_id)

    token_username = payload.get("username")
    if token_username:
        ids_to_revoke.add(token_username)

    db_name = payload.get("db_name")
    if db_name:
        # This won't execute because db_name is None
        ids_to_revoke.add("65abc123def456789012abcd")

    # Only username is revoked, NOT the ObjectId
    assert "john_doe" in ids_to_revoke
    assert "65abc123def456789012abcd" not in ids_to_revoke, (
        "Without db_name, the ObjectId should NOT be in ids_to_revoke — "
        "this means the portal's verify won't detect the revocation."
    )


# ── B. Verify Redis-backed revocation functions ────────────────────

def test_revoke_user_session_skips_without_cache_manager():
    """
    If cache_manager is None, revoke_user_session() silently skips.
    This means the logout has no cross-worker effect.
    """
    import asyncio
    from core.token_blacklist import revoke_user_session

    async def _test():
        # Should not raise — just silently skip
        await revoke_user_session(None, "some_user_id")

    asyncio.run(_test())
    # If we get here, it didn't crash — but it also didn't revoke.
    # This is the "silent degradation" that makes debugging hard.


def test_is_user_session_revoked_returns_false_without_cache_manager():
    """
    If cache_manager is None, is_user_session_revoked() returns False,
    allowing the session through even if it was revoked.
    """
    import asyncio
    from core.token_blacklist import is_user_session_revoked

    async def _test():
        result = await is_user_session_revoked(None, "some_user_id")
        assert result is False, (
            "With no cache_manager, should return False (allow session)"
        )

    asyncio.run(_test())


# ── C. Verify in-memory blacklist is per-instance ──────────────────

def test_token_blacklist_is_per_instance():
    """
    The in-memory TokenBlacklist is per-worker.  Two separate instances
    don't share state.  This confirms that token-level revocation can't
    work across 8 Uvicorn workers — only Redis-backed user-level works.
    """
    from core.token_blacklist import TokenBlacklist

    bl1 = TokenBlacklist()
    bl2 = TokenBlacklist()

    bl1.revoke("token_abc")

    assert bl1.is_revoked("token_abc") is True
    assert bl2.is_revoked("token_abc") is False, (
        "Different blacklist instance should NOT see revoked token — "
        "confirms per-worker isolation."
    )


# ── D. Verify the verify endpoint uses get_current_user_dual_auth ──

def test_verify_endpoint_uses_dual_auth():
    """
    The /auth/verify endpoint must call get_current_user_dual_auth,
    which checks both cookie and Bearer paths and includes user-level
    revocation checks in both paths.
    """
    verify_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "v1", "auth_async.py"
    )
    verify_path = os.path.normpath(verify_path)

    with open(verify_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Find the verify_token function
    func_start = source.find('async def verify_token(')
    assert func_start != -1, "verify_token function not found"

    # Find the next function after it
    func_end = source.find('\nasync def ', func_start + 10)
    if func_end == -1:
        func_end = source.find('\n@router.', func_start + 100)
    if func_end == -1:
        func_end = len(source)

    func_body = source[func_start:func_end]

    assert 'get_current_user_dual_auth' in func_body, (
        "verify_token must use get_current_user_dual_auth for revocation checks"
    )


# ── E. Verify dual auth checks user-level revocation in Bearer path ─

def test_dual_auth_checks_user_revocation_in_bearer_path():
    """
    get_current_user_dual_auth's Bearer fallback path must call
    is_user_session_revoked() — this is the cross-worker mechanism.
    """
    cookie_auth_path = os.path.join(
        os.path.dirname(__file__), "..", "core", "cookie_auth.py"
    )
    cookie_auth_path = os.path.normpath(cookie_auth_path)

    with open(cookie_auth_path, "r", encoding="utf-8") as f:
        source = f.read()

    func_start = source.find('async def get_current_user_dual_auth')
    assert func_start != -1

    func_end = source.find('\ndef ', func_start + 10)
    if func_end == -1:
        func_end = len(source)

    func_body = source[func_start:func_end]

    assert 'is_user_session_revoked' in func_body, (
        "get_current_user_dual_auth must check is_user_session_revoked "
        "for Redis-backed cross-worker revocation"
    )


# ── F. Verify remote-logout endpoint exists and revokes all IDs ─────

def test_remote_logout_endpoint_exists():
    """
    /auth/remote-logout must exist — this is what the desktop agent calls.
    """
    auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "v1", "auth_async.py"
    )
    auth_path = os.path.normpath(auth_path)

    with open(auth_path, "r", encoding="utf-8") as f:
        source = f.read()

    assert '"/remote-logout"' in source, (
        "BUG: /auth/remote-logout endpoint missing from auth_async.py. "
        "Desktop agent logout will silently fail."
    )


def test_remote_logout_revokes_user_session():
    """
    The remote-logout endpoint must call revoke_user_session() for
    cross-worker revocation via Redis.
    """
    auth_path = os.path.join(
        os.path.dirname(__file__), "..", "api", "v1", "auth_async.py"
    )
    auth_path = os.path.normpath(auth_path)

    with open(auth_path, "r", encoding="utf-8") as f:
        source = f.read()

    func_start = source.find('async def remote_logout(')
    assert func_start != -1

    func_end = source.find('\n@router.', func_start + 10)
    if func_end == -1:
        func_end = len(source)

    func_body = source[func_start:func_end]

    assert 'revoke_user_session' in func_body, (
        "BUG: remote_logout must call revoke_user_session for Redis-backed "
        "cross-worker revocation"
    )


# ── G. Frontend periodic verify check ──────────────────────────────

def test_frontend_has_periodic_token_check():
    """
    AuthContext.tsx must have a periodic token validation check that
    calls /auth/verify and auto-logs-out on 401.
    """
    auth_ctx_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "frontend", "src", "contexts", "AuthContext.tsx",
    )
    auth_ctx_path = os.path.normpath(auth_ctx_path)

    assert os.path.exists(auth_ctx_path), f"File not found: {auth_ctx_path}"

    with open(auth_ctx_path, "r", encoding="utf-8") as f:
        source = f.read()

    assert '/auth/verify' in source, (
        "AuthContext.tsx must periodically call /auth/verify"
    )
    assert 'setInterval' in source, (
        "AuthContext.tsx must use setInterval for periodic token checks"
    )
    assert '401' in source, (
        "AuthContext.tsx must check for 401 status to trigger auto-logout"
    )


# ── H. Desktop agent actually calls remote-logout ───────────────────

def test_desktop_agent_calls_remote_logout():
    """
    The desktop agent's logout() must call /auth/remote-logout
    (not /auth/logout, which requires a different auth scheme).
    """
    client_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "stoody-ble-agent", "agent", "src",
        "stoody_agent", "ui_client.py",
    )
    client_path = os.path.normpath(client_path)

    assert os.path.exists(client_path), f"File not found: {client_path}"

    with open(client_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Find the logout function
    func_start = source.find('def logout(self)')
    assert func_start != -1, "logout function not found in ui_client.py"

    func_end = source.find('\n    def ', func_start + 10)
    if func_end == -1:
        func_end = len(source)

    func_body = source[func_start:func_end]

    assert 'remote-logout' in func_body, (
        "Desktop agent must call /auth/remote-logout for cross-client logout"
    )
    assert 'fetch_user_token' in func_body, (
        "Desktop agent must fetch user token before revoking"
    )


# ── I. Agent user-token endpoint includes db_name ───────────────────

def test_agent_user_token_includes_db_name():
    """
    The pen backend's /user-token endpoint must include db_name in the
    JWT payload.  Without it, the remote-logout can't resolve the
    username → ObjectId mapping.
    """
    routes_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "stoody-ble-agent", "server", "api", "agent_routes.py",
    )
    routes_path = os.path.normpath(routes_path)

    assert os.path.exists(routes_path), f"File not found: {routes_path}"

    with open(routes_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Find the issue_user_token function
    func_start = source.find('async def issue_user_token(')
    assert func_start != -1, "issue_user_token not found in agent_routes.py"

    func_end = source.find('\n@router.', func_start + 10)
    if func_end == -1:
        func_end = source.find('\nasync def ', func_start + 10)
    if func_end == -1:
        func_end = len(source)

    func_body = source[func_start:func_end]

    assert '"db_name"' in func_body or "'db_name'" in func_body, (
        "BUG: user-token JWT must include db_name for remote-logout "
        "to resolve username → ObjectId"
    )
