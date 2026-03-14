"""
Bug #3 — Signout doesn't work: cookie-logout endpoint has broken dependency injection.

get_current_user_dual_auth(request, auth_manager) takes auth_manager as a plain
parameter. When FastAPI resolves it via Depends(), auth_manager has no type
annotation, so FastAPI treats it as a required query parameter (string).

This means /cookie-logout always fails with a 422 or 500, and the frontend
silently catches the error. The user-level revocation (Redis) never fires,
so other clients don't detect the logout.
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_get_current_user_dual_auth_has_injectable_auth_manager():
    """
    Reproduce: get_current_user_dual_auth's `auth_manager` parameter lacks
    a type annotation or Depends() default, making FastAPI treat it as a
    query parameter instead of injecting the AuthManager instance.
    """
    from core.cookie_auth import get_current_user_dual_auth

    sig = inspect.signature(get_current_user_dual_auth)
    params = sig.parameters

    assert "auth_manager" in params, "auth_manager parameter missing entirely"

    am_param = params["auth_manager"]

    # For FastAPI DI to work, auth_manager must either:
    # 1. Have a Depends() default, OR
    # 2. Have a type annotation that FastAPI can resolve
    has_default = am_param.default is not inspect.Parameter.empty
    has_annotation = am_param.annotation is not inspect.Parameter.empty

    assert has_default or has_annotation, (
        "BUG REPRODUCED: get_current_user_dual_auth's `auth_manager` parameter "
        "has no type annotation and no default value. FastAPI will treat it as a "
        "required query parameter (string), not an injected dependency.\n"
        f"Parameter: {am_param}"
    )


def test_cookie_logout_endpoint_resolves_dependencies():
    """
    Reproduce: The /cookie-logout endpoint uses Depends(get_current_user_dual_auth)
    but that function's sub-dependencies can't be resolved by FastAPI.

    We simulate FastAPI's dependency resolution to check if it would fail.
    """
    from core.cookie_auth import get_current_user_dual_auth

    sig = inspect.signature(get_current_user_dual_auth)
    params = sig.parameters

    # Check each parameter can be resolved by FastAPI
    for name, param in params.items():
        if name == "request":
            # FastAPI injects Request automatically
            continue

        # All other params need either:
        # - A Depends() default
        # - A type annotation matching a known injectable
        has_depends = (
            param.default is not inspect.Parameter.empty
            and hasattr(param.default, "dependency")
        )
        has_annotation = param.annotation is not inspect.Parameter.empty

        assert has_depends or has_annotation, (
            f"BUG REPRODUCED: Parameter '{name}' in get_current_user_dual_auth "
            f"cannot be resolved by FastAPI DI. It has no Depends() default "
            f"and no type annotation.\n"
            f"This causes /cookie-logout to fail silently."
        )
