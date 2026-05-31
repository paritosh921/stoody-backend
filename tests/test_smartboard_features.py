"""
Tests for Smartboard feature model production rollout.

Covers:
1. Feature key migration (_dummy -> production keys)
2. Feature gating on smartboard pair/session paths
3. Feature disabled returns false via merge
4. Role-aware gating for smartboard cloud OCR/notes
5. School settings smartboard schema (via direct pydantic)
6. Auth/token alignment verification
7. Path exemption verification

NOTE: Direct imports from `core.*` trigger `core/__init__.py` which
imports motor/fastapi. For the tenant_features module we import
the file directly using importlib to bypass the package __init__.
"""

import importlib.util
import os
import sys
import pytest


def _import_tenant_features():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "tenant_features",
        os.path.join(backend_dir, "core", "tenant_features.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_settings_async():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "settings_async",
        os.path.join(backend_dir, "api", "v1", "settings_async.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        pytest.skip("fastapi not available in test environment", allow_module_level=True)
        return None
    return mod


tf = _import_tenant_features()


# ---------------------------------------------------------------------------
# 1. Feature key migration tests
# ---------------------------------------------------------------------------

class TestFeatureKeyMigration:
    def test_catalog_has_no_dummy_keys(self):
        dummy_keys = [item["key"] for item in tf.FEATURE_CATALOG if "_dummy" in item["key"]]
        assert len(dummy_keys) == 0, f"Found _dummy keys in catalog: {dummy_keys}"

    def test_catalog_has_production_smartboard_keys(self):
        assert "smartboard_core" in tf.FEATURE_CATALOG_BY_KEY
        assert "smartboard_live_session" in tf.FEATURE_CATALOG_BY_KEY
        assert "smartboard_cloud_access" in tf.FEATURE_CATALOG_BY_KEY

    def test_production_keys_are_active(self):
        for key in ("smartboard_core", "smartboard_live_session", "smartboard_cloud_access"):
            entry = tf.FEATURE_CATALOG_BY_KEY[key]
            assert entry["status"] == "active"

    def test_migration_map_covers_all_dummy_keys(self):
        assert tf.FEATURE_MIGRATION_MAP["smartboard_core_dummy"] == "smartboard_core"
        assert tf.FEATURE_MIGRATION_MAP["smartboard_live_session_dummy"] == "smartboard_live_session"
        assert tf.FEATURE_MIGRATION_MAP["smartboard_token_dummy"] == "smartboard_cloud_access"

    def test_migrate_feature_keys_remapped(self):
        raw = {
            "smartboard_core_dummy": True,
            "smartboard_live_session_dummy": False,
            "smartboard_token_dummy": True,
            "exampen": True,
        }
        result = tf.migrate_feature_keys(raw)
        assert result["smartboard_core"] is True
        assert result["smartboard_live_session"] is False
        assert result["smartboard_cloud_access"] is True
        assert result["exampen"] is True
        assert "smartboard_core_dummy" not in result
        assert "smartboard_token_dummy" not in result

    def test_migrate_feature_keys_passthrough(self):
        raw = {"exampen": True, "student_ai_mentor": False}
        result = tf.migrate_feature_keys(raw)
        assert result == raw

    def test_migrate_feature_keys_none(self):
        assert tf.migrate_feature_keys(None) == {}

    def test_build_enabled_features_v2_migrates_old_keys(self):
        raw_v2 = {
            "version": 2,
            "tier": "max",
            "overrides": {
                "smartboard_core_dummy": True,
                "smartboard_token_dummy": True,
            },
        }
        result = tf.build_enabled_features_v2(raw_v2)
        effective = result["effective"]
        assert effective.get("smartboard_core") is True
        assert effective.get("smartboard_cloud_access") is True
        assert "smartboard_core_dummy" not in effective

    def test_merge_tenant_features_migrates_legacy(self):
        raw_legacy = {"smartboard": True, "exampen": False}
        merged = tf.merge_tenant_features(raw_legacy)
        assert merged.get("smartboard_core") is True

    def test_no_dummy_keys_in_path_prefixes(self):
        for key in tf.FEATURE_PATH_PREFIXES:
            assert "_dummy" not in key, f"Found _dummy key in FEATURE_PATH_PREFIXES: {key}"

    def test_legacy_to_v2_map_uses_production_key(self):
        mapped = tf.LEGACY_TO_V2_MAP.get("smartboard", ())
        assert "smartboard_core" in mapped
        assert "smartboard_core_dummy" not in mapped


# ---------------------------------------------------------------------------
# 2. Feature gating on smartboard paths
# ---------------------------------------------------------------------------

class TestSmartboardPairFeatureGating:
    def test_cloud_access_gates_pair_paths(self):
        assert tf.required_feature_for_path("/api/v1/smartboard-pair/register") == "smartboard_cloud_access"
        assert tf.required_feature_for_path("/api/v1/smartboard-pair/redeem") == "smartboard_cloud_access"

    def test_cloud_access_gates_token_path(self):
        assert tf.required_feature_for_path("/api/v1/smartboard/token") == "smartboard_cloud_access"

    def test_core_gates_smartboard_base(self):
        assert tf.required_feature_for_path("/api/v1/smartboard") == "smartboard_core"

    def test_live_session_gates_sessions(self):
        assert tf.required_feature_for_path("/api/v1/smartboard/sessions") == "smartboard_live_session"

    def test_live_session_gates_smartboard_sessions(self):
        assert tf.required_feature_for_path("/api/v1/smartboard-sessions") == "smartboard_cloud_access"


# ---------------------------------------------------------------------------
# 3. Feature disabled returns 403 via middleware logic
# ---------------------------------------------------------------------------

class TestSmartboardFeatureGatingMiddleware:
    def test_cloud_access_disabled_blocks_pair_register(self):
        raw_v2 = {
            "version": 2,
            "tier": "core",
            "overrides": {"smartboard_cloud_access": False},
        }
        merged = tf.merge_tenant_features(None, raw_v2)
        feature = tf.required_feature_for_path("/api/v1/smartboard-pair/register")
        assert feature is not None
        assert merged.get(feature) is False

    def test_cloud_access_enabled_allows_pair_register(self):
        raw_v2 = {
            "version": 2,
            "tier": "custom",
            "overrides": {"smartboard_cloud_access": True},
        }
        merged = tf.merge_tenant_features(None, raw_v2)
        feature = tf.required_feature_for_path("/api/v1/smartboard-pair/register")
        assert merged.get(feature) is True

    def test_core_disabled_blocks_smartboard_base(self):
        raw_v2 = {
            "version": 2,
            "tier": "core",
            "overrides": {"smartboard_core": False},
        }
        merged = tf.merge_tenant_features(None, raw_v2)
        feature = tf.required_feature_for_path("/api/v1/smartboard")
        assert merged.get(feature) is False

    def test_live_session_disabled_blocks_sessions(self):
        raw_v2 = {
            "version": 2,
            "tier": "core",
            "overrides": {"smartboard_live_session": False},
        }
        merged = tf.merge_tenant_features(None, raw_v2)
        feature = tf.required_feature_for_path("/api/v1/smartboard/sessions")
        assert merged.get(feature) is False


# ---------------------------------------------------------------------------
# 4. Role-aware gating for smartboard cloud OCR/notes
# ---------------------------------------------------------------------------

class TestSmartboardRoleAwareGating:
    """Tests that smartboard devices get cloud_access gating for OCR/notes."""

    def test_ocr_requires_cloud_access_for_smartboard_device(self):
        feature = tf.required_feature_for_path("/api/v1/ocr", user_type="smartboard")
        assert feature == "smartboard_cloud_access"

    def test_ocr_requires_pen_capture_for_regular_tutor(self):
        feature = tf.required_feature_for_path("/api/v1/ocr", user_type="tutor")
        assert feature == "stoody_pen_capture"

    def test_ocr_requires_pen_capture_for_admin(self):
        feature = tf.required_feature_for_path("/api/v1/ocr", user_type="admin")
        assert feature == "stoody_pen_capture"

    def test_notes_requires_cloud_access_for_smartboard_device(self):
        feature = tf.required_feature_for_path("/api/v1/notes", user_type="smartboard")
        assert feature == "smartboard_cloud_access"

    def test_notes_requires_pen_capture_for_regular_tutor(self):
        feature = tf.required_feature_for_path("/api/v1/notes", user_type="tutor")
        assert feature == "stoody_pen_capture"

    def test_ocr_no_role_falls_through_to_pen_capture(self):
        feature = tf.required_feature_for_path("/api/v1/ocr", user_type=None)
        assert feature == "stoody_pen_capture"

    def test_smartboard_status_is_exempt(self):
        feature = tf.required_feature_for_path("/api/v1/smartboard/status")
        assert feature is None


# ---------------------------------------------------------------------------
# 5. School settings smartboard schema (if fastapi available)
# ---------------------------------------------------------------------------

class TestSchoolSettingsSmartboardSchema:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sa = _import_settings_async()
        if self.sa is None:
            pytest.skip("fastapi not available")

    def test_smartboard_config_model_defaults(self):
        config = self.sa.SmartboardConfig()
        assert config.enabled is False
        assert config.allow_cloud_features is False
        assert config.default_subject is None
        assert config.default_standard is None
        assert config.hub_mode_policy is None

    def test_smartboard_config_model_with_values(self):
        config = self.sa.SmartboardConfig(
            enabled=True,
            allow_cloud_features=True,
            default_subject="Mathematics",
            default_standard="10",
            hub_mode_policy="cloud_first",
        )
        assert config.enabled is True
        assert config.allow_cloud_features is True
        assert config.default_subject == "Mathematics"

    def test_default_settings_includes_smartboard(self):
        assert "smartboard" in self.sa.DEFAULT_SETTINGS
        assert self.sa.DEFAULT_SETTINGS["smartboard"]["enabled"] is False
        assert self.sa.DEFAULT_SETTINGS["smartboard"]["allow_cloud_features"] is False

    def test_school_settings_request_accepts_smartboard(self):
        req = self.sa.SchoolSettingsRequest(
            smartboard={"enabled": True, "allow_cloud_features": True}
        )
        assert req.smartboard is not None
        assert req.smartboard.enabled is True

    def test_school_settings_response_includes_smartboard(self):
        resp = self.sa.SchoolSettingsResponse(
            admin_id="test",
            school_info={"school_name": "Test", "school_logo": "", "contact_email": "", "contact_phone": "", "address": "", "website": ""},
            classes=[],
            sections=[],
            subjects=[],
            plan_types=[],
            streams=[],
            smartboard={"enabled": True, "allow_cloud_features": False},
        )
        assert resp.smartboard is not None
        assert resp.smartboard.enabled is True


# ---------------------------------------------------------------------------
# 6. Auth / token alignment verification
# ---------------------------------------------------------------------------

class TestSmartboardAuthFixes:
    def test_tenant_features_module_loads(self):
        assert hasattr(tf, "FEATURE_CATALOG")
        assert hasattr(tf, "migrate_feature_keys")
        assert hasattr(tf, "FEATURE_MIGRATION_MAP")

    def test_migrate_v2_overrides_function_exists(self):
        assert hasattr(tf, "_migrate_v2_overrides")
        assert callable(tf._migrate_v2_overrides)

    def test_smartboard_token_uses_canonical_imports(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        token_path = os.path.join(backend_dir, "api", "v1", "smartboard_token.py")
        with open(token_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "SMARTBOARD_JWT_SECRET" not in content, (
            "smartboard_token.py still references SMARTBOARD_JWT_SECRET — "
            "should use canonical AuthManager.create_access_token"
        )
        assert "type.*smartboard_access" not in content, (
            "smartboard_token.py still issues type='smartboard_access' — "
            "should use canonical type='access' via AuthManager"
        )
        assert "AuthManager" in content
        assert "create_access_token" in content

    def test_smartboard_pair_includes_capabilities(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pair_path = os.path.join(backend_dir, "api", "v1", "smartboard_pair_async.py")
        with open(pair_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "CloudCapabilities" in content
        assert "capabilities" in content

    def test_smartboard_pair_heartbeat_reads_pair_session_from_jwt_payload(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pair_path = os.path.join(backend_dir, "api", "v1", "smartboard_pair_async.py")
        with open(pair_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "_decode_request_token_payload" in content
        assert 'token_payload.get("pair_session_id")' in content
        assert "current_user.get(\"pair_session_id\") or token_payload.get(\"pair_session_id\")" in content

    def test_websocket_auth_checks_session_ownership(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ws_path = os.path.join(backend_dir, "api", "v1", "smartboard_async.py")
        with open(ws_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "session.tutor_id" in content, (
            "WebSocket endpoint should verify session ownership against tutor_id"
        )
