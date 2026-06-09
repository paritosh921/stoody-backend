import jwt

from services.online_class.jitsi_provider import JitsiProviderService


def _make_provider(
    domain="meet.jit.si",
    base_url="",
    jwt_enabled="false",
    jwt_secret="",
    jwt_app_id="stoody",
    jwt_audience="jitsi",
    jwt_ttl="7200",
):
    env = {
        "ONLINE_CLASS_JITSI_DOMAIN": domain,
        "ONLINE_CLASS_JITSI_BASE_URL": base_url,
        "ONLINE_CLASS_JITSI_JWT_ENABLED": jwt_enabled,
        "ONLINE_CLASS_JITSI_JWT_SECRET": jwt_secret,
        "ONLINE_CLASS_JITSI_JWT_APP_ID": jwt_app_id,
        "ONLINE_CLASS_JITSI_JWT_AUDIENCE": jwt_audience,
        "ONLINE_CLASS_JITSI_JWT_TTL_SECONDS": jwt_ttl,
    }
    import os
    saved = {}
    for key, val in env.items():
        saved[key] = os.environ.get(key)
        if val is not None:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]
    try:
        return JitsiProviderService()
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


TEST_SECRET = "test-secret-key-min-32-chars-long!!"


class TestJwtDisabled:
    def test_no_token_when_jwt_disabled(self):
        provider = _make_provider(jwt_enabled="false", jwt_secret=TEST_SECRET)
        details = provider.get_provider_details(
            "MTG-001", user_id="u1", user_name="Alice", moderator=True
        )
        assert details["token_required"] is False
        assert details["token"] is None
        assert details["configured"] is True
        assert details["room_name"] == "stoody-MTG-001"

    def test_existing_test_still_passes_domain_from_base_url(self):
        provider = _make_provider(
            domain="",
            base_url="https://class.stoody.in",
            jwt_enabled="true",
            jwt_secret="",
        )
        details = provider.get_provider_details("MTG 123")
        assert details["domain"] == "class.stoody.in"
        assert details["room_name"] == "stoody-MTG-123"
        assert details["url"] == ""
        assert details["token_required"] is True
        assert details["token"] is None
        assert details["configured"] is False


class TestJwtEnabledWithSecret:
    def test_generates_valid_jwt_with_claims(self):
        provider = _make_provider(jwt_enabled="true", jwt_secret=TEST_SECRET)
        details = provider.get_provider_details(
            "MTG-XYZ",
            user_id="tutor-42",
            user_name="Prof Smith",
            user_email="smith@school.edu",
            moderator=True,
        )
        assert details["token_required"] is True
        assert details["token"] is not None

        decoded = jwt.decode(
            details["token"],
            TEST_SECRET,
            algorithms=["HS256"],
            audience="jitsi",
            issuer="stoody",
        )

        assert decoded["aud"] == "jitsi"
        assert decoded["iss"] == "stoody"
        assert decoded["sub"] == "meet.jit.si"
        assert decoded["room"] == "stoody-MTG-XYZ"
        assert "nbf" in decoded
        assert "exp" in decoded
        assert decoded["context"]["user"]["id"] == "tutor-42"
        assert decoded["context"]["user"]["name"] == "Prof Smith"
        assert decoded["context"]["user"]["email"] == "smith@school.edu"
        assert decoded["context"]["user"]["moderator"] is True

    def test_student_gets_moderator_false(self):
        provider = _make_provider(jwt_enabled="true", jwt_secret=TEST_SECRET)
        details = provider.get_provider_details(
            "MTG-ABC",
            user_id="stu-1",
            user_name="Alice",
            moderator=False,
        )
        decoded = jwt.decode(
            details["token"],
            TEST_SECRET,
            algorithms=["HS256"],
            audience="jitsi",
        )
        assert decoded["context"]["user"]["moderator"] is False

    def test_custom_app_id_and_audience(self):
        provider = _make_provider(
            jwt_enabled="true",
            jwt_secret=TEST_SECRET,
            jwt_app_id="my-app",
            jwt_audience="custom-aud",
        )
        details = provider.get_provider_details(
            "MTG-CUSTOM", user_id="u1", user_name="User"
        )
        decoded = jwt.decode(
            details["token"],
            TEST_SECRET,
            algorithms=["HS256"],
            audience="custom-aud",
        )
        assert decoded["iss"] == "my-app"
        assert decoded["aud"] == "custom-aud"

    def test_ttl_is_respected(self):
        provider = _make_provider(
            jwt_enabled="true", jwt_secret=TEST_SECRET, jwt_ttl="3600"
        )
        details = provider.get_provider_details(
            "MTG-TTL", user_id="u1", user_name="User"
        )
        decoded = jwt.decode(
            details["token"],
            TEST_SECRET,
            algorithms=["HS256"],
            audience="jitsi",
        )
        assert decoded["exp"] - decoded["nbf"] == 3600


class TestJwtEnabledWithoutSecret:
    def test_no_token_emitted_when_secret_absent(self):
        provider = _make_provider(
            jwt_enabled="true",
            jwt_secret="",
        )
        details = provider.get_provider_details(
            "MTG-NOSEC", user_id="u1", user_name="User"
        )
        assert details["token_required"] is True
        assert details["token"] is None
        assert details["configured"] is False

    def test_jwt_available_property_false_without_secret(self):
        provider = _make_provider(jwt_enabled="true", jwt_secret="")
        assert provider.jwt_available is False
        assert provider.missing_required_jwt_secret is True


class TestRoomNameStability:
    def test_room_name_is_deterministic(self):
        provider = _make_provider(jwt_enabled="false")
        name1 = provider.generate_room_name("meeting-42")
        name2 = provider.generate_room_name("meeting-42")
        assert name1 == name2

    def test_special_characters_sanitized(self):
        provider = _make_provider(jwt_enabled="false")
        name = provider.generate_room_name("abc/def hij!123")
        assert name == "stoody-abc-def-hij-123"
        assert all(c.isalnum() or c == "-" for c in name.replace("stoody-", "", 1))

    def test_room_name_matches_jwt_room_claim(self):
        provider = _make_provider(jwt_enabled="true", jwt_secret=TEST_SECRET)
        meeting_id = "MTG-ROOM-01"
        details = provider.get_provider_details(
            meeting_id, user_id="u1", user_name="User"
        )
        decoded = jwt.decode(
            details["token"],
            TEST_SECRET,
            algorithms=["HS256"],
            audience="jitsi",
        )
        assert decoded["room"] == details["room_name"]

    def test_canvas_room_names_are_deterministic_and_scoped(self):
        provider = _make_provider(jwt_enabled="false")
        teacher_room = provider.generate_canvas_room_name("MTG 123", "teacher")
        student_room = provider.generate_canvas_room_name("MTG 123", "student", "STU/A 1")

        assert teacher_room == "stoody-mtg-123-canvas-teacher"
        assert student_room == "stoody-mtg-123-canvas-student-stu-a-1"
        assert teacher_room != student_room

    def test_canvas_room_jwt_uses_exact_canvas_room_claim(self):
        provider = _make_provider(jwt_enabled="true", jwt_secret=TEST_SECRET)
        room_name = provider.generate_canvas_room_name("MTG-ROOM-01", "student", "STU-1")
        details = provider.get_provider_details_for_room(
            room_name,
            user_id="STU-1",
            user_name="Student One",
            moderator=False,
        )

        decoded = jwt.decode(
            details["token"],
            TEST_SECRET,
            algorithms=["HS256"],
            audience="jitsi",
        )
        assert details["room_name"] == room_name
        assert details["room_name"] == details["room_name"].lower()
        assert decoded["room"] == room_name
        assert decoded["context"]["user"]["id"] == "STU-1"
        assert decoded["context"]["user"]["moderator"] is False


class TestBuildProviderDetailsIntegration:
    def test_build_provider_details_passes_user_info(self):
        from api.v1.meeting_async import _build_provider_details

        import os
        saved = {}
        env_vars = {
            "ONLINE_CLASS_JITSI_DOMAIN": "meet.jit.si",
            "ONLINE_CLASS_JITSI_JWT_ENABLED": "true",
            "ONLINE_CLASS_JITSI_JWT_SECRET": TEST_SECRET,
            "ONLINE_CLASS_JITSI_BASE_URL": "",
        }
        for key, val in env_vars.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = val
        try:
            from services.online_class.jitsi_provider import JitsiProviderService
            import api.v1.meeting_async as meeting_module
            original = meeting_module.jitsi_provider_service
            meeting_module.jitsi_provider_service = JitsiProviderService()
            try:
                current_user = {
                    "user_type": "tutor",
                    "tutor_id": "t-1",
                    "name": "Prof Test",
                    "email": "prof@test.edu",
                }
                pd = _build_provider_details(
                    "MTG-INT", current_user=current_user, moderator=True
                )
                assert pd.token is not None
                assert pd.configured is True
                decoded = jwt.decode(
                    pd.token,
                    TEST_SECRET,
                    algorithms=["HS256"],
                    audience="jitsi",
                )
                assert decoded["context"]["user"]["id"] == "t-1"
                assert decoded["context"]["user"]["name"] == "Prof Test"
                assert decoded["context"]["user"]["email"] == "prof@test.edu"
                assert decoded["context"]["user"]["moderator"] is True
            finally:
                meeting_module.jitsi_provider_service = original
        finally:
            for key, old in saved.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

    def test_build_provider_details_no_user_no_crash(self):
        from api.v1.meeting_async import _build_provider_details

        import os
        saved = {}
        env_vars = {
            "ONLINE_CLASS_JITSI_DOMAIN": "meet.jit.si",
            "ONLINE_CLASS_JITSI_JWT_ENABLED": "true",
            "ONLINE_CLASS_JITSI_JWT_SECRET": TEST_SECRET,
            "ONLINE_CLASS_JITSI_BASE_URL": "",
        }
        for key, val in env_vars.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = val
        try:
            from services.online_class.jitsi_provider import JitsiProviderService
            import api.v1.meeting_async as meeting_module
            original = meeting_module.jitsi_provider_service
            meeting_module.jitsi_provider_service = JitsiProviderService()
            try:
                pd = _build_provider_details("MTG-NOUSER")
                assert pd.token is None
                assert pd.token_required is True
                assert pd.configured is False
            finally:
                meeting_module.jitsi_provider_service = original
        finally:
            for key, old in saved.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
