from datetime import datetime

from core.ai_usage.metrics_exporter import public_identity_ref
from core.ai_usage.user_lookup import build_user_ref_lookup_response


def test_user_ref_lookup_matches_hashed_user_and_summarizes_usage():
    target_ref = public_identity_ref("student-1", prefix="user")
    response = build_user_ref_lookup_response(
        target_ref,
        events=[
            {
                "user_id": "student-1",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "stage": "stoody_book",
                "status": "success",
                "actual_input_tokens": 10,
                "actual_output_tokens": 15,
                "estimated_total_tokens": 99,
                "created_at": datetime(2026, 6, 21),
            },
            {
                "user_id": "student-1",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "stage": "question_structuring",
                "status": "success",
                "estimated_total_tokens": 40,
                "created_at": datetime(2026, 6, 21),
            },
            {
                "user_id": "student-2",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "stage": "stoody_book",
                "status": "success",
                "estimated_total_tokens": 1000,
                "created_at": datetime(2026, 6, 21),
            },
        ],
        profiles={
            "student-1": {
                "role": "student",
                "name": "Anika Student",
                "username": "anika",
                "email": "anika@example.com",
            }
        },
    )

    assert response["found"] is True
    assert response["user_ref"] == target_ref
    assert response["matches"][0]["user_id"] == "student-1"
    assert response["matches"][0]["profile"]["name"] == "Anika Student"
    assert response["summary"]["total_tokens"] == 65
    assert response["summary"]["calls"] == 2
    assert response["summary"]["models"]["gpt-4o-mini"]["total_tokens"] == 25
    assert response["summary"]["models"]["openai/gpt-oss-120b"]["total_tokens"] == 40
    assert "student-2" not in repr(response)


def test_user_ref_lookup_returns_not_found_for_unknown_ref():
    response = build_user_ref_lookup_response(
        "user_doesnotexist",
        events=[{"user_id": "student-1", "estimated_total_tokens": 10}],
        profiles={},
    )

    assert response == {
        "found": False,
        "user_ref": "user_doesnotexist",
        "matches": [],
        "summary": {
            "calls": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "models": {},
            "providers": {},
            "stages": {},
        },
    }
