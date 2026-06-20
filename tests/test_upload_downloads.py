import pytest
from fastapi import HTTPException

from api.v1.upload_downloads_async import _authorize_download


def test_download_denies_unknown_purpose_even_for_same_tenant_owner():
    verdict = {
        "tenant_db": "skb_ciel",
        "user_id": "admin-1",
        "purpose_metadata": {
            "purpose": "unregistered_upload_purpose",
            "created_by": "admin-1",
        },
    }
    current_user = {"db_name": "skb_ciel", "user_id": "admin-1"}

    with pytest.raises(HTTPException) as exc:
        _authorize_download(verdict, current_user)

    assert exc.value.status_code == 403


def test_download_denies_same_tenant_non_owner_for_known_purpose():
    verdict = {
        "tenant_db": "skb_ciel",
        "user_id": "admin-1",
        "purpose_metadata": {
            "purpose": "teaching_material",
            "created_by": "tutor-1",
            "tutor_id": "tutor-1",
        },
    }
    current_user = {"db_name": "skb_ciel", "user_id": "tutor-2", "tutor_id": "tutor-2"}

    with pytest.raises(HTTPException) as exc:
        _authorize_download(verdict, current_user)

    assert exc.value.status_code == 403


def test_download_allows_known_purpose_owner():
    verdict = {
        "tenant_db": "skb_ciel",
        "user_id": "tutor-1",
        "purpose_metadata": {
            "purpose": "teaching_material",
            "created_by": "tutor-1",
            "tutor_id": "tutor-1",
        },
    }
    current_user = {"db_name": "skb_ciel", "user_id": "tutor-1", "tutor_id": "tutor-1"}

    _authorize_download(verdict, current_user)
