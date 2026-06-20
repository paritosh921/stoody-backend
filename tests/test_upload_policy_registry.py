import importlib

import pytest


def reload_policies():
    import core.upload_security.policies as policies

    return importlib.reload(policies)


def test_default_policy_values_load():
    policies = reload_policies()

    pdf_policy = policies.get_upload_policy("pdf_document")
    camera_policy = policies.get_upload_policy("camera_answer_image")
    stroke_policy = policies.get_upload_policy("hub_stroke_finalize")

    assert pdf_policy.max_size_bytes == 50 * 1024 * 1024
    assert pdf_policy.allowed_extensions == ("pdf",)
    assert pdf_policy.max_pdf_pages == 250
    assert camera_policy.max_size_bytes == 12 * 1024 * 1024
    assert camera_policy.allowed_magic_types == ("jpeg", "png")
    assert stroke_policy.max_size_bytes == 10 * 1024 * 1024
    assert stroke_policy.max_total_chunks == 5000


def test_env_override_changes_only_named_policy_field(monkeypatch):
    monkeypatch.setenv("UPLOAD_POLICY_CAMERA_ANSWER_IMAGE_MAX_SIZE_MB", "7")
    monkeypatch.setenv("UPLOAD_POLICY_HUB_RAW_DATA_BATCH_MAX_FRAMES_PER_BATCH", "123")
    policies = reload_policies()

    camera_policy = policies.get_upload_policy("camera_answer_image")
    hub_policy = policies.get_upload_policy("hub_raw_data_batch")
    pdf_policy = policies.get_upload_policy("pdf_document")

    assert camera_policy.max_size_bytes == 7 * 1024 * 1024
    assert hub_policy.max_frames_per_batch == 123
    assert pdf_policy.max_size_bytes == 50 * 1024 * 1024


def test_invalid_policy_id_raises_configuration_error():
    policies = reload_policies()

    with pytest.raises(policies.UploadPolicyConfigError):
        policies.get_upload_policy("missing_policy")


def test_public_policy_serialization_exposes_no_storage_paths():
    policies = reload_policies()

    public = policies.all_public_upload_policies()

    assert "pdf_document" in public
    assert public["pdf_document"]["max_size_bytes"] == 50 * 1024 * 1024
    serialized = repr(public).lower()
    assert "private" not in serialized
    assert "quarantine" not in serialized
    assert "released" not in serialized
