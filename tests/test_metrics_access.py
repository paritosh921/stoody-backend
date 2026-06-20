from core.metrics_access import is_metrics_request_authorized


def test_metrics_request_requires_token_in_production():
    assert not is_metrics_request_authorized({}, access_token="", debug_mode=False)


def test_metrics_request_allows_debug_without_token():
    assert is_metrics_request_authorized({}, access_token="", debug_mode=True)


def test_metrics_request_accepts_bearer_token():
    assert is_metrics_request_authorized(
        {"authorization": "Bearer expected-token"},
        access_token="expected-token",
        debug_mode=False,
    )


def test_metrics_request_accepts_x_metrics_token():
    assert is_metrics_request_authorized(
        {"x-metrics-token": "expected-token"},
        access_token="expected-token",
        debug_mode=False,
    )


def test_metrics_request_rejects_wrong_token():
    assert not is_metrics_request_authorized(
        {"authorization": "Bearer wrong-token", "x-metrics-token": "wrong-token"},
        access_token="expected-token",
        debug_mode=False,
    )
