"""Security tests: API-key gate, rate limiting, and simulation concurrency cap."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from supply_twin.api.main import _PUBLIC_PATHS, _whatif_semaphore, app, require_api_key


@pytest.fixture()
def _keys_enabled(monkeypatch: pytest.MonkeyPatch):
    from supply_twin.config import get_settings

    monkeypatch.setenv("SUPPLY_TWIN_API_KEYS", '["ci-key-123"]')
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_health_always_public(_keys_enabled):
    client = TestClient(app)
    assert client.get("/health").status_code == 200


def test_api_401_without_key(_keys_enabled):
    client = TestClient(app)
    assert client.get("/skus").status_code == 401
    assert "API key" in client.get("/skus").json()["detail"]


def test_api_401_with_wrong_key(_keys_enabled):
    client = TestClient(app)
    assert client.get("/skus", headers={"X-API-Key": "nope"}).status_code == 401


def test_api_200_with_valid_key(_keys_enabled):
    client = TestClient(app)
    resp = client.get("/skus", headers={"X-API-Key": "ci-key-123"})
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


def test_api_open_when_no_keys(monkeypatch: pytest.MonkeyPatch):
    from supply_twin.config import get_settings

    monkeypatch.delenv("SUPPLY_TWIN_API_KEYS", raising=False)
    get_settings.cache_clear()
    client = TestClient(app)
    assert client.get("/skus").status_code == 200
    get_settings.cache_clear()


def test_public_paths_constant() -> None:
    assert "/health" in _PUBLIC_PATHS


def test_require_api_key_direct() -> None:
    from supply_twin.config import get_settings

    class _HttpRequest:
        def __init__(self, path: str) -> None:
            self.url = type("Url", (), {"path": path})()

    request = _HttpRequest("/skus")
    # Auth disabled with empty keys.
    monkeypatched = get_settings()
    monkeypatched.api_keys = []
    assert require_api_key(request) is None

    # Without a key -> 401.
    monkeypatched.api_keys = ["k1"]
    with pytest.raises(Exception) as exc:
        require_api_key(request, None)
    assert getattr(exc.value, "status_code", None) == 401


def _client() -> TestClient:
    return TestClient(app)


def test_rate_limit_429_and_reset(monkeypatch: pytest.MonkeyPatch):
    from supply_twin.api import rate_limit
    from supply_twin.config import get_settings

    monkeypatch.delenv("SUPPLY_TWIN_API_KEYS", raising=False)
    monkeypatch.setenv("SUPPLY_TWIN_RATE_LIMIT_WINDOW", "60")
    monkeypatch.setenv("SUPPLY_TWIN_RATE_LIMIT_MAX_REQUESTS", "3")
    get_settings.cache_clear()
    try:
        rate_limit.FixedWindowRateLimiter._instances[0]._hits.clear()
        clock_values = iter([1000.0, 1000.0, 1000.0, 1000.0, 1061.0, 1061.0, 1061.0])
        monkeypatch.setattr(rate_limit, "_clock", lambda: next(clock_values))
        client = _client()

        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        blocked = client.get("/health")
        assert blocked.status_code == 429
        assert blocked.headers["Retry-After"] == "60"

        # Window elapsed -> allowed again.
        assert client.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()


def test_rate_limit_disabled_at_zero(monkeypatch: pytest.MonkeyPatch):
    from supply_twin.api import rate_limit
    from supply_twin.config import get_settings

    monkeypatch.setenv("SUPPLY_TWIN_RATE_LIMIT_MAX_REQUESTS", "0")
    get_settings.cache_clear()
    try:
        rate_limit.FixedWindowRateLimiter._instances[0]._hits.clear()
        client = _client()
        for _ in range(5):
            assert client.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()


def test_what_if_busy_returns_503(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SUPPLY_TWIN_API_KEYS", raising=False)
    client = _client()

    class _FullSemaphore:
        def acquire(self, blocking: bool = True) -> bool:  # noqa: ARG002
            return False

    monkeypatch.setattr("supply_twin.api.main._whatif_semaphore", _FullSemaphore())
    payload = [{"kind": "supplier_failure", "supplier_id": "SUP-A", "duration_days": 14}]
    resp = client.post("/simulate/what-if?horizon=2&n_runs=1", json=payload)
    assert resp.status_code == 503
    assert "busy" in resp.json()["detail"]


def test_what_if_releases_semaphore():
    """Happy path returns while holding/releasing the real semaphore."""
    payload = [{"kind": "supplier_failure", "supplier_id": "SUP-A", "duration_days": 14}]
    client = _client()
    with _whatif_semaphore:  # prove baselines don't wedge on prior releases
        pass
    resp = client.post("/simulate/what-if?horizon=2&n_runs=1", json=payload)
    assert resp.status_code == 200
    # After handling, the cap is fully available again.
    assert _whatif_semaphore.acquire(blocking=False)
    _whatif_semaphore.release()
