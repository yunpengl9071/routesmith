"""Tests for fallback_model retry on primary model failure (P0.3)."""

import pytest

from routesmith.utils.retry import RetryExhaustedError
from tests.helpers import fake_response, make_rs


def test_fallback_used_on_primary_failure(monkeypatch):
    models_called = []

    def fake_completion(*args, **kwargs):
        model = kwargs.get("model", "gpt-4o-mini")
        models_called.append(model)
        if model == "gpt-4o-mini":
            raise RuntimeError("primary failure")
        return fake_response(model="gpt-4o")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)

    rs = make_rs(fallback_model="gpt-4o")
    response = rs.completion(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4o-mini",
    )
    assert response.model == "gpt-4o"
    assert "gpt-4o" in models_called
    assert "gpt-4o-mini" in models_called
    md = rs.last_routing_metadata
    assert md is not None
    assert md.fallback_from == "gpt-4o-mini"


def test_no_fallback_configured_reraises(monkeypatch):
    def fake_completion(*args, **kwargs):
        raise RuntimeError("no fallback")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)

    rs = make_rs()  # no fallback_model
    with pytest.raises(RetryExhaustedError):
        rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o-mini",
        )


def test_fallback_not_registered_reraises(monkeypatch):
    def fake_completion(*args, **kwargs):
        raise RuntimeError("unregistered fallback")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)

    rs = make_rs(fallback_model="claude-3")  # not registered
    with pytest.raises(RetryExhaustedError):
        rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o-mini",
        )


def test_fallback_same_as_selected_reraises(monkeypatch):
    def fake_completion(*args, **kwargs):
        raise RuntimeError("same model")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)

    rs = make_rs(fallback_model="gpt-4o")
    with pytest.raises(RetryExhaustedError):
        rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o",
        )


def test_fallback_failure_reraises_primary(monkeypatch):
    models_called = []

    def fake_completion(*args, **kwargs):
        model = kwargs.get("model", "gpt-4o-mini")
        models_called.append(model)
        if model == "gpt-4o-mini":
            raise RuntimeError("primary error")
        raise RuntimeError("fallback error")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)

    rs = make_rs(fallback_model="gpt-4o")
    with pytest.raises(RuntimeError, match="primary error"):
        rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-4o-mini",
        )
    assert "gpt-4o" in models_called
    assert "gpt-4o-mini" in models_called


async def test_async_fallback_used_on_primary_failure(monkeypatch):
    models_called = []

    async def fake_acompletion(*args, **kwargs):
        model = kwargs.get("model", "gpt-4o-mini")
        models_called.append(model)
        if model == "gpt-4o-mini":
            raise RuntimeError("primary failure")
        return fake_response(model="gpt-4o")

    monkeypatch.setattr("routesmith.client.litellm.acompletion", fake_acompletion)

    rs = make_rs(fallback_model="gpt-4o")
    response = await rs.acompletion(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4o-mini",
    )
    assert response.model == "gpt-4o"
    assert "gpt-4o" in models_called
    assert "gpt-4o-mini" in models_called
    md = rs.last_routing_metadata
    assert md is not None
    assert md.fallback_from == "gpt-4o-mini"
