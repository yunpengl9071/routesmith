"""Tests for semantic cache wiring in the completion path."""
from __future__ import annotations

import time

from routesmith.config import CacheConfig
from tests.helpers import fake_response, make_rs


def test_cache_disabled_by_default(monkeypatch):
    call_count = [0]

    def fake_completion(*args, **kwargs):
        call_count[0] += 1
        return fake_response()

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)
    rs = make_rs()
    assert rs._cache is None
    msg = [{"role": "user", "content": "hello"}]
    rs.completion(messages=msg)
    rs.completion(messages=msg)
    assert call_count[0] == 2


def test_cache_hit_skips_model_call(monkeypatch):
    call_count = [0]

    def fake_completion(*args, **kwargs):
        call_count[0] += 1
        return fake_response(content=f"response_{call_count[0]}")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)
    rs = make_rs(cache=CacheConfig(enabled=True))
    rs._cache_semantic = False  # force exact-match mode
    msg = [{"role": "user", "content": "hello"}]
    resp1 = rs.completion(messages=msg)
    resp2 = rs.completion(messages=msg)
    assert call_count[0] == 1
    assert resp2.choices[0].message.content == resp1.choices[0].message.content
    assert rs.stats["cache_hits"] == 1


def test_cache_hit_returns_copy(monkeypatch):
    call_count = [0]

    def fake_completion(*args, **kwargs):
        call_count[0] += 1
        return fake_response(content=f"response_{call_count[0]}")

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)
    rs = make_rs(cache=CacheConfig(enabled=True))
    rs._cache_semantic = False
    msg = [{"role": "user", "content": "hello"}]
    resp1 = rs.completion(messages=msg)
    resp1.choices[0].message.content = "mutated"
    resp2 = rs.completion(messages=msg)
    assert resp2.choices[0].message.content != "mutated"
    assert resp2.choices[0].message.content == "response_1"


def test_cache_bypassed_for_tools(monkeypatch):
    call_count = [0]

    def fake_completion(*args, **kwargs):
        call_count[0] += 1
        return fake_response()

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)
    rs = make_rs(cache=CacheConfig(enabled=True))
    rs._cache_semantic = False
    msg = [{"role": "user", "content": "hello"}]
    rs.completion(messages=msg, tools=[{"type": "function", "function": {"name": "test"}}])
    rs.completion(messages=msg, tools=[{"type": "function", "function": {"name": "test"}}])
    assert call_count[0] == 2


def test_cache_hit_latency(monkeypatch):
    call_count = [0]

    def fake_completion(*args, **kwargs):
        call_count[0] += 1
        return fake_response()

    monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)
    rs = make_rs(cache=CacheConfig(enabled=True))
    rs._cache_semantic = False
    msg = [{"role": "user", "content": "hello"}]
    rs.completion(messages=msg)  # warm the cache
    start = time.perf_counter()
    rs.completion(messages=msg)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 10  # ms
