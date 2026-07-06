"""Tests for shared test helpers."""
from tests.helpers import fake_response


def test_fake_response_shape():
    resp = fake_response()
    assert resp.usage.total_tokens == 30
    assert resp.model_dump()["choices"][0]["message"]["content"] == "ok"
