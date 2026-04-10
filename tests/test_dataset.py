# tests/test_dataset.py
"""Tests for dataset loading (uses cached data or mocks HF)."""
from __future__ import annotations
import json
import pytest


def test_extract_answer_direct():
    from benchmark.dataset import extract_answer
    assert extract_answer("A") == "A"
    assert extract_answer("B. some text") == "B"
    assert extract_answer("The answer is C.") == "C"
    assert extract_answer("D\n") == "D"


def test_extract_answer_none():
    from benchmark.dataset import extract_answer
    assert extract_answer("") is None
    assert extract_answer("Not sure") is None


def test_extract_gsm8k_answer():
    from benchmark.dataset import extract_gsm8k_answer
    assert extract_gsm8k_answer("The answer is #### 42") == "42"
    assert extract_gsm8k_answer("She had 3 apples.\n#### 3") == "3"
    assert extract_gsm8k_answer("No marker here") is None


def test_format_mmlu_prompt():
    from benchmark.dataset import _format_mmlu_prompt
    item = {
        "question": "What is 2+2?",
        "choices": ["3", "4", "5", "6"],
    }
    prompt = _format_mmlu_prompt(item)
    assert "A. 3" in prompt
    assert "B. 4" in prompt
    assert "single letter" in prompt.lower()


def test_mmlu_cache_load(tmp_path):
    """If cache file exists, load_mmlu_sample reads from it without HF download."""
    from benchmark.dataset import load_mmlu_sample
    fake = [{
        "query_id": 1, "dataset": "mmlu", "category": "STEM", "subject": "math",
        "question": "Q?", "choices": ["A","B","C","D"],
        "answer_idx": 0, "answer_letter": "A",
        "prompt": "Question: Q?\n\nA. A\nB. B\nC. C\nD. D\n\nAnswer with a single letter (A, B, C, or D):",
    }]
    cache = tmp_path / "mmlu.json"
    cache.write_text(json.dumps(fake))
    result = load_mmlu_sample(cache_path=str(cache))
    assert len(result) == 1
    assert result[0]["category"] == "STEM"
    assert result[0]["dataset"] == "mmlu"
