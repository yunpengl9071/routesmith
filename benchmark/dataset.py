# benchmark/dataset.py
"""Load and cache benchmark datasets: MMLU, GSM8K, MBPP."""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path


# ── MMLU ─────────────────────────────────────────────────────────────────────
MMLU_CATEGORY_SUBJECTS = {
    "STEM": [
        "high_school_mathematics", "high_school_physics",
        "high_school_chemistry", "college_mathematics",
    ],
    "Medicine": [
        "medical_genetics", "anatomy", "clinical_knowledge", "college_medicine",
    ],
    "Humanities": [
        "high_school_us_history", "high_school_world_history",
        "philosophy", "jurisprudence",
    ],
    "Social": [
        "high_school_economics", "sociology",
        "high_school_psychology", "political_science",
    ],
    "Common": [
        "common_sense_morality", "nutrition", "formal_logic", "marketing",
    ],
}
MMLU_PER_CATEGORY = 120  # 5 × 120 = 600 total


def load_mmlu_sample(seed: int = 42, cache_path: str | None = None) -> list[dict]:
    """Load 600 MMLU questions (120/category). Reads from cache if available."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    from datasets import load_dataset

    rng = random.Random(seed)
    queries: list[dict] = []

    for category, subjects in MMLU_CATEGORY_SUBJECTS.items():
        pool: list[dict] = []
        for subject in subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
                for item in ds:
                    pool.append({
                        "category": category,
                        "subject": subject,
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer_idx": item["answer"],
                        "answer_letter": ["A", "B", "C", "D"][item["answer"]],
                        "prompt": _format_mmlu_prompt(item),
                    })
            except Exception as e:
                print(f"  Warning: could not load {subject}: {e}", file=sys.stderr)

        sampled = rng.sample(pool, min(MMLU_PER_CATEGORY, len(pool)))
        queries.extend(sampled)
        print(f"  MMLU {category}: {len(sampled)} questions")

    for i, q in enumerate(queries):
        q["query_id"] = i + 1
        q["dataset"] = "mmlu"

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(queries, f, indent=2)
        print(f"Saved {len(queries)} MMLU queries to {cache_path}")

    return queries


def _format_mmlu_prompt(item: dict) -> str:
    choices_str = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(["A", "B", "C", "D"], item["choices"])
    )
    return (
        f"Question: {item['question']}\n\n"
        f"{choices_str}\n\n"
        "Answer with a single letter (A, B, C, or D):"
    )


def extract_answer(response_text: str) -> str | None:
    """Extract A/B/C/D from MCQ model response."""
    text = response_text.strip().upper()
    if text and text[0] in "ABCD":
        return text[0]
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


# ── GSM8K ─────────────────────────────────────────────────────────────────────
def load_gsm8k_sample(seed: int = 42, cache_path: str | None = None) -> list[dict]:
    """Load 300 GSM8K questions. Ground truth is numeric after '####'."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    all_items = list(ds)
    sampled = rng.sample(all_items, min(300, len(all_items)))

    queries = []
    for i, item in enumerate(sampled):
        answer = extract_gsm8k_answer(item["answer"]) or ""
        queries.append({
            "query_id": i + 1,
            "dataset": "gsm8k",
            "question": item["question"],
            "answer": answer,
            "full_solution": item["answer"],
            "prompt": (
                f"{item['question']}\n\n"
                "Solve step by step. End your answer with #### followed by the number only."
            ),
        })

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(queries, f, indent=2)
        print(f"Saved {len(queries)} GSM8K queries to {cache_path}")

    return queries


def extract_gsm8k_answer(response_text: str) -> str | None:
    """Extract numeric answer after '####' marker."""
    m = re.search(r"####\s*([\d,]+)", response_text)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


# ── MBPP ─────────────────────────────────────────────────────────────────────
def load_mbpp_sample(seed: int = 42, cache_path: str | None = None) -> list[dict]:
    """Load 100 MBPP coding problems (sanitized split)."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)

    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True
    )
    all_items = list(ds)
    sampled = rng.sample(all_items, min(100, len(all_items)))

    queries = []
    for i, item in enumerate(sampled):
        # The sanitized split uses "prompt" for the task description
        task_text = item.get("text") or item.get("prompt", "")
        queries.append({
            "query_id": i + 1,
            "dataset": "mbpp",
            "task_id": item["task_id"],
            "text": task_text,
            "test_list": item["test_list"],
            "prompt": (
                f"Write a Python function for the following task:\n\n"
                f"{task_text}\n\n"
                "Return only the Python code, no explanation."
            ),
        })

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(queries, f, indent=2)
        print(f"Saved {len(queries)} MBPP queries to {cache_path}")

    return queries


def evaluate_mbpp_code(code: str, test_list: list[str], timeout_s: int = 10) -> bool:
    """Execute generated code against test_list assertions. Returns True if all pass."""
    test_str = "\n".join(test_list)
    script = f"{code}\n\n{test_str}\nprint('PASS')"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout_s
        )
        return result.returncode == 0 and "PASS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
