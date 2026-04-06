"""
Shared 27-dim feature extraction utilities for benchmarks.

Exact replication of routesmith/predictor/features.py logic so benchmarks
can be run standalone without importing the full routesmith package.
"""

import math
import re

# ---------------------------------------------------------------------------
# Keyword sets (must match src/routesmith/predictor/features.py exactly)
# ---------------------------------------------------------------------------
_MATH_WORDS = frozenset([
    "calculate", "compute", "solve", "equation", "formula", "derivative",
    "integral", "matrix", "vector", "probability", "statistics", "algebra",
    "geometry", "calculus", "theorem", "proof", "inequality", "function",
    "variable", "coefficient", "polynomial",
])
_REASONING_WORDS = frozenset([
    "why", "because", "therefore", "hence", "implies", "conclude", "infer",
    "deduce", "analyze", "evaluate", "compare", "contrast", "assess",
    "justify", "explain", "hypothesis", "argument", "evidence", "logical",
    "causal", "reason", "think",
])
_CODE_WORDS = frozenset([
    "code", "function", "implement", "algorithm", "bug", "error", "debug",
    "compile", "syntax", "api", "database", "sql", "python", "javascript",
    "rust", "java", "typescript", "html", "css", "git", "docker", "kubernetes",
])
_CREATIVE_WORDS = frozenset([
    "write", "story", "poem", "creative", "imagine", "fiction", "character",
    "narrative", "dialogue", "scene", "metaphor", "lyric", "compose", "draft",
    "essay", "blog", "article", "describe",
])
_RE_CODE_BLOCK = re.compile(r"```")

# Model specs (GPT-4-Turbo vs Mixtral-8x7B-Instruct — matches paper)
_MODEL_SPECS = {
    "gpt4": {
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "quality_score": 0.806,
        "latency_p50_ms": 1500,
        "context_window": 128000,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_json_mode": True,
    },
    "mixtral": {
        "cost_per_1k_input": 0.00024,
        "cost_per_1k_output": 0.00024,
        "quality_score": 0.681,
        "latency_p50_ms": 400,
        "context_window": 32768,
        "supports_function_calling": False,
        "supports_vision": False,
        "supports_json_mode": False,
    },
}


def _kw_overlap(words: set, target: frozenset) -> float:
    if not words:
        return 0.0
    return min(1.0, len(words & target) / len(words) * 10)


def extract_features_27d(prompt: str, model_id: str) -> list:
    """
    Extract the exact 27-dimensional feature vector used by CB-RouteSmith.

    Layout:
      [0-10]  : 11 original message features
      [11-16] : 6 extended features (query type + difficulty)
      [17-24] : 8 model features
      [25-26] : 2 interaction features

    Features are L2-normalized before return.
    """
    spec = _MODEL_SPECS.get(model_id, _MODEL_SPECS["gpt4"])
    messages = [{"role": "user", "content": prompt}]

    # ── Message features (dims 0-10) ──────────────────────────────────────
    msg_count = float(len(messages))
    lengths = [len(m.get("content", "")) for m in messages]
    total_char = float(sum(lengths))
    avg_msg = total_char / msg_count if msg_count > 0 else 0.0
    max_msg = float(max(lengths)) if lengths else 0.0
    user_msg_count = float(sum(1 for m in messages if m.get("role") == "user"))
    system_present = float(any(m.get("role") == "system" for m in messages))
    last_content = messages[-1].get("content", "") if messages else ""
    last_len = float(len(last_content))
    q_marks = float(last_content.count("?"))
    words = last_content.split()
    word_count = float(len(words))
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
    tools_present = 0.0  # no tool calls in benchmark context

    # ── Extended message features (dims 11-16) ────────────────────────────
    all_text = " ".join(m.get("content", "") for m in messages)
    all_words_lower = set(all_text.lower().split())

    math_score = _kw_overlap(all_words_lower, _MATH_WORDS)
    reasoning_score = _kw_overlap(all_words_lower, _REASONING_WORDS)
    code_score = _kw_overlap(all_words_lower, _CODE_WORDS)
    creative_score = _kw_overlap(all_words_lower, _CREATIVE_WORDS)

    has_code = bool(_RE_CODE_BLOCK.search(all_text)) or any(
        k in all_text for k in ("def ", "function ", "class ")
    )
    if has_code:
        code_score = min(1.0, code_score + 0.3)

    length_signal = min(1.0, len(words) / 200.0)
    vocab_signal = min(1.0, avg_word_len / 8.0)
    structure_signal = (0.3 if has_code else 0.0) + 0.2 * math_score + 0.2 * reasoning_score
    difficulty = min(
        1.0,
        0.3 * length_signal + 0.25 * vocab_signal
        + 0.35 * structure_signal + 0.1 * min(1.0, q_marks / 3.0),
    )
    vocab_richness = len(set(w.lower() for w in words)) / max(1, len(words))

    # ── Model features (dims 17-24) ───────────────────────────────────────
    model_feats = [
        spec["cost_per_1k_input"],
        spec["cost_per_1k_output"],
        spec["quality_score"],
        spec["latency_p50_ms"] / 1000.0,   # ms → seconds
        math.log(max(spec["context_window"], 1)),
        1.0 if spec["supports_function_calling"] else 0.0,
        1.0 if spec["supports_vision"] else 0.0,
        1.0 if spec["supports_json_mode"] else 0.0,
    ]

    # ── Interaction features (dims 25-26) ─────────────────────────────────
    est_resp_tokens = min(1.0, (word_count * 2 + difficulty * 500) / 2000.0)
    diff_x_quality = difficulty * spec["quality_score"]

    feats = [
        msg_count, total_char, avg_msg, max_msg, user_msg_count,
        system_present, last_len, q_marks, word_count, avg_word_len,
        tools_present,
        math_score, reasoning_score, code_score, creative_score,
        difficulty, vocab_richness,
    ] + model_feats + [est_resp_tokens, diff_x_quality]

    assert len(feats) == 27, f"Expected 27 dims, got {len(feats)}"

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in feats))
    if norm > 1e-9:
        feats = [v / norm for v in feats]
    return feats


def l2_norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


# Expose model cost for reward computation
def model_cost_total(model_id: str) -> float:
    spec = _MODEL_SPECS.get(model_id, _MODEL_SPECS["gpt4"])
    return spec["cost_per_1k_input"] + spec["cost_per_1k_output"]


MAX_COST = model_cost_total("gpt4")
