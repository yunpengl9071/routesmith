# benchmark/config.py
"""Central configuration for all benchmark experiments."""
from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

# ── API ─────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
    )

# ── Binary routing models (Experiment 1) ────────────────────────────────────
STRONG_MODEL = "openai/gpt-4o"
WEAK_MODEL = "openai/gpt-4o-mini"

# ── Multi-model arms (Experiment 2) ─────────────────────────────────────────
MULTI_ARMS: list[dict] = [
    {"id": "openai/gpt-4o",                  "name": "GPT-4o",        "role": "frontier"},
    {"id": "anthropic/claude-sonnet-4-5",    "name": "Claude-Sonnet", "role": "reasoning"},
    {"id": "qwen/qwen-plus",                 "name": "Qwen-Plus",     "role": "math"},
    {"id": "minimax/minimax-m1",             "name": "MiniMax-M1",    "role": "general"},
    {"id": "deepseek/deepseek-chat-v3-0324", "name": "DeepSeek-V3",   "role": "knowledge"},
]

# ── Pricing (USD per 1M tokens) ──────────────────────────────────────────────
PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-4o":               {"input": 2.50,  "output": 10.00},
    "openai/gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "anthropic/claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "qwen/qwen-plus":              {"input": 0.40,  "output": 1.20},
    "minimax/minimax-m1":          {"input": 0.30,  "output": 1.10},
    "deepseek/deepseek-chat-v3-0324": {"input": 0.20, "output": 0.77},
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute API cost in USD from token counts."""
    p = PRICING[model]
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


# ── Experiment settings ──────────────────────────────────────────────────────
MMLU_N = 600
MMLU_CATS_N = 120        # questions per MMLU category (5 × 120 = 600)
GSM8K_N = 300
MBPP_N = 100
MULTI_MODEL_N = 1000     # queries per Experiment 2 run

NUM_SEEDS = 5
SEEDS = list(range(42, 42 + NUM_SEEDS))   # [42, 43, 44, 45, 46]
RATE_LIMIT_S = 1.5       # seconds between API calls
MAX_TOKENS_MCQ = 5       # MCQ answers are single letters
MAX_TOKENS_MATH = 512    # GSM8K needs full chain-of-thought to reach #### answer
MBPP_TIMEOUT_S = 10      # code execution timeout in seconds

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MMLU_CACHE = RESULTS_DIR / "mmlu_600_queries.json"
GSM8K_CACHE = RESULTS_DIR / "gsm8k_300_queries.json"
MBPP_CACHE = RESULTS_DIR / "mbpp_100_queries.json"
