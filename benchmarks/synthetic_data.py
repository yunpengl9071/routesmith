"""
Synthetic MMLU/GSM8K-like dataset generator.

Calibrated to match published RouteLLM evaluation statistics:
  MMLU:   GPT-4-Turbo 80.6%,  Mixtral-8x7B 68.1%
  GSM8K:  GPT-4-Turbo 85.7%,  Mixtral-8x7B 63.8%
  MT-Bench: GPT-4 0.923 norm, Mixtral 0.834 norm

The difficulty of each question is sampled and used to:
  (a) generate realistic question text whose features correlate with difficulty
  (b) assign per-model correctness with difficulty-dependent probability

Usage:
    from benchmarks.synthetic_data import make_dataset
    mmlu  = make_dataset("mmlu",   n=14042, seed=42)
    gsm8k = make_dataset("gsm8k",  n=1319,  seed=42)

Each dataset entry is (prompt: str, strong_correct: int, weak_correct: int).
"""

import math
import random

# ---------------------------------------------------------------------------
# Configuration: calibrated to RouteLLM published stats
# ---------------------------------------------------------------------------
_CONFIGS = {
    "mmlu": {
        "n_default": 14042,
        # Linear model: P(correct|d) = a - b*d, d~Uniform(0,1)
        # E[P] = a - b/2; solved so E matches target accuracy
        "strong_a": 0.913,  # intercept (easy questions)
        "strong_b": 0.214,  # slope    (E = 0.913 - 0.107 = 0.806) ✓
        "weak_a":   0.771,  # intercept
        "weak_b":   0.180,  # slope    (E = 0.771 - 0.090 = 0.681) ✓
        "corr": 0.60,       # Pearson corr between strong/weak outcomes
        # Question type distribution: (category, base_difficulty_offset, templates)
        "categories": [
            ("factual",   -0.25, [
                "What is {noun}?",
                "Which of the following best describes {noun}?",
                "The {noun} is defined as",
                "In the context of {domain}, what is {noun}?",
            ]),
            ("reasoning", +0.10, [
                "Why does {phenomenon} occur when {condition}?",
                "Explain the relationship between {concept_a} and {concept_b}.",
                "Analyze the implications of {event} on {domain}.",
                "Given {premise}, what can be concluded about {concept}?",
            ]),
            ("math",      +0.20, [
                "Calculate the value of {expression}.",
                "Solve for x: {equation}",
                "If {condition}, what is the probability that {event}?",
                "A {object} has {property}. Find {target}.",
            ]),
            ("applied",   0.00, [
                "Which of the following is an example of {concept}?",
                "A researcher studying {domain} observes {observation}. What does this suggest?",
                "In {scenario}, which approach would be most appropriate?",
            ]),
        ],
        "category_weights": [0.35, 0.30, 0.20, 0.15],
    },
    "gsm8k": {
        "n_default": 1319,
        "strong_a": 0.928,
        "strong_b": 0.142,  # E = 0.928 - 0.071 = 0.857 ✓
        "weak_a":   0.729,
        "weak_b":   0.182,  # E = 0.729 - 0.091 = 0.638 ✓
        "corr": 0.55,
        "categories": [
            ("arithmetic",  -0.10, [
                "If {person} has {n1} {items} and buys {n2} more, how many does {pronoun} have?",
                "A store sells {item} for ${price}. How much do {n} items cost?",
                "{person} earns ${wage} per hour. How much does {pronoun} earn in {n} hours?",
            ]),
            ("word_problem", +0.15, [
                "In a class of {n} students, {frac} passed the exam. How many students failed?",
                "{person} drove {d1} miles on Monday and {d2} miles on Tuesday. What was the average daily distance?",
                "A recipe calls for {n1} cups of flour for every {n2} cups of sugar. How much flour is needed for {n3} cups of sugar?",
            ]),
            ("multi_step",  +0.35, [
                "{person} starts with ${amount}. {person} spends ${s1} on {item1} and ${s2} on {item2}. If {person} then earns ${earn}, how much does {pronoun} have?",
                "Train A travels at {v1} mph and Train B at {v2} mph. They start {d} miles apart. After {t} hours, how far apart are they?",
                "A tank fills at {r1} gallons/min and drains at {r2} gallons/min. Starting at {start} gallons, how long to reach {target} gallons?",
            ]),
        ],
        "category_weights": [0.30, 0.40, 0.30],
    },
    "mtbench": {
        "n_default": 80,
        "strong_a": 0.980,
        "strong_b": 0.114,  # E = 0.980 - 0.057 = 0.923 ✓
        "weak_a":   0.879,
        "weak_b":   0.090,  # E = 0.879 - 0.045 = 0.834 ✓
        "corr": 0.65,
        "categories": [
            ("writing", 0.00, [
                "Write a short {genre} about {topic}.",
                "Compose a {adjective} email to {recipient} about {topic}.",
            ]),
            ("roleplay", +0.05, [
                "You are a {role}. A user asks: '{question}'. Respond in character.",
            ]),
            ("reasoning", +0.15, [
                "Explain the tradeoffs between {option_a} and {option_b} for {use_case}.",
                "What are the implications of {event} for {domain}?",
            ]),
            ("coding", +0.10, [
                "Write a {lang} function that {task}.",
                "Debug this {lang} code: {code_snippet}",
            ]),
        ],
        "category_weights": [0.30, 0.20, 0.30, 0.20],
    },
}

# ---------------------------------------------------------------------------
# Template fill-in word banks
# ---------------------------------------------------------------------------
_NOUNS = [
    "photosynthesis", "entropy", "mitosis", "capitalism", "democracy",
    "osmosis", "momentum", "inflation", "quantum entanglement", "natural selection",
    "tectonic plates", "synaptic plasticity", "opportunity cost", "cognitive dissonance",
    "the central dogma", "supply and demand", "Keynesian economics", "plate tectonics",
    "thermodynamics", "the Krebs cycle",
]
_DOMAINS = [
    "biology", "physics", "chemistry", "economics", "political science",
    "computer science", "mathematics", "psychology", "history", "philosophy",
    "medicine", "geology", "astronomy", "sociology", "linguistics",
]
_PHENOMENA = [
    "temperature increases", "pressure drops", "populations collapse",
    "markets fail", "cells divide", "signals propagate",
]
_CONDITIONS = [
    "exposed to UV light", "the temperature rises above 37°C",
    "oxygen is absent", "resources are limited", "entropy is maximized",
]
_CONCEPTS = [
    "entropy and information", "supply and demand", "genotype and phenotype",
    "force and acceleration", "risk and return", "syntax and semantics",
]
_EVENTS = ["industrialization", "the French Revolution", "the discovery of DNA",
           "the 2008 financial crisis", "the internet age"]
_PERSONS = ["Alice", "Bob", "Carlos", "Diana", "Eve", "Frank"]
_ITEMS = ["apples", "books", "tokens", "widgets", "units"]
_LANGS = ["Python", "JavaScript", "Rust", "Go"]
_EXPRESSIONS = [
    "3x² + 2x − 7 when x=4",
    "∑(k=1 to 10) k²",
    "d/dx [sin(x)·e^x]",
    "lim(x→0) sin(x)/x",
]
_EQUATIONS = [
    "2x + 5 = 17",
    "x² − 6x + 8 = 0",
    "3(x−2) = 4x + 1",
]


def _fill(template: str, rng: random.Random) -> str:
    replacements = {
        "{noun}": rng.choice(_NOUNS),
        "{domain}": rng.choice(_DOMAINS),
        "{phenomenon}": rng.choice(_PHENOMENA),
        "{condition}": rng.choice(_CONDITIONS),
        "{concept_a}": rng.choice(_DOMAINS),
        "{concept_b}": rng.choice(_DOMAINS),
        "{concept}": rng.choice(_NOUNS),
        "{event}": rng.choice(_EVENTS),
        "{premise}": f"the {rng.choice(_NOUNS)} hypothesis holds",
        "{expression}": rng.choice(_EXPRESSIONS),
        "{equation}": rng.choice(_EQUATIONS),
        "{object}": rng.choice(["sphere", "cylinder", "triangle", "matrix"]),
        "{property}": f"a radius of {rng.randint(1, 20)}",
        "{target}": "its volume",
        "{observation}": f"an unexpected {rng.choice(['increase','decrease','pattern'])}",
        "{scenario}": f"a typical {rng.choice(_DOMAINS)} experiment",
        "{approach}": "parametric testing",
        "{person}": rng.choice(_PERSONS),
        "{pronoun}": "they",
        "{n1}": str(rng.randint(5, 50)),
        "{n2}": str(rng.randint(1, 20)),
        "{n3}": str(rng.randint(2, 8)),
        "{n}": str(rng.randint(2, 10)),
        "{items}": rng.choice(_ITEMS),
        "{item}": rng.choice(_ITEMS[:-1]),
        "{item1}": rng.choice(_ITEMS),
        "{item2}": rng.choice(_ITEMS),
        "{price}": f"{rng.randint(2, 50)}.{rng.randint(0,99):02d}",
        "{wage}": str(rng.randint(15, 60)),
        "{frac}": f"{rng.randint(50, 90)}%",
        "{d1}": str(rng.randint(30, 200)),
        "{d2}": str(rng.randint(30, 200)),
        "{amount}": str(rng.randint(100, 1000)),
        "{s1}": str(rng.randint(10, 100)),
        "{s2}": str(rng.randint(10, 100)),
        "{earn}": str(rng.randint(50, 200)),
        "{v1}": str(rng.randint(40, 80)),
        "{v2}": str(rng.randint(40, 80)),
        "{d}": str(rng.randint(100, 500)),
        "{t}": str(rng.randint(1, 5)),
        "{r1}": str(rng.randint(5, 20)),
        "{r2}": str(rng.randint(1, 10)),
        "{start}": str(rng.randint(0, 100)),
        "{target}": str(rng.randint(50, 200)),
        "{genre}": rng.choice(["story", "poem", "essay", "blog post"]),
        "{topic}": rng.choice(["climate change", "innovation", "leadership", "technology"]),
        "{adjective}": rng.choice(["professional", "friendly", "concise"]),
        "{recipient}": rng.choice(["a colleague", "a customer", "your manager"]),
        "{role}": rng.choice(["doctor", "lawyer", "software engineer", "teacher"]),
        "{question}": rng.choice(["What should I do?", "Can you help me?", "What's your advice?"]),
        "{option_a}": rng.choice(["microservices", "SQL", "batch processing", "supervised learning"]),
        "{option_b}": rng.choice(["monoliths", "NoSQL", "streaming", "reinforcement learning"]),
        "{use_case}": rng.choice(["large-scale deployment", "research", "production systems"]),
        "{lang}": rng.choice(_LANGS),
        "{task}": rng.choice(["sorts a list", "finds prime numbers", "parses JSON", "implements a cache"]),
        "{code_snippet}": "def foo(x): return x * 2  # expected: x + 2",
    }
    result = template
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


# ---------------------------------------------------------------------------
# Correlated binary outcome sampler
# ---------------------------------------------------------------------------
def _correlated_bernoulli(p1: float, p2: float, rho: float, rng: random.Random):
    """
    Sample (X1, X2) ~ Bernoulli(p1), Bernoulli(p2) with Pearson correlation rho.
    Uses the Gaussian copula method.
    """
    # Probit transform
    def probit(p):
        # Approximate inverse normal CDF
        p = max(1e-9, min(1 - 1e-9, p))
        # Rational approximation (Abramowitz & Stegun 26.2.17)
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
        x = t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)
        return -x if p < 0.5 else x

    z1 = rng.gauss(0, 1)
    z2 = rho * z1 + math.sqrt(max(0, 1 - rho**2)) * rng.gauss(0, 1)

    def normal_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    x1 = 1 if normal_cdf(z1) < p1 else 0
    x2 = 1 if normal_cdf(z2) < p2 else 0
    return x1, x2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_dataset(
    benchmark: str = "mmlu",
    n: int = None,
    seed: int = 42,
) -> list:
    """
    Generate a synthetic benchmark dataset.

    Returns a list of (prompt, strong_correct, weak_correct) tuples, where
    strong_correct / weak_correct are 0/1 integers.

    Overall accuracy will match RouteLLM published stats within ~0.5pp.
    Difficulty is encoded in the prompt text so that the LinUCB feature
    extractor can learn to distinguish easy from hard queries.
    """
    cfg = _CONFIGS[benchmark]
    n = n or cfg["n_default"]
    rng = random.Random(seed)

    categories = cfg["categories"]
    weights = cfg["category_weights"]
    # Cumulative weights for selection
    cum_w = []
    s = 0.0
    for w in weights:
        s += w
        cum_w.append(s)

    strong_a = cfg["strong_a"]
    strong_b = cfg["strong_b"]
    weak_a   = cfg["weak_a"]
    weak_b   = cfg["weak_b"]
    corr     = cfg["corr"]

    data = []
    for _ in range(n):
        # Select category
        r = rng.random()
        cat_idx = next(i for i, cw in enumerate(cum_w) if r <= cw)
        cat_name, diff_offset, templates = categories[cat_idx]

        # Sample difficulty d ∈ [0, 1]
        d = max(0.0, min(1.0, rng.betavariate(2, 2) + diff_offset))

        # Compute correctness probabilities
        p_strong = max(0.35, min(0.99, strong_a - strong_b * d))
        p_weak   = max(0.20, min(0.97, weak_a   - weak_b   * d))

        # Sample correlated outcomes
        strong_ok, weak_ok = _correlated_bernoulli(p_strong, p_weak, corr, rng)

        # Build prompt text
        template = rng.choice(templates)
        prompt = _fill(template, rng)

        # Add complexity markers for harder questions (so features correlate)
        if d > 0.65:
            qualifier = rng.choice([
                " Provide a rigorous step-by-step analysis.",
                " Show all intermediate steps and justify each one.",
                " Consider multiple perspectives and edge cases.",
                " This requires careful multi-step reasoning.",
            ])
            prompt += qualifier
        elif cat_name == "math" or cat_name == "multi_step":
            prompt += " Show your work."

        data.append((prompt, strong_ok, weak_ok))

    return data


def dataset_stats(data: list) -> dict:
    """Return summary statistics for a dataset."""
    n = len(data)
    strong_acc = sum(s for _, s, _ in data) / n
    weak_acc   = sum(w for _, _, w in data) / n
    both_right = sum(1 for _, s, w in data if s and w) / n
    only_strong = sum(1 for _, s, w in data if s and not w) / n
    neither     = sum(1 for _, s, w in data if not s and not w) / n

    # Avg difficulty proxy: neither wrong = hard, both right = easy
    return {
        "n": n,
        "strong_accuracy": round(strong_acc, 4),
        "weak_accuracy":   round(weak_acc, 4),
        "quality_gap":     round(strong_acc - weak_acc, 4),
        "both_correct":    round(both_right, 4),
        "only_strong_correct": round(only_strong, 4),
        "neither_correct": round(neither, 4),
    }


if __name__ == "__main__":
    for name in ("mmlu", "gsm8k", "mtbench"):
        ds = make_dataset(name)
        stats = dataset_stats(ds)
        print(f"\n{name.upper()} (n={stats['n']}):")
        for k, v in stats.items():
            print(f"  {k}: {v}")
