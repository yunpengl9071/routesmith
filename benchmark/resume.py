# benchmark/resume.py
"""Print current status of all experiment result files."""
from __future__ import annotations

import json
from benchmark.config import RESULTS_DIR


def main() -> None:
    files = sorted(RESULTS_DIR.glob("*_results.json"))
    if not files:
        print(f"No results yet in {RESULTS_DIR}")
        return

    print(f"{'File':<55} {'N':>5} {'Acc':>8} {'Cost':>12}")
    print("-" * 84)
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            n = len(data)
            acc = sum(r.get("correct", False) for r in data) / n if n else 0
            cost = sum(r.get("cost_usd", 0) for r in data)
            print(f"{f.name:<55} {n:>5} {acc:>7.1%} ${cost:>11.4f}")
        except Exception as e:
            print(f"{f.name:<55} ERROR: {e}")


if __name__ == "__main__":
    main()
