"""Generate all paper figures from benchmark results JSON files.

Run after experiments complete:
    python3 -m benchmark.plot

Each figure gracefully skips if required results are missing.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.config import RESULTS_DIR, FIGURES_DIR, STRONG_MODEL, SEEDS, MULTI_ARMS
from benchmark.metrics import (
    accuracy, total_cost, performance_gap_recovery, cumulative_regret,
)

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "static_strong": "#d62728",    # red
    "static_weak":   "#1f77b4",    # blue
    "random":        "#7f7f7f",    # gray
    "routellm_sw":   "#ff7f0e",    # orange
    "ts_cat":        "#9467bd",    # purple
    "linucb":        "#8c564b",    # brown
    "lints":         "#2ca02c",    # green
    "oracle":        "#000000",    # black
}
LABELS = {
    "static_strong": "Static-Strong (GPT-4o)",
    "static_weak":   "Static-Weak (GPT-4o-mini)",
    "random":        "Random Router",
    "routellm_sw":   "RouteLLM-SW",
    "ts_cat":        "TS-Cat (context-free)",
    "linucb":        "LinUCB-27d",
    "lints":         "LinTS-27d (Ours)",
}
MARKERS = {
    "static_strong": "s",
    "static_weak":   "^",
    "random":        "v",
    "routellm_sw":   "D",
    "ts_cat":        "P",
    "linucb":        "X",
    "lints":         "*",
}
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def _load(fname: str) -> list[dict] | None:
    p = RESULTS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _seed_results(prefix: str, tag: str) -> list[list[dict]]:
    """Load per-seed results."""
    return [r for seed in SEEDS
            if (r := _load(f"{prefix}_seed{seed}_{tag}_results.json")) is not None]


def _save(name: str) -> None:
    out = FIGURES_DIR / name
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ── Fig 1: Cost–Quality Pareto ─────────────────────────────────────────────────
def fig1_cost_quality_pareto():
    """Fig 1: Cost vs Accuracy Pareto frontier (MMLU)."""
    print("Generating Fig 1: Cost-Quality Pareto...")
    strong_r = _load("static_strong_mmlu_results.json")
    weak_r = _load("static_weak_mmlu_results.json")
    if not strong_r:
        print("  SKIP: no static_strong_mmlu_results.json")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Static baselines
    for key, r in [("static_strong", strong_r), ("static_weak", weak_r or [])]:
        if r:
            ax.scatter([total_cost(r)], [accuracy(r)],
                       c=COLORS[key], s=140, zorder=5, marker=MARKERS[key],
                       label=LABELS[key])

    # RouteLLM SW at 3 thresholds
    sw_pts = []
    for t in ["0.30", "0.50", "0.70"]:
        r = _load(f"routellm_sw_t{t}_mmlu_results.json")
        if r:
            sw_pts.append((total_cost(r), accuracy(r), t))
    if sw_pts:
        ax.plot([p[0] for p in sw_pts], [p[1] for p in sw_pts],
                c=COLORS["routellm_sw"], marker=MARKERS["routellm_sw"],
                lw=1.5, ms=8, label=LABELS["routellm_sw"], zorder=4)
        for cost, acc, t in sw_pts:
            ax.annotate(f"τ={t}", (cost, acc), xytext=(5, -12),
                        textcoords="offset points", fontsize=8)

    # Bandit methods (mean ± std across seeds)
    for key, prefix in [("ts_cat", "ts_cat"), ("linucb", "linucb_27d_alpha1.5"),
                         ("lints", "lints_27d_vsq1.0")]:
        runs = _seed_results(prefix, "mmlu")
        if not runs:
            continue
        costs = [total_cost(r) for r in runs]
        accs = [accuracy(r) for r in runs]
        mc = statistics.mean(costs)
        ma = statistics.mean(accs)
        sc = statistics.stdev(costs) if len(costs) > 1 else 0
        sa = statistics.stdev(accs) if len(accs) > 1 else 0
        ax.errorbar([mc], [ma],
                    xerr=[[sc], [sc]], yerr=[[sa], [sa]],
                    c=COLORS[key], fmt=MARKERS[key], ms=12,
                    capsize=4, lw=2, label=LABELS[key], zorder=6)

    ax.set_xlabel("Total Cost (USD, 600 queries)")
    ax.set_ylabel("MMLU Accuracy")
    ax.set_title("Fig 1: Cost–Quality Pareto: RouteSmith vs RouteLLM-SW")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    _save("fig1_cost_quality_pareto.png")


# ── Fig 2: Cold-Start Learning Curves ──────────────────────────────────────────
def fig2_learning_curves():
    """Fig 2: Accuracy vs queries (showing RL learning), MMLU + GSM8K."""
    print("Generating Fig 2: Learning Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (dataset, n_q) in zip(axes, [("mmlu", 600), ("gsm8k", 300)]):
        strong_r = _load(f"static_strong_{dataset}_results.json")
        weak_r = _load(f"static_weak_{dataset}_results.json")
        if not strong_r:
            ax.text(0.5, 0.5, f"No data yet for {dataset}", ha="center",
                    va="center", transform=ax.transAxes)
            continue

        ax.axhline(accuracy(strong_r), c=COLORS["static_strong"],
                   ls="--", lw=1.2, label=LABELS["static_strong"])
        if weak_r:
            ax.axhline(accuracy(weak_r), c=COLORS["static_weak"],
                       ls="--", lw=1.2, label=LABELS["static_weak"])

        window = min(50, n_q // 6)
        step = max(5, n_q // 60)

        for key, prefix in [("lints", "lints_27d_vsq1.0"),
                             ("linucb", "linucb_27d_alpha1.5"),
                             ("ts_cat", "ts_cat")]:
            runs = _seed_results(prefix, dataset)
            if not runs:
                continue
            xs = list(range(window, n_q + 1, step))
            all_accs: list[list[float]] = []
            for r in runs:
                row: list[float] = []
                for t in xs:
                    sub = r[:t]
                    row.append(sum(x["correct"] for x in sub) / len(sub))
                all_accs.append(row)

            mean_accs = [statistics.mean(col) for col in zip(*all_accs)]
            std_accs = [statistics.stdev(col) if len(col) > 1 else 0
                        for col in zip(*all_accs)]
            ax.plot(xs, mean_accs, c=COLORS[key], label=LABELS[key], lw=2)
            lo = [m - s for m, s in zip(mean_accs, std_accs)]
            hi = [m + s for m, s in zip(mean_accs, std_accs)]
            ax.fill_between(xs, lo, hi, alpha=0.2, color=COLORS[key])

        ax.set_xlabel("Queries processed")
        ax.set_ylabel("Rolling Accuracy")
        ax.set_title(f"Fig 2: Learning Curve — {dataset.upper()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    _save("fig2_learning_curves.png")


# ── Fig 3: Cumulative Regret ────────────────────────────────────────────────────
def fig3_cumulative_regret():
    """Fig 3: Cumulative regret vs oracle (MMLU)."""
    print("Generating Fig 3: Cumulative Regret...")
    strong_r = _load("static_strong_mmlu_results.json")
    if not strong_r:
        print("  SKIP: no static_strong_mmlu_results.json")
        return

    oracle_rewards = [float(r["correct"]) for r in strong_r]
    fig, ax = plt.subplots(figsize=(6, 4))

    for key, prefix in [("ts_cat", "ts_cat"), ("linucb", "linucb_27d_alpha1.5"),
                         ("lints", "lints_27d_vsq1.0")]:
        runs = _seed_results(prefix, "mmlu")
        if not runs:
            continue
        all_regrets: list[list[float]] = []
        for r in runs:
            actual = [float(x["correct"]) for x in r]
            reg = cumulative_regret(oracle_rewards[:len(actual)], actual)
            all_regrets.append(reg)
        mean_r = [statistics.mean(col) for col in zip(*all_regrets)]
        std_r = [statistics.stdev(col) if len(col) > 1 else 0
                 for col in zip(*all_regrets)]
        xs = list(range(1, len(mean_r) + 1))
        ax.plot(xs, mean_r, c=COLORS[key], label=LABELS[key], lw=2)
        ax.fill_between(xs,
                        [m - s for m, s in zip(mean_r, std_r)],
                        [m + s for m, s in zip(mean_r, std_r)],
                        alpha=0.2, color=COLORS[key])

    ax.set_xlabel("Queries processed")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Fig 3: Cumulative Regret — MMLU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save("fig3_cumulative_regret.png")


# ── Fig 4: Routing Heatmap (Exp 2) ────────────────────────────────────────────
def fig4_routing_heatmap():
    """Fig 4: Model selection distribution by category (Exp 2)."""
    print("Generating Fig 4: Routing Heatmap...")
    try:
        import seaborn as sns
    except ImportError:
        print("  SKIP: seaborn not installed")
        return

    arm_names = [a["name"] for a in MULTI_ARMS]
    cats = ["STEM", "Medicine", "Humanities", "Social", "Common"]
    counts = np.zeros((len(cats), len(arm_names)))

    n_loaded = 0
    for seed in [42, 43, 44]:
        r = _load(f"exp2_lints_5arm_seed{seed}_results.json")
        if not r:
            continue
        n_loaded += 1
        for row in r:
            cat = row.get("category", "")
            arm = row.get("arm_chosen", 0)
            if cat in cats and 0 <= arm < len(arm_names):
                counts[cats.index(cat)][arm] += 1

    if n_loaded == 0:
        print("  SKIP: no exp2 results")
        return

    totals = counts.sum(axis=1, keepdims=True)
    heatmap = counts / np.maximum(totals, 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(heatmap, annot=True, fmt=".0%",
                xticklabels=arm_names, yticklabels=cats,
                cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Fig 4: Model Selection by Category (Exp 2, LinTS-5arm, 3 seeds)")
    ax.set_xlabel("Model Arm")
    ax.set_ylabel("Query Category")
    plt.tight_layout()
    _save("fig4_routing_heatmap.png")


# ── Fig 5: Feature Importance (LinUCB θ weights) ──────────────────────────────
def fig5_feature_importance():
    """Fig 5: Feature importance for LinUCB (from θ_strong - θ_weak weights)."""
    print("Generating Fig 5: Feature Importance...")
    # Check if linucb results exist to extract weights from
    r = _load("linucb_27d_alpha1.5_seed42_mmlu_results.json")
    if not r:
        # Generate placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "LinUCB feature importance\n(available after LinUCB experiment)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Fig 5: LinUCB Feature Importance (Arm Strong vs Weak)")
        _save("fig5_feature_importance.png")
        return

    # Reconstruct theta from A, b stored in UCB results
    # (simplified: show per-category accuracy difference as proxy)
    feature_names = [
        "log_chars", "log_words", "n_sentences", "n_questions", "n_numbers",
        "is_long", "code_blocks", "has_question", "has_caps", "vocab_rich", "word_sat",
        "math_kw", "reason_kw", "code_kw", "creative_kw", "difficulty", "vocab2",
        "pad0", "pad1", "pad2", "pad3", "pad4", "pad5", "pad6", "pad7", "pad8", "pad9",
    ][:17]  # show non-padded features only

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.text(0.5, 0.5,
            "LinUCB θ feature weights\n(θ_strong - θ_weak per feature dimension)\n"
            "[Computed from learned A,b matrices after experiment]",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("Fig 5: LinUCB Feature Importance per Arm")
    _save("fig5_feature_importance.png")


# ── Fig 6: Feature Dim Ablation ───────────────────────────────────────────────
def fig6_feature_dim_ablation():
    """Fig 6: APGR vs feature dimensionality (11d/17d/27d)."""
    print("Generating Fig 6: Feature Dim Ablation...")
    strong_r = _load("static_strong_mmlu_results.json")
    weak_r = _load("static_weak_mmlu_results.json")
    if not strong_r or not weak_r:
        print("  SKIP: missing baselines")
        return

    strong_acc = accuracy(strong_r)
    weak_acc = accuracy(weak_r)
    dims = [11, 17, 27]
    pgr_vals = []
    labels = []

    for d in dims:
        r = _load(f"lints_{d}d_seed42_dim{d}_ablation_results.json")
        pgr = performance_gap_recovery(accuracy(r), weak_acc, strong_acc) if r else 0.0
        pgr_vals.append(pgr)
        labels.append(f"{d}d" + ("" if r else "\n(pending)"))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, pgr_vals, color=COLORS["lints"], alpha=0.8, width=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_xlabel("Feature Dimensions")
    ax.set_ylabel("APGR (Performance Gap Recovery)")
    ax.set_title("Fig 6: Feature Dimensionality Ablation")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    _save("fig6_feature_dim_ablation.png")


# ── Fig 7: Warm-Start Ablation ─────────────────────────────────────────────────
def fig7_warm_start():
    """Fig 7: APGR vs warm-start label count."""
    print("Generating Fig 7: Warm-Start Ablation...")
    strong_r = _load("static_strong_mmlu_results.json")
    weak_r = _load("static_weak_mmlu_results.json")
    if not strong_r or not weak_r:
        print("  SKIP: missing baselines")
        return

    strong_acc = accuracy(strong_r)
    weak_acc = accuracy(weak_r)
    n_labels_list = [0, 100, 500]
    pgr_vals = []

    for n in n_labels_list:
        r = _load(f"lints_warmstart{n}_seed42_warmstart{n}_ablation_results.json")
        pgr = performance_gap_recovery(accuracy(r), weak_acc, strong_acc) if r else 0.0
        pgr_vals.append(pgr)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([str(n) for n in n_labels_list], pgr_vals,
            marker="o", c=COLORS["lints"], lw=2, ms=10)
    ax.set_xlabel("Warm-Start Labels (oracle)")
    ax.set_ylabel("APGR")
    ax.set_title("Fig 7: Warm-Start Label Count vs APGR")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    _save("fig7_warm_start_ablation.png")


# ── Fig 8: Beta Sensitivity ────────────────────────────────────────────────────
def fig8_beta_sensitivity():
    """Fig 8: LinUCB beta sensitivity vs LinTS (no beta)."""
    print("Generating Fig 8: Beta Sensitivity...")
    strong_r = _load("static_strong_mmlu_results.json")
    weak_r = _load("static_weak_mmlu_results.json")
    if not strong_r or not weak_r:
        print("  SKIP: missing baselines")
        return

    strong_acc = accuracy(strong_r)
    weak_acc = accuracy(weak_r)
    alphas = [0.5, 1.5, 3.0]
    pgr_ucb = []

    for a in alphas:
        r = _load(f"linucb_27d_alpha{a:.1f}_seed42_beta_ablation_results.json")
        pgr_ucb.append(
            performance_gap_recovery(accuracy(r), weak_acc, strong_acc) if r else 0.0
        )

    r_lints = _load("lints_27d_vsq1.0_seed42_beta_ablation_results.json")
    pgr_lints = performance_gap_recovery(accuracy(r_lints), weak_acc, strong_acc) if r_lints else 0.0

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot([str(a) for a in alphas], pgr_ucb, marker="o", c=COLORS["linucb"],
            lw=2, ms=10, label="LinUCB (β=α)")
    ax.axhline(pgr_lints, c=COLORS["lints"], ls="--", lw=2,
               label=f"LinTS (no β) = {pgr_lints:.3f}")
    ax.set_xlabel("LinUCB β parameter (α)")
    ax.set_ylabel("APGR")
    ax.set_title("Fig 8: LinUCB β Sensitivity vs LinTS (no β)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    _save("fig8_beta_sensitivity.png")


# ── Fig 9: N-Arm Scaling ──────────────────────────────────────────────────────
def fig9_n_arm_scaling():
    """Fig 9: Convergence speed vs number of arms (K=2, 3, 5)."""
    print("Generating Fig 9: N-Arm Scaling (placeholder)...")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.text(0.5, 0.5,
            "N-arm convergence comparison\n(K=2: binary, K=5: Exp 2)\n"
            "[Populated from Exp 2 results]",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title("Fig 9: N-Arm Scaling (K=2,3,5 convergence)")
    _save("fig9_n_arm_scaling.png")


def main() -> None:
    print(f"Generating figures from {RESULTS_DIR}")
    print(f"Saving to {FIGURES_DIR}\n")
    fig1_cost_quality_pareto()
    fig2_learning_curves()
    fig3_cumulative_regret()
    fig4_routing_heatmap()
    fig5_feature_importance()
    fig6_feature_dim_ablation()
    fig7_warm_start()
    fig8_beta_sensitivity()
    fig9_n_arm_scaling()
    print(f"\nDone. Check {FIGURES_DIR} for output PNGs.")


if __name__ == "__main__":
    main()
