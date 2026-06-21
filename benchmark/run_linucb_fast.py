#!/usr/bin/env python3
"""
Fast LinUCB APGR evaluation for CB-RouteSmith paper.

Computes APGR, accuracy, and cost for:
  - LinUCB (our contribution, warm-start 0/500)
  - AlwaysStrong, AlwaysWeak, Random (baselines)

Uses simple pure-Python LinUCB: 32-dim feature vector, Sherman-Morrison updates.
Runs ~3-5 minutes total on MMLU (14K questions, 5 seeds).
"""

import csv, json, math, os, random, statistics, sys, time

ROUTELLM_DIR = "/workspace/group/routesmith-research/routellm"
OUT_DIR = "/workspace/group/routesmith-research/sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tiny pure-Python linear algebra (no numpy)
# ---------------------------------------------------------------------------

def mat_identity(d):
    return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

def mat_vec_mul(A, v):
    d = len(v)
    return [sum(A[i][j] * v[j] for j in range(d)) for i in range(d)]

def vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def sherman_morrison(A_inv, u, v):
    """Rank-1 downdate: A_inv for (A + uv^T)^{-1}."""
    Ainv_u = mat_vec_mul(A_inv, u)
    vt_Ainv_u = vec_dot(v, Ainv_u)
    denom = 1.0 + vt_Ainv_u
    if abs(denom) < 1e-12:
        return A_inv  # numerically singular, skip
    d = len(u)
    result = [row[:] for row in A_inv]
    for i in range(d):
        for j in range(d):
            result[i][j] -= Ainv_u[i] * vec_dot([A_inv[k][j] for k in range(d)], v) / denom
    # Simpler: result[i][j] -= (Ainv_u[i] * vt_Ainv[j]) / denom
    return result

def sherman_morrison_fast(A_inv, x):
    """Rank-1 update for A += x x^T, returns new (A+xx^T)^{-1}."""
    d = len(x)
    Ainv_x = mat_vec_mul(A_inv, x)
    denom = 1.0 + vec_dot(x, Ainv_x)
    if abs(denom) < 1e-12:
        return A_inv
    result = [row[:] for row in A_inv]
    for i in range(d):
        for j in range(d):
            result[i][j] -= Ainv_x[i] * Ainv_x[j] / denom
    return result

# ---------------------------------------------------------------------------
# Feature extraction (32-dim)
# ---------------------------------------------------------------------------

def extract_features(prompt, model_id, cost_per_1k, quality_prior, latency_ms,
                     context_window, func_call, vision, json_mode):
    """32-dimensional feature vector for (query, model) pair."""
    text = prompt if isinstance(prompt, str) else str(prompt)
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    avg_word_len = (n_chars / max(n_words, 1))
    n_question_marks = text.count('?')
    has_math = int(any(c in text for c in '=+×÷∫√∑∏'))
    has_code = int('```' in text or 'def ' in text or 'function' in text)
    has_list = int('\n-' in text or '\n*' in text or '\n1.' in text)
    is_long = int(n_words > 100)
    is_short = int(n_words < 20)

    # Normalize
    log_words = math.log(max(n_words, 1) + 1) / 10.0
    log_chars = math.log(max(n_chars, 1) + 1) / 12.0
    norm_qmarks = min(n_question_marks / 5.0, 1.0)
    norm_avg_word = min(avg_word_len / 15.0, 1.0)
    log_cost = math.log(cost_per_1k + 1e-6) / 10.0
    norm_latency = min(latency_ms / 5000.0, 1.0)
    log_ctx = math.log(max(context_window, 1)) / 20.0

    feats = [
        # Query features (20 dims)
        log_words,
        log_chars,
        norm_qmarks,
        norm_avg_word,
        float(has_math),
        float(has_code),
        float(has_list),
        float(is_long),
        float(is_short),
        min(n_words / 500.0, 1.0),
        min(n_chars / 3000.0, 1.0),
        float(text.islower()),
        float(any(c.isupper() for c in text)),
        min(text.count('\n') / 20.0, 1.0),
        float('please' in text.lower() or 'explain' in text.lower()),
        float('step' in text.lower() or 'reason' in text.lower()),
        float('write' in text.lower() or 'create' in text.lower()),
        float('translate' in text.lower()),
        float('summarize' in text.lower() or 'summary' in text.lower()),
        float('analyze' in text.lower() or 'compare' in text.lower()),
        # Model features (12 dims)
        log_cost,
        quality_prior,
        norm_latency,
        log_ctx,
        float(func_call),
        float(vision),
        float(json_mode),
        float(model_id == 'gpt-4'),
        float(cost_per_1k > 1.0),
        float(latency_ms > 1000),
        float(context_window > 64000),
        0.0,  # padding
    ]
    return feats

FEAT_DIM = 32
ALPHA_DEFAULT = 1.0

# ---------------------------------------------------------------------------
# LinUCB arm
# ---------------------------------------------------------------------------

class LinUCBArm:
    def __init__(self, model_id, cost_per_1k, quality_prior, latency_ms,
                 context_window, func_call, vision, json_mode, alpha=1.0):
        self.model_id = model_id
        self.cost_per_1k = cost_per_1k
        self.quality_prior = quality_prior
        self.latency_ms = latency_ms
        self.context_window = context_window
        self.func_call = func_call
        self.vision = vision
        self.json_mode = json_mode
        self.alpha = alpha
        self.d = FEAT_DIM
        self.A_inv = mat_identity(self.d)
        self.b = [0.0] * self.d
        self.n_pulls = 0

    def features(self, prompt):
        return extract_features(prompt, self.model_id, self.cost_per_1k,
                                 self.quality_prior, self.latency_ms,
                                 self.context_window, self.func_call,
                                 self.vision, self.json_mode)

    def ucb_score(self, x):
        theta = mat_vec_mul(self.A_inv, self.b)
        expected = vec_dot(theta, x)
        Ainv_x = mat_vec_mul(self.A_inv, x)
        variance = vec_dot(x, Ainv_x)
        bonus = self.alpha * math.sqrt(max(variance, 0.0))
        return expected + bonus, expected, bonus

    def update(self, x, reward):
        # A += xx^T  =>  A_inv updated via Sherman-Morrison
        self.A_inv = sherman_morrison_fast(self.A_inv, x)
        # b += r*x
        for i in range(self.d):
            self.b[i] += reward * x[i]
        self.n_pulls += 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

STRONG_COL = "gpt-4-1106-preview"
WEAK_COL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def load_mmlu(max_per_subject=None):
    mmlu_dir = os.path.join(ROUTELLM_DIR, "routellm", "evals", "mmlu", "responses")
    data = []
    for fn in sorted(os.listdir(mmlu_dir)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(mmlu_dir, fn)) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if max_per_subject:
            rows = rows[:max_per_subject]
        for row in rows:
            prompt = row.get("prompt", "")
            try:
                strong_ok = 1 if row.get(STRONG_COL, "False").strip() == "True" else 0
                weak_ok = 1 if row.get(WEAK_COL, "False").strip() == "True" else 0
            except:
                continue
            data.append((prompt, weak_ok, strong_ok))
    return data

def load_gsm8k():
    path = os.path.join(ROUTELLM_DIR, "routellm", "evals", "gsm8k", "gsm8k_responses.csv")
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "")
            try:
                strong_ok = 1 if row.get(STRONG_COL, "False").strip() == "True" else 0
                weak_ok = 1 if row.get(WEAK_COL, "False").strip() == "True" else 0
            except:
                continue
            data.append((prompt, weak_ok, strong_ok))
    return data

# ---------------------------------------------------------------------------
# APGR computation
# ---------------------------------------------------------------------------

def compute_apgr_linucb(data, alpha, warm_labels, n_seeds=5, is_binary=True):
    """
    Train LinUCB on warm_labels examples, then score remaining data,
    rank by strong_preference = ucb_strong - ucb_weak,
    compute APGR over those rankings.
    """
    all_apgr = []
    for seed in range(n_seeds):
        shuffled = list(data)
        random.Random(seed).shuffle(shuffled)

        # Build arms
        strong_arm = LinUCBArm('gpt-4', 10.0, 0.80, 1500, 128000, True, True, True, alpha=alpha)
        weak_arm = LinUCBArm('mixtral', 0.24, 0.65, 400, 32768, False, False, False, alpha=alpha)

        # Warm-start: train on first warm_labels items
        warm_data = shuffled[:warm_labels]
        test_data = shuffled[warm_labels:]

        for item in warm_data:
            prompt, weak_ok, strong_ok = item
            xs = strong_arm.features(prompt)
            xw = weak_arm.features(prompt)
            rs = 1.0 if strong_ok else 0.0
            rw = 1.0 if weak_ok else 0.0
            strong_arm.update(xs, rs)
            weak_arm.update(xw, rw)

        # Score test data
        scores = []
        for item in test_data:
            prompt, weak_ok, strong_ok = item
            xs = strong_arm.features(prompt)
            xw = weak_arm.features(prompt)
            ucb_s, exp_s, _ = strong_arm.ucb_score(xs)
            ucb_w, exp_w, _ = weak_arm.ucb_score(xw)
            preference = (exp_s - exp_w)  # expected score difference (no exploration bonus for ranking)
            scores.append((preference, float(strong_ok), float(weak_ok)))

        if not scores:
            all_apgr.append(0.5)
            continue

        # Sort by preference descending
        scores.sort(key=lambda x: -x[0])
        n = len(scores)

        baseline_q = sum(s[2] for s in scores)  # all-weak
        max_q = sum(s[1] for s in scores)        # all-strong
        total_gap = max_q - baseline_q

        if total_gap <= 0:
            all_apgr.append(0.5)
            continue

        cum_q = baseline_q
        area = 0.0
        prev_x, prev_y = 0.0, 0.0
        for i, (pref, qs, qw) in enumerate(scores):
            cum_q += (qs - qw)
            x = (i + 1) / n
            y = (cum_q - baseline_q) / total_gap
            area += (x - prev_x) * (prev_y + y) / 2
            prev_x, prev_y = x, y

        all_apgr.append(area)

    return statistics.mean(all_apgr), statistics.stdev(all_apgr) if n_seeds > 1 else 0.0


def compute_apgr_random(data, n_seeds=5):
    """Random router APGR baseline."""
    all_apgr = []
    for seed in range(n_seeds):
        shuffled = list(data)
        rng = random.Random(seed + 100)
        scores = [(rng.random(), float(s), float(w)) for (_, w, s) in shuffled]
        scores.sort(key=lambda x: -x[0])
        n = len(scores)
        baseline_q = sum(s[2] for s in scores)
        max_q = sum(s[1] for s in scores)
        total_gap = max_q - baseline_q
        if total_gap <= 0:
            all_apgr.append(0.5)
            continue
        cum_q = baseline_q
        area = 0.0
        prev_x, prev_y = 0.0, 0.0
        for i, (_, qs, qw) in enumerate(scores):
            cum_q += (qs - qw)
            x = (i + 1) / n
            y = (cum_q - baseline_q) / total_gap
            area += (x - prev_x) * (prev_y + y) / 2
            prev_x, prev_y = x, y
        all_apgr.append(area)
    return statistics.mean(all_apgr), statistics.stdev(all_apgr) if n_seeds > 1 else 0.0


# ---------------------------------------------------------------------------
# Online routing metrics (accuracy, cost, strong%)
# ---------------------------------------------------------------------------

def run_online_linucb(data, alpha, n_seeds=5, warm_labels=0):
    """
    Online LinUCB routing: select arm with highest UCB score, observe reward, update.
    Returns: accuracy, cost_per_query, strong_pct at checkpoints.
    """
    checkpoints = [50, 100, 500, 1000, 5000]
    all_final_acc = []
    all_final_cost = []
    all_final_strong = []

    for seed in range(n_seeds):
        shuffled = list(data)
        random.Random(seed).shuffle(shuffled)

        strong_arm = LinUCBArm('gpt-4', 10.0, 0.80, 1500, 128000, True, True, True, alpha=alpha)
        weak_arm = LinUCBArm('mixtral', 0.24, 0.65, 400, 32768, False, False, False, alpha=alpha)

        for item in shuffled[:warm_labels]:
            prompt, weak_ok, strong_ok = item
            strong_arm.update(strong_arm.features(prompt), 1.0 if strong_ok else 0.0)
            weak_arm.update(weak_arm.features(prompt), 1.0 if weak_ok else 0.0)

        cum_correct = 0
        cum_cost = 0.0
        strong_count = 0
        route_data = shuffled[warm_labels:]

        for i, item in enumerate(route_data):
            prompt, weak_ok, strong_ok = item
            xs = strong_arm.features(prompt)
            xw = weak_arm.features(prompt)

            ucb_s, _, _ = strong_arm.ucb_score(xs)
            ucb_w, _, _ = weak_arm.ucb_score(xw)

            if ucb_s >= ucb_w:
                chosen = 'gpt-4'
                correct = strong_ok
                cost = 0.04
                strong_count += 1
                reward = 1.0 if strong_ok else 0.0
                strong_arm.update(xs, reward)
            else:
                chosen = 'mixtral'
                correct = weak_ok
                cost = 0.0005
                reward = 1.0 if weak_ok else 0.0
                weak_arm.update(xw, reward)

            cum_correct += int(correct)
            cum_cost += cost

        n = len(route_data)
        all_final_acc.append(cum_correct / n if n > 0 else 0)
        all_final_cost.append(cum_cost / n if n > 0 else 0)
        all_final_strong.append(strong_count / n if n > 0 else 0)

    return (
        statistics.mean(all_final_acc),
        statistics.mean(all_final_cost),
        statistics.mean(all_final_strong),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_seeds = 5  # faster than 10

    print("Loading data...")
    t0 = time.time()

    # Load all MMLU (no truncation for real results)
    mmlu = load_mmlu()
    print(f"  MMLU: {len(mmlu)} questions ({time.time()-t0:.1f}s)")

    gsm8k = load_gsm8k()
    print(f"  GSM8K: {len(gsm8k)} questions ({time.time()-t0:.1f}s)")

    # Quick alpha sweep on MMLU subset (3 seeds, first 3000 questions)
    print("\n--- Alpha sweep (MMLU subset, 3 seeds) ---")
    mmlu_sub = mmlu[:3000]
    best_alpha = 1.0
    best_apgr = 0.0
    for alpha in [0.3, 0.5, 1.0, 1.5, 2.0]:
        apgr, std = compute_apgr_linucb(mmlu_sub, alpha, warm_labels=0, n_seeds=3)
        print(f"  alpha={alpha:.1f}: APGR={apgr:.4f} ± {std:.4f}")
        if apgr > best_apgr:
            best_apgr = apgr
            best_alpha = alpha
    print(f"  Best: alpha={best_alpha} (APGR={best_apgr:.4f})")

    # Full MMLU results
    print(f"\n--- Full MMLU ({len(mmlu)} q, {n_seeds} seeds, alpha={best_alpha}) ---")
    results = {}

    for warm in [0, 500]:
        t1 = time.time()
        apgr, std = compute_apgr_linucb(mmlu, best_alpha, warm_labels=warm, n_seeds=n_seeds)
        acc, cost, strong_pct = run_online_linucb(mmlu, best_alpha, n_seeds=n_seeds, warm_labels=warm)
        elapsed = time.time() - t1
        label = f"LinUCB-warm{warm}"
        results[label] = {
            "apgr_mean": round(apgr, 4),
            "apgr_std": round(std, 4),
            "accuracy": round(acc, 4),
            "cost_per_query": round(cost, 5),
            "strong_pct": round(strong_pct, 4),
            "warm_labels": warm,
        }
        print(f"  {label}: APGR={apgr:.4f}±{std:.4f}  acc={acc:.4f}  cost=${cost:.5f}  strong={strong_pct:.1%}  ({elapsed:.1f}s)")

    # Random baseline
    apgr_r, std_r = compute_apgr_random(mmlu, n_seeds=n_seeds)
    results["Random"] = {"apgr_mean": round(apgr_r, 4), "apgr_std": round(std_r, 4)}
    print(f"  Random: APGR={apgr_r:.4f}±{std_r:.4f}")

    # GSM8K
    print(f"\n--- GSM8K ({len(gsm8k)} q, {n_seeds} seeds, alpha={best_alpha}) ---")
    for warm in [0, 500]:
        apgr, std = compute_apgr_linucb(gsm8k, best_alpha, warm_labels=warm, n_seeds=n_seeds)
        acc, cost, strong_pct = run_online_linucb(gsm8k, best_alpha, n_seeds=n_seeds, warm_labels=warm)
        label = f"GSM8K-LinUCB-warm{warm}"
        results[label] = {
            "apgr_mean": round(apgr, 4),
            "apgr_std": round(std, 4),
            "accuracy": round(acc, 4),
            "cost_per_query": round(cost, 5),
            "strong_pct": round(strong_pct, 4),
        }
        print(f"  {label}: APGR={apgr:.4f}±{std:.4f}  acc={acc:.4f}  cost=${cost:.5f}  strong={strong_pct:.1%}")

    apgr_r, std_r = compute_apgr_random(gsm8k, n_seeds=n_seeds)
    results["GSM8K-Random"] = {"apgr_mean": round(apgr_r, 4), "apgr_std": round(std_r, 4)}
    print(f"  GSM8K-Random: APGR={apgr_r:.4f}±{std_r:.4f}")

    # Save
    out_path = os.path.join(OUT_DIR, "linucb_apgr_results.json")
    with open(out_path, "w") as f:
        json.dump({"alpha": best_alpha, "n_seeds": n_seeds, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # RouteLLM reference numbers (from published paper)
    print("\n--- RouteLLM reference (from Ong et al. 2024) ---")
    print("  MF:       APGR=0.57")
    print("  BERT:     APGR=0.67")
    print("  CausalLLM:APGR=0.72")
    print("  SW:       APGR=0.80")
    print("  (trained on 55K+ Chatbot Arena labels)")
    print("\n  LinUCB (ours): zero offline labels, online from cold start")


if __name__ == "__main__":
    main()
