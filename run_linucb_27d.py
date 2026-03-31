#!/usr/bin/env python3
"""
LinUCB benchmark using the EXACT 27-dim feature extractor from routesmith/predictor/features.py.
Produces APGR numbers consistent with the paper's feature space description.

Feature layout (matches routesmith FeatureExtractor exactly):
  [0-10]  : 11 original message features
  [11-16] : 6 extended message features (query type + difficulty)
  [17-24] : 8 model features
  [25-26] : 2 interaction features
  Total   : 27 dims
"""

import csv, json, math, os, random, re, statistics

# ── Keyword sets (from routesmith/predictor/features.py) ──────────────────────
_MATH_WORDS = frozenset([
    "calculate","compute","solve","equation","formula","derivative","integral",
    "matrix","vector","probability","statistics","algebra","geometry","calculus",
    "theorem","proof","inequality","function","variable","coefficient","polynomial",
])
_REASONING_WORDS = frozenset([
    "why","because","therefore","hence","implies","conclude","infer","deduce",
    "analyze","evaluate","compare","contrast","assess","justify","explain",
    "hypothesis","argument","evidence","logical","causal","reason","think",
])
_CODE_WORDS = frozenset([
    "code","function","implement","algorithm","bug","error","debug","compile",
    "syntax","api","database","sql","python","javascript","rust","java",
    "typescript","html","css","git","docker","kubernetes",
])
_CREATIVE_WORDS = frozenset([
    "write","story","poem","creative","imagine","fiction","character","narrative",
    "dialogue","scene","metaphor","lyric","compose","draft","essay","blog",
    "article","describe",
])
_RE_CODE_BLOCK = re.compile(r"```")

def _kw_overlap(words, target):
    if not words: return 0.0
    return min(1.0, len(words & target) / len(words) * 10)

# ── Model specs ────────────────────────────────────────────────────────────────
class _Spec:
    def __init__(self, cost_in, cost_out, quality, latency, ctx, fc, vis, json_):
        self.cost_per_1k_input = cost_in
        self.cost_per_1k_output = cost_out
        self.quality_score = quality
        self.latency_p50_ms = latency
        self.context_window = ctx
        self.supports_function_calling = fc
        self.supports_vision = vis
        self.supports_json_mode = json_
        self.cost_per_1k_total = cost_in + cost_out

GPT4  = _Spec(0.01, 0.03,  0.806, 1500, 128000, True,  True,  True)
MIXTRAL = _Spec(0.00024, 0.00024, 0.681, 400, 32768, False, False, False)
MODELS = {"gpt4": GPT4, "mixtral": MIXTRAL}

def extract_features_27d(prompt: str, model_id: str) -> list:
    """Exact 27-dim replication of routesmith FeatureExtractor."""
    messages = [{"role": "user", "content": prompt}]
    spec = MODELS[model_id]

    # ── Message features (17 dims) ──
    msg_count = 1.0
    lengths = [len(m.get("content","")) for m in messages]
    total_char = float(sum(lengths))
    avg_msg = total_char / msg_count
    max_msg = float(max(lengths))
    user_msg_count = 1.0
    system_present = 0.0
    last_content = prompt
    last_len = float(len(last_content))
    q_marks = float(last_content.count("?"))
    words = last_content.split()
    word_count = float(len(words))
    avg_word_len = sum(len(w) for w in words) / word_count if words else 0.0
    tools_present = 0.0

    all_text = prompt
    all_words_lower = set(all_text.lower().split())
    math_score = _kw_overlap(all_words_lower, _MATH_WORDS)
    reasoning_score = _kw_overlap(all_words_lower, _REASONING_WORDS)
    code_score = _kw_overlap(all_words_lower, _CODE_WORDS)
    creative_score = _kw_overlap(all_words_lower, _CREATIVE_WORDS)
    has_code = bool(_RE_CODE_BLOCK.search(all_text)) or any(
        k in all_text for k in ("def ","function ","class "))
    if has_code: code_score = min(1.0, code_score + 0.3)

    length_signal = min(1.0, len(words) / 200.0)
    vocab_signal = min(1.0, avg_word_len / 8.0)
    structure_signal = (0.3 if has_code else 0.0) + 0.2*math_score + 0.2*reasoning_score
    difficulty = min(1.0, 0.3*length_signal + 0.25*vocab_signal +
                     0.35*structure_signal + 0.1*min(1.0, q_marks/3.0))
    vocab_richness = len(set(w.lower() for w in words)) / max(1, len(words))

    msg_feats = [msg_count, total_char, avg_msg, max_msg, user_msg_count,
                 system_present, last_len, q_marks, word_count, avg_word_len,
                 tools_present, math_score, reasoning_score, code_score,
                 creative_score, difficulty, vocab_richness]

    # ── Model features (8 dims) ──
    model_feats = [
        spec.cost_per_1k_input, spec.cost_per_1k_output,
        spec.quality_score, spec.latency_p50_ms,
        math.log(max(spec.context_window, 1)),
        1.0 if spec.supports_function_calling else 0.0,
        1.0 if spec.supports_vision else 0.0,
        1.0 if spec.supports_json_mode else 0.0,
    ]

    # ── Interaction features (2 dims) ──
    est_resp_tokens = min(1.0, (word_count*2 + difficulty*500) / 2000.0)
    diff_x_quality = difficulty * spec.quality_score
    inter_feats = [est_resp_tokens, diff_x_quality]

    feats = msg_feats + model_feats + inter_feats
    assert len(feats) == 27, f"Expected 27, got {len(feats)}"

    # L2 normalize
    norm = math.sqrt(sum(v*v for v in feats))
    if norm > 0: feats = [v/norm for v in feats]
    return feats

# ── LinUCB (same as run_linucb_fast.py but 27-dim) ─────────────────────────────
D = 27

class LinUCBArm:
    def __init__(self, alpha=1.0):
        self.A_inv = [[1.0 if i==j else 0.0 for j in range(D)] for i in range(D)]
        self.b = [0.0]*D
        self.alpha = alpha
        self.n = 0

    def score(self, x):
        Ax = [sum(self.A_inv[i][j]*x[j] for j in range(D)) for i in range(D)]
        theta_x = sum(Ax[i]*x[i] for i in range(D))
        xAx = sum(x[i]*Ax[i] for i in range(D))
        return theta_x + self.alpha * math.sqrt(max(0.0, xAx))

    def update(self, x, r):
        Ax = [sum(self.A_inv[i][j]*x[j] for j in range(D)) for i in range(D)]
        denom = 1.0 + sum(x[j]*Ax[j] for j in range(D))
        for i in range(D):
            for j in range(D):
                self.A_inv[i][j] -= Ax[i]*Ax[j]/denom
        for i in range(D): self.b[i] += r*x[i]
        self.n += 1

# ── Data loading ───────────────────────────────────────────────────────────────
RD = "/workspace/group/routesmith-research/routellm"

def load_mmlu():
    d = os.path.join(RD,"routellm","evals","mmlu","responses")
    data = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"): continue
        with open(os.path.join(d,f)) as fp:
            for row in csv.DictReader(fp):
                try:
                    s = 1 if row["gpt-4-1106-preview"].strip().lower() in ("true","1") else 0
                    w = 1 if row["mistralai/Mixtral-8x7B-Instruct-v0.1"].strip().lower() in ("true","1") else 0
                    data.append((row["prompt"], s, w))
                except: pass
    return data

def load_gsm8k():
    path = os.path.join(RD,"routellm","evals","gsm8k","gsm8k_responses.csv")
    data = []
    with open(path) as fp:
        for row in csv.DictReader(fp):
            try:
                s = 1 if row["gpt-4-1106-preview"].strip().lower() in ("true","1") else 0
                w = 1 if row["mistralai/Mixtral-8x7B-Instruct-v0.1"].strip().lower() in ("true","1") else 0
                data.append((row["prompt"], s, w))
            except: pass
    return data

# ── APGR ──────────────────────────────────────────────────────────────────────
def compute_apgr(data, n_seeds, alpha, warm_labels, cost_lambda=0.0):
    """Online LinUCB APGR with exact 27-dim features."""
    max_cost = GPT4.cost_per_1k_total
    results = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        d = list(data); rng.shuffle(d)
        strong = LinUCBArm(alpha)
        weak   = LinUCBArm(alpha)

        # warm start
        if warm_labels > 0:
            warm = d[:warm_labels]
            for prompt,s_ok,w_ok in warm:
                xs = extract_features_27d(prompt, "gpt4")
                xw = extract_features_27d(prompt, "mixtral")
                rs = (1.0 if s_ok else 0.0) - cost_lambda*(GPT4.cost_per_1k_total/max_cost)
                rw = (1.0 if w_ok else 0.0) - cost_lambda*(MIXTRAL.cost_per_1k_total/max_cost)
                strong.update(xs, rs); weak.update(xw, rw)
            eval_data = d[warm_labels:]
        else:
            eval_data = d

        # APGR: score test data using warm-trained model only (no online updates on test set)
        # This matches run_linucb_fast.py methodology: train on warm set, score test set, compute APGR.
        # The expected reward (theta·x, no exploration bonus) is used for ranking preference.
        def expected_reward(arm, x):
            """theta·x = (A_inv @ b) · x"""
            theta = [sum(arm.A_inv[i][j]*arm.b[j] for j in range(D)) for i in range(D)]
            return sum(theta[i]*x[i] for i in range(D))

        scored = [(expected_reward(strong, extract_features_27d(p,"gpt4")) -
                   expected_reward(weak,   extract_features_27d(p,"mixtral")),
                   float(s), float(w))
                  for p,s,w in eval_data]
        scored.sort(key=lambda x: -x[0])
        n = len(scored)

        baseline_q = sum(x[2] for x in scored)   # total correct if all → weak
        max_q      = sum(x[1] for x in scored)   # total correct if all → strong
        total_gap  = max_q - baseline_q
        if total_gap <= 0: results.append(0.5); continue

        cum_q = baseline_q
        area  = 0.0
        prev_x, prev_y = 0.0, 0.0
        for i, (pref, qs, qw) in enumerate(scored):
            cum_q += (qs - qw)          # swap query i from weak→strong
            x = (i + 1) / n
            y = (cum_q - baseline_q) / total_gap   # PGR at threshold k
            area += (x - prev_x) * (prev_y + y) / 2  # trapezoidal
            prev_x, prev_y = x, y
        results.append(area)

    return statistics.mean(results), statistics.stdev(results) if len(results)>1 else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    mmlu = load_mmlu()
    gsm8k = load_gsm8k()
    print(f"  MMLU={len(mmlu)} GSM8K={len(gsm8k)}")

    N_SEEDS = 5
    out = {}

    for dataset_name, data in [("MMLU", mmlu), ("GSM8K", gsm8k)]:
        print(f"\n{dataset_name} ({len(data)} questions)")
        for warm in [0, 500]:
            mean, std = compute_apgr(data, N_SEEDS, alpha=1.0, warm_labels=warm)
            key = f"LinUCB-27d-warm{warm}"
            out[f"{dataset_name}-{key}"] = {"apgr_mean": round(mean,4), "apgr_std": round(std,4), "warm_labels": warm}
            print(f"  warm={warm:4d}: APGR={mean:.4f} ± {std:.4f}")

    # Also random baseline
    for dataset_name, data in [("MMLU", mmlu), ("GSM8K", gsm8k)]:
        r_results = []
        for seed in range(N_SEEDS):
            rng = random.Random(seed+100)
            d = list(data); rng.shuffle(d)
            aw = sum(w for _,_,w in d)/len(d)
            as_ = sum(s for _,s,_ in d)/len(d)
            gap = as_ - aw
            correct = sum(rng.choice([s,w]) for _,s,w in d) / len(d)
            pgr = (correct - aw)/gap if gap>0 else 0.5
            r_results.append(pgr)
        out[f"{dataset_name}-Random"] = {"apgr_mean": round(statistics.mean(r_results),4)}

    os.makedirs("/workspace/group/routesmith-research/sweep_results", exist_ok=True)
    with open("/workspace/group/routesmith-research/sweep_results/linucb_27d_results.json","w") as f:
        json.dump({"feature_dim": 27, "n_seeds": N_SEEDS, "results": out}, f, indent=2)

    print("\n=== Summary ===")
    for k,v in out.items():
        print(f"  {k}: {v['apgr_mean']:.4f} ± {v.get('apgr_std',0):.4f}")
    print("\nSaved to sweep_results/linucb_27d_results.json")
