# RouteSmith v0.8.0 Launch Kit

Ready-to-post content for the v0.8.0 release push. Every number below comes from
the paper (`paper/main.pdf`) or the repo — nothing invented. Edit voice/tone to
taste before posting.

**Core facts (source of truth for all posts):**

- PyPI: `pip install routesmith-llm` — first published release (0.8.0)
- Zero pretraining labels; learns online from ~100 queries (RouteLLM/Not Diamond need 55K+)
- 5-arm multi-model routing (GPT-4o, Claude Sonnet 4.5, Qwen-Plus, MiniMax-M1, DeepSeek-V3): **71.0% accuracy at 45% cost savings** (LinTS)
- Binary routing: LinTS-27d achieves APGR 0.593 with **46% MMLU cost reduction** vs Always-Strong
- Sub-millisecond routing overhead (<0.5ms P99, 5-arm deployment)
- Self-hosted, open source — prompts never transit a third-party router
- Works as a local proxy for Claude Code, Codex, OpenCode, any OpenAI- or Anthropic-SDK client

---

## 1. GitHub Release notes (paste when tagging `v0.8.0`)

**Title:** `v0.8.0 — RouteSmith is on PyPI`

```markdown
RouteSmith is now pip-installable:

    pip install "routesmith-llm[proxy]"
    routesmith init
    routesmith serve
    # → OpenAI-compatible proxy at http://localhost:9119/v1

(`routesmith` was taken on PyPI, so the package is `routesmith-llm`.
The import name is unchanged: `import routesmith`.)

## Highlights

- **Anthropic-native endpoint** — `POST /v1/messages` with streaming.
  Set `ANTHROPIC_BASE_URL=http://localhost:9119` and any Anthropic SDK
  client routes through RouteSmith.
- **Agent-framework examples** — OpenAI SDK, Pydantic AI, LlamaIndex,
  LangChain, DSPy, CrewAI, AutoGen, with a CI smoke harness.
- **Quickstart CLI** — `routesmith init` interactive setup.
- **Proxy auth** — `--api-key` flag.
- **Per-project cost stats, decision audit log, per-role policy CLI.**
- **Research paper** — contextual-bandit routing evaluated with real API
  calls: 71% accuracy at 45% cost savings on 5-arm multi-model routing,
  zero pretraining labels. PDF in `paper/`.

Full details in [CHANGELOG.md](CHANGELOG.md).
```

---

## 2. Show HN (Hacker News)

**Title:**
`Show HN: RouteSmith – self-hosted LLM router that learns from your own traffic`

**Body:**

```
Hi HN — I built RouteSmith, an open-source router that sits between your
AI coding tool (Claude Code, Codex, OpenCode, any OpenAI/Anthropic SDK
client) and the LLM, and picks the cheapest model that can handle each
request.

The interesting part is *how* it picks. Existing routers (RouteLLM,
Not Diamond, OpenRouter's auto mode) are supervised classifiers trained
once on 55K+ preference labels — static, binary (strong vs weak model),
and trained on everyone's traffic, not yours. RouteSmith frames routing
as a contextual bandit (LinUCB / Linear Thompson Sampling over a 27-dim
feature space): zero pretraining labels, learns online from your own
feedback, scales past two models, and adds <0.5ms routing overhead.

In our experiments (real API calls, MMLU + GSM8K), 5-arm routing across
GPT-4o, Claude Sonnet 4.5, Qwen-Plus, MiniMax-M1, and DeepSeek-V3 hit
71% accuracy at 45% cost savings, converging after ~100 queries. Paper
with full results and ablations is in the repo.

It's self-hosted, so prompts never transit a third-party router, and you
get budget caps, per-project cost tracking, and a full audit log of every
routing decision.

    pip install "routesmith-llm[proxy]"
    routesmith init && routesmith serve

Repo: https://github.com/yunpengl9071/routesmith

Happy to answer questions about the bandit formulation, the feature
space, or where it falls down (GSM8K is honest in the paper — aggressive
routing can waste cost there).
```

**Timing:** Tuesday–Thursday, 8–10am ET. Stay in the thread all day answering
comments — Show HN lives or dies on author engagement.

---

## 3. X / Twitter thread

```
1/ Your AI coding assistant sends "fix this typo" to the same frontier
model as "refactor this architecture." That's 10–80x overpaying on most
requests.

We built RouteSmith: an open-source, self-hosted router that learns
which model each request actually needs. Now on PyPI. 🧵

2/ Existing routers (RouteLLM, Not Diamond) are static classifiers:
trained once on 55K+ labels, binary strong-vs-weak only, and tuned to
everyone's traffic — not yours.

RouteSmith treats routing as a contextual bandit. Zero pretraining.
It learns online, from YOUR traffic, in ~100 queries.

3/ Results (real API calls, MMLU + GSM8K):

Routing across 5 models — GPT-4o, Claude Sonnet 4.5, Qwen-Plus,
MiniMax-M1, DeepSeek-V3 — Linear Thompson Sampling hits 71% accuracy
at 45% cost savings. Routing overhead: <0.5ms P99.

Full paper with ablations in the repo.

4/ It's a local proxy, so it works with tools you already use:

Claude Code, Codex, OpenCode → point OPENAI_BASE_URL at localhost
Anthropic SDK → ANTHROPIC_BASE_URL, native /v1/messages endpoint

pip install "routesmith-llm[proxy]"
routesmith init && routesmith serve

5/ Self-hosted means your prompts never transit a third-party router.
Plus: hard budget caps, per-project cost tracking, and an audit log
showing exactly why every request went where it did.

6/ Open source, Python 3.10+, works with any provider (one OpenRouter
key gets you 400+ models).

⭐ https://github.com/yunpengl9071/routesmith
📦 https://pypi.org/project/routesmith-llm/
```

---

## 4. LinkedIn post

```
We just shipped RouteSmith v0.8.0 — an open-source, self-hosted LLM
router — to PyPI.

The problem: teams route every request to a frontier model "to be safe,"
paying 10–80x more per token than necessary on requests a smaller model
handles fine. Existing routers are static classifiers that need 55K+
training labels and only choose between two models.

Our approach: treat routing as a contextual bandit. RouteSmith learns
online from your own production traffic — no pretraining labels, any
number of models, sub-millisecond overhead. In experiments with real API
calls, routing across five models (GPT-4o, Claude Sonnet 4.5, Qwen-Plus,
MiniMax-M1, DeepSeek-V3) reached 71% accuracy at 45% cost savings,
converging after roughly 100 queries.

Because it's self-hosted, prompts never leave your infrastructure — and
you get budget enforcement, per-project cost allocation, and a complete
audit trail of every routing decision, which matters if you're running
LLMs under compliance constraints.

It drops in as a local proxy for Claude Code, Codex, OpenCode, and any
OpenAI- or Anthropic-SDK application.

pip install "routesmith-llm[proxy]"

Repo (research paper included): https://github.com/yunpengl9071/routesmith
```

---

## 5. Reddit — r/LocalLLaMA

**Title:**
`RouteSmith: self-hosted router that learns which model each query needs (contextual bandits, no pretraining) — now on PyPI`

**Body:**

```
Open-sourced a router I've been building. It sits between your client
(Claude Code / Codex / OpenCode / any OpenAI or Anthropic SDK app) and
your models, and learns online which model each query actually needs.

What makes it different from RouteLLM / OpenRouter auto:

- Contextual bandits (LinUCB / Linear Thompson Sampling), not a
  pretrained classifier. Zero labels needed — it converges from your own
  traffic in ~100 queries.
- N models, not just strong-vs-weak binary. Tested with 5 arms.
- Fully self-hosted: prompts never transit a third-party router. Works
  with local models too — register any endpoint with a cost and quality
  prior.
- Budget caps (per-project, per-request), audit log of every decision.

Benchmarks (real API calls, MMLU+GSM8K, paper in repo): 5-arm routing
across GPT-4o / Claude Sonnet 4.5 / Qwen-Plus / MiniMax-M1 / DeepSeek-V3
→ 71% accuracy at 45% cost savings. <0.5ms P99 routing overhead.

Where it's weak (honest): on GSM8K, aggressive cost-optimization can
route too cheap and waste money on wrong answers — details and ablations
in the paper.

pip install "routesmith-llm[proxy]"
GitHub: https://github.com/yunpengl9071/routesmith

Feedback welcome, especially from anyone routing across local + hosted
models.
```

Also consider: r/MachineLearning ([P] tag, lead with the bandit
formulation and paper), r/ChatGPTCoding, r/ClaudeAI (lead with the
Claude Code proxy integration).

---

## 6. Launch checklist

**Before posting anything:**
- [ ] Tag `v0.8.0` and publish the GitHub Release (notes above) — posts should link to a release, not a bare repo
- [ ] Verify `pip install "routesmith-llm[proxy]" && routesmith init && routesmith serve` works in a fresh venv (this exact snippet is in every post)
- [ ] README renders correctly on the PyPI page

**Sequence (spread over ~1 week, don't blast everything at once):**
1. GitHub Release + X thread (day 1)
2. Show HN (day 2–3, Tue–Thu morning ET; be present in comments all day)
3. r/LocalLLaMA (day 4 — mention the HN discussion if it went well)
4. LinkedIn (day 5)
5. Submit to newsletters: TLDR AI, Latent Space, Python Weekly

**Ongoing visibility (higher ROI than any single post):**
- Add topics to the GitHub repo: `llm`, `llm-routing`, `contextual-bandits`, `claude-code`, `openrouter`, `cost-optimization`, `proxy`
- Put the paper on arXiv and link it from the README — it's the most defensible differentiator vs. every other router
- Short demo GIF/asciinema of `routesmith init → serve → routing decisions appearing in the audit log` at the top of the README
- Answer "which model should I use / how do I cut LLM costs" questions on HN/Reddit/Discord with genuinely useful answers; link RouteSmith only when directly relevant
