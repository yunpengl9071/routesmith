# RouteSmith Adoption & Growth Strategy

**Date**: 2026-06-17
**Status**: Draft — for validation
**Context**: All four implementation phases complete, v0.3.0 tagged, 639 tests passing. Product is ready. Now: adoption.

---

## Position

RouteSmith is an invisible quality-cost layer that makes any LLM-powered application smarter about model selection. It sits between your code and the LLM, routes each query to the right model, and you never think about model IDs again.

Two audiences, two stories:

| User | Pain | RouteSmith's answer |
|------|------|---------------------|
| Pays for APIs | $50-250/day bills | "Cut AI coding costs 40-60% with zero quality loss" |
| Uses free models | Inconsistent output | "Get paid-model quality from free models" |

Both converge on the same product experience: `pip install routesmith && routesmith serve`.

---

## The "One Command" Entry Point

```bash
pip install routesmith
routesmith serve
# → OpenAI-compatible proxy at http://localhost:8787/v1
```

Every AI coding tool already supports OpenAI-compatible endpoints. One URL, one setup, everything works.

**Why this matters for coding assistants:**

1. **Mixed workloads = natural routing target.** Typos, completions, refactors, architecture — coding assistants generate all difficulty levels. RouteSmith sends each to the appropriate model tier.
2. **Token bills compound silently.** A heavy Claude Code day can hit $50. OpenClaw multi-agent workflows burn 5-10x more. RouteSmith's budget enforcement + cascade routing catches this.
3. **Self-hosted matches the audience.** Coding tool users are technical, privacy-aware, and skeptical of vendor SaaS. A local proxy they control is the right trust model.
4. **The window is open now.** Claude Code GA, Codex growing, pi and OpenCode have active communities, OpenClaw is new with multi-agent orchestration. Be the default smart router for this category.

---

## Distribution Channels

### Channel 1: Platform-specific integration guides

One guide per platform. Same proxy, different config snippet.

| Platform | Integration point | Guide content |
|----------|-------------------|---------------|
| Claude Code | `.claude.json` `apiKeyHelper` | 3-line config to route through the proxy |
| Codex | OpenAI base URL override | Same pattern, Codex-specific YAML |
| pi | Provider config entry | OpenClaw-compatible provider (already tested) |
| OpenClaw | `routesmith openclaw-config` | Already built — ship this first |
| OpenCode | Provider config | Generic OpenAI-compatible setup |

**OpenClaw first** because it's already smoke-tested end-to-end, its multi-agent users feel the cost pain most acutely, and the `openclaw-config` CLI command is built and working.

### Channel 2: Social proof flywheel

Get RouteSmith into 5-10 power users (8+ hours/day of AI coding). Help them set it up in 5 minutes. Ask them to tweet their before/after bills. One good thread on X or HN hits the entire ecosystem.

### Channel 3: Awesome-list presence

Submit to `awesome-claude-code`, `awesome-ai-coding`, `awesome-llm-tools`. These lists are surprisingly high-signal discovery channels for this audience.

### Channel 4: The OpenRouter free tier combo

OpenRouter gives access to 20+ free models but expects you to pick one. RouteSmith uses all of them intelligently — routing, cascading, caching. The demo: "Here's raw Qwen3-8b on a complex refactor vs. RouteSmith across 5 free models." The gap is dramatic and the setup is zero-configuration.

---

## The "Free Model Quality" Narrative

For users who don't pay for APIs, RouteSmith's value shifts from cost to quality:

> Free models are good individually. But none is great at everything. RouteSmith orchestrates them: hard refactors go to the strongest free model, simple edits to the fastest, and if the first answer looks weak it cascades to a second opinion. The user never picks a model — they just get better results.

This isn't a different product. It's the same proxy, same routing engine, just told from a quality perspective rather than cost.

---

## Implementation Plan

### Phase 1: Ship the proxy story (Week 1-2)

| Task | Effort | Priority |
|------|--------|----------|
| Write `docs/integrations/openclaw.md` guide | 1 day | **P0** — already smoke-tested, fastest win |
| Write `docs/integrations/claude-code.md` guide | 1 day | **P0** — biggest audience |
| Write `docs/integrations/opencode.md` + pi + codex guides | 2 days | **P1** — template from the first two |
| Add `RouteSmith.with_free_models()` factory preset | 1 day | **P1** — enables Funnel B narrative |
| Record 90-second demo video | 1 day | **P1** — proxy start → OpenClaw config → before/after |

### Phase 2: Landing and discovery (Week 3-4)

| Task | Effort | Priority |
|------|--------|----------|
| Redesign README.md with two-funnel narrative | 1 day | **P0** |
| Add "Why RouteSmith" comparison table in docs | 1 day | **P1** |
| Add before/after cost examples with real numbers | 1 day | **P1** |
| Submit to `awesome-claude-code`, `awesome-llm-tools` | 1 day | **P1** |
| Find 5-10 power users for beta + social proof | ongoing | **P0** |

### Phase 3: Native integrations (earned, not assumed)

| Platform | Trigger | Integration |
|----------|---------|-------------|
| Claude Code MCP server | 100+ proxy users from this platform | Inline routing decisions, budget controls |
| pi skill/extension | 50+ pi proxy users | Provider config + stats dashboard |
| OpenClaw provider plugin | 50+ OpenClaw proxy users | Deeper stats, agent-level budgets |

**Only build native integrations for platforms that show traction.** The proxy works everywhere today.

---

## What NOT to Build

- ❌ SaaS dashboard or hosted version — stay self-hosted OSS until there's demand
- ❌ Native platform plugins upfront — let proxy usage data decide priorities
- ❌ Paid support tiers — too early, signals premature commercialization
- ❌ Marketing website — README + docs site serve the same purpose now
- ❌ New enterprise features — v0.3.0's HIPAA/SOC2 tagging, budget enforcement, and multi-project isolation are sufficient

---

## Success Metrics

| Metric | Target | Timeline |
|--------|--------|----------|
| GitHub stars | 500 | Month 3 |
| Proxy users (unique installs) | 200 | Month 3 |
| Integration guide page views | 2,000 | Month 2 |
| Social proof posts from users | 5 | Month 2 |
| OpenClaw users adopting | 30 | Month 3 |
