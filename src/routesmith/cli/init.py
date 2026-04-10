# src/routesmith/cli/init.py
"""routesmith init — interactive model selection wizard.

Fetches the OpenRouter model catalog with live pricing, lets the user
select 3–10 models via questionary checkbox (arrow keys + type-to-filter),
then writes routesmith.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from routesmith.registry.openrouter import OpenRouterModel, fetch_models, model_to_yaml_block


def run_init(args: Any) -> int:
    out_path = Path(args.output)

    if out_path.exists() and not args.force:
        print(f"'{out_path}' already exists. Use --force to overwrite.")
        return 1

    print("Fetching model catalog from OpenRouter...")
    try:
        models = fetch_models()
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        return 1

    print(f"Found {len(models)} models with pricing.\n")

    selected = _select_models(models)
    if selected is None:
        print("Aborted.")
        return 1

    predictor = _select_predictor()
    yaml_text = _render_yaml(selected, predictor)

    out_path.write_text(yaml_text)
    print(f"\nWrote {out_path}")
    print("\nNext steps:")
    print(f"  export OPENROUTER_API_KEY=sk-or-...")
    print(f"  routesmith serve --config {out_path}")
    return 0


def _select_models(models: list[OpenRouterModel]) -> list[OpenRouterModel] | None:
    """Questionary checkbox — arrow keys to navigate, type to filter, space to select."""
    try:
        import questionary
        from questionary import Choice
    except ImportError:
        print("questionary not installed. Run: pip install 'routesmith[proxy]'")
        return _fallback_select(models)

    # Build choices: value = index, title = formatted label
    # questionary filters on the title string, so embed cost there
    choices = [
        Choice(
            title=f"{m.id:<52}  ${m.cost_per_1k_input:.4f}/1k in   ${m.cost_per_1k_output:.4f}/1k out",
            value=i,
        )
        for i, m in enumerate(models)
    ]

    print("  Arrow keys to navigate  ·  Space to select  ·  Type to filter  ·  Enter to confirm\n")

    result = questionary.checkbox(
        "Select candidate models (3–10):",
        choices=choices,
        validate=lambda sel: (
            True if 3 <= len(sel) <= 10
            else "Select between 3 and 10 models"
        ),
        instruction="(type to search, space to pick, enter when done)",
        style=questionary.Style([
            ("checkbox-selected", "fg:#2ca02c bold"),   # green for selected
            ("highlighted",       "fg:#1f77b4 bold"),   # blue for cursor
            ("instruction",       "fg:#7f7f7f italic"),
        ]),
    ).ask()

    if result is None:
        return None

    return [models[i] for i in result]


def _select_predictor() -> str:
    try:
        import questionary
    except ImportError:
        return _fallback_predictor()

    choice = questionary.select(
        "Routing algorithm:",
        choices=[
            questionary.Choice(
                "LinTS-27d  — Thompson Sampling, no hyperparameter  (recommended)",
                value="lints",
            ),
            questionary.Choice(
                "LinUCB-27d — UCB exploration, higher accuracy, needs alpha tuning",
                value="linucb",
            ),
            questionary.Choice(
                "Embedding  — static quality priors, no online learning",
                value="embedding",
            ),
        ],
        style=questionary.Style([
            ("selected",     "fg:#2ca02c bold"),
            ("highlighted",  "fg:#1f77b4 bold"),
        ]),
    ).ask()

    return choice or "lints"


# ── Fallback (no questionary) ─────────────────────────────────────────────────

def _fallback_select(models: list[OpenRouterModel]) -> list[OpenRouterModel] | None:
    """Minimal numbered list fallback when questionary is unavailable."""
    PAGE = 25
    offset = 0
    chosen: set[int] = set()

    while True:
        page = models[offset: offset + PAGE]
        total = len(models)
        print(f"\n{'─'*70}")
        print(f"  {'#':>3}  {'Model ID':<44}  {'$/1k in':>8}")
        print(f"{'─'*70}")
        for i, m in enumerate(page, 1):
            idx = offset + i - 1
            mark = "✓" if idx in chosen else " "
            mid = m.id[:44] if len(m.id) <= 44 else m.id[:41] + "..."
            print(f"  {i:>3} {mark}  {mid:<44}  ${m.cost_per_1k_input:.4f}")
        print(f"{'─'*70}")
        print(f"  Page {offset//PAGE+1}/{(total+PAGE-1)//PAGE}  |  {len(chosen)} selected")

        try:
            raw = input("\nToggle numbers / n=next / p=prev / done / q=quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw in ("q", "quit"):
            return None
        if raw in ("done", "d", ""):
            if 3 <= len(chosen) <= 10:
                return [models[i] for i in sorted(chosen)]
            print(f"  Select 3–10 models (have {len(chosen)}).")
            continue
        if raw == "n" and offset + PAGE < total:
            offset += PAGE
        elif raw == "p" and offset >= PAGE:
            offset -= PAGE
        else:
            for tok in raw.replace(",", " ").split():
                try:
                    n = int(tok)
                    idx = offset + n - 1
                    if 0 <= idx < total:
                        chosen.discard(idx) if idx in chosen else chosen.add(idx)
                except ValueError:
                    pass

    return None


def _fallback_predictor() -> str:
    print("\n1. lints (recommended)  2. linucb  3. embedding")
    try:
        c = input("Select [1/2/3, default=1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "lints"
    return {"2": "linucb", "3": "embedding"}.get(c, "lints")


# ── YAML renderer ─────────────────────────────────────────────────────────────

def _render_yaml(models: list[OpenRouterModel], predictor: str) -> str:
    model_blocks = "\n".join(model_to_yaml_block(m) for m in models)

    predictor_extra = ""
    if predictor == "linucb":
        predictor_extra = "\n  linucb_alpha: 1.5  # exploration parameter (0.5–3.0)"
    elif predictor == "lints":
        predictor_extra = "\n  lints_v_sq: 1.0   # posterior variance scaling"

    return f"""\
# routesmith.yaml — generated by `routesmith init`
# Edit freely. Run `routesmith serve` to start the proxy.

routing:
  strategy: direct
  predictor: {predictor}{predictor_extra}

budget:
  max_cost_per_request: 0.10
  max_cost_per_day: 50.0
  quality_threshold: 0.75

cache:
  enabled: false
  similarity_threshold: 0.95

feedback:
  enabled: true
  sample_rate: 0.1

models:
{model_blocks}
"""
