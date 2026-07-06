"""ProductRouterStrategy — benchmarks the REAL RouteSmith product stack."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

from benchmark.config import PRICING, STRONG_MODEL, WEAK_MODEL, cost_usd
from benchmark.strategies.base import BaseStrategy, call_llm
from benchmark.strategies.linucb import _correct, _max_tokens
from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig, RoutingStrategy


class ProductRouterStrategy(BaseStrategy):
    """
    ProductRouterStrategy — uses the REAL RouteSmith product stack.

    Creates a RouteSmith instance with predictor_type='lints', registers
    the benchmark's STRONG_MODEL and WEAK_MODEL, and delegates routing
    decisions to RouteSmith's Router DIRECT strategy.

    Result dict shape mirrors the other benchmark strategies (lints.py etc.)
    so that existing analysis scripts can read results transparently.
    """

    def __init__(self) -> None:
        config = RouteSmithConfig(predictor_type="lints")
        self.rs = RouteSmith(config=config)

        pricing = PRICING
        self.rs.register_model(
            STRONG_MODEL,
            cost_per_1k_input=pricing[STRONG_MODEL]["input"] / 1000,
            cost_per_1k_output=pricing[STRONG_MODEL]["output"] / 1000,
        )
        self.rs.register_model(
            WEAK_MODEL,
            cost_per_1k_input=pricing[WEAK_MODEL]["input"] / 1000,
            cost_per_1k_output=pricing[WEAK_MODEL]["output"] / 1000,
        )

        self.router = self.rs.router
        self._c_max = cost_usd(STRONG_MODEL, 300, 50)

    @property
    def name(self) -> str:
        return "product_router_lints"

    def route(self, query: dict) -> dict:
        messages = [{"role": "user", "content": query["prompt"]}]

        # Use the product's Router DIRECT strategy (no quality threshold)
        model_id = self.router.route(
            messages=messages,
            strategy=RoutingStrategy.DIRECT,
            min_quality=0.0,
        )
        arm_chosen = 1 if model_id == STRONG_MODEL else 0
        routing_decision = "strong" if arm_chosen == 1 else "weak"
        max_tok = _max_tokens(query.get("dataset", "mmlu"))

        # Always call weak model (for reward signal + baseline comparison)
        weak_resp, weak_pt, weak_ct = call_llm(
            WEAK_MODEL, query["prompt"], max_tokens=max_tok
        )
        weak_correct = _correct(query, weak_resp)
        weak_cost = cost_usd(WEAK_MODEL, weak_pt, weak_ct)

        if routing_decision == "strong":
            strong_resp, strong_pt, strong_ct = call_llm(
                STRONG_MODEL, query["prompt"], max_tokens=max_tok
            )
            strong_correct = _correct(query, strong_resp)
            strong_cost = cost_usd(STRONG_MODEL, strong_pt, strong_ct)
            final_correct = strong_correct
            final_model = STRONG_MODEL
            total_cost = weak_cost + strong_cost
        else:
            final_correct = weak_correct
            final_model = WEAK_MODEL
            total_cost = weak_cost
            strong_cost = 0.0

        reward = float(final_correct) - 0.15 * (total_cost / max(self._c_max, 1e-9))

        # Update product's predictor (uses LinTSPredictor.update internally)
        self.router.predictor.update(
            messages=messages,
            model_id=model_id,
            actual_quality=reward,
        )

        return {
            "query_id": query["query_id"],
            "dataset": query.get("dataset", ""),
            "category": query.get("category", ""),
            "strategy": self.name,
            "arm_chosen": arm_chosen,
            "routing_decision": routing_decision,
            "model": final_model,
            "final_model": final_model,
            "weak_correct": weak_correct,
            "correct": final_correct,
            "cost_usd": total_cost,
            "weak_cost_usd": weak_cost,
            "strong_cost_usd": strong_cost,
            "prompt_tokens": weak_pt,
            "completion_tokens": weak_ct,
            "router_state": self.router.predictor._router.get_state(),
        }

    def _on_resume(self, existing_results: list[dict]) -> None:
        """Restore product LinTS predictor state from last saved result."""
        if existing_results:
            last_state = existing_results[-1].get("router_state")
            if last_state:
                self.router.predictor._router.load_state(last_state)
                print(f"  Restored product LinTS state at t={self.router.predictor._router._t}")
