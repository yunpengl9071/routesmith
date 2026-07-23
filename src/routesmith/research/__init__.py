"""Research instruments for off-policy evaluation (Proposal B v4.1).

Additive, self-contained tooling that wraps the production router without changing its
hot path. The centerpiece is propensity logging with an enforced exploration floor and
exact seeded replay (WU-0.2) — the substrate an OPE study needs and production routing
never had.
"""
from routesmith.research.propensity import (
    DecisionLog,
    ExplorationFloor,
    PropensityLogger,
    RoutingDecision,
)

__all__ = [
    "ExplorationFloor",
    "PropensityLogger",
    "RoutingDecision",
    "DecisionLog",
]
