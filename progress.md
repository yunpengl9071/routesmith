# Progress

## Status
All code tasks complete on `feature/v0.3.0-enterprise`.

## Tasks Completed
### Section A: CostModel + Capacity
- [x] A1: CostModel enum, BudgetBehavior enum, ModelConfig fields
- [x] A2: CapacityTracker (rolling window RPM tracking)
- [x] A3: CapacityExhaustedError, NoCompliantModelError exceptions
- [x] A4: Wire CapacityTracker into ModelRegistry

### Section B: Compliance Filtering
- [x] B1: compliance_tags on ModelConfig, filter_by_compliance on ModelRegistry
- [x] B2: Per-request required_compliance in client + router

### Section C: PROVISIONED_FIRST Strategy
- [x] C1: Add PROVISIONED_FIRST to RoutingStrategy enum
- [x] C2: _route_provisioned_first in Router with overflow

### Section D: Budget Enforcement + Per-Project
- [x] D1: BudgetBehavior enforcement (FAIL/FALLBACK/QUEUE)
- [x] D2: Per-project isolation via project parameter
- [x] D3: Enhanced stats (budget_events, by_cost_model, provisioned_utilization)
- [x] D4: Integration tests (9 tests) + full suite verification

## Test Count
639 passed, 16 skipped (up from 595 baseline)

## Next Steps
- [ ] Create PR from feature/v0.3.0-enterprise → dev
- [ ] Write user documentation: cost-models, compliance, multi-project, budget-enforcement guides
- [ ] A/B test framework (Phase 2 leftover)