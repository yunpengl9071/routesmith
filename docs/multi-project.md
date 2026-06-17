# Multi-Project Cost Allocation

RouteSmith supports per-project isolation through separate `RouteSmith` instances. Each project has its own model registry, budget, cost tracking, and feedback storage — **no shared state**.

This is useful for:

- **Multi-team setups** — Each team gets their own client with their own models and budget
- **Multi-environment** — Separate dev, staging, and production instances
- **Multi-tenant applications** — Each customer or workspace gets an isolated instance
- **Cost allocation** — Track spending per project for billing or reporting

---

## Creating Projects

Pass a `project` name when creating a RouteSmith instance:

```python
# Customer support bot
rs_support = RouteSmith(project="customer-support")
rs_support.register_model("gpt-4o-mini", 0.00015, 0.0006)
rs_support.register_model("gpt-4o", 0.005, 0.015)

# Internal search tool
rs_search = RouteSmith(project="internal-search")
rs_search.register_model("gpt-4o-mini", 0.00015, 0.0006)
```

Each instance is fully self-contained — registering models on one does not affect the other.

---

## Per-Project Stats

```python
print(rs_support.stats["project"])  # "customer-support"
print(rs_support.stats["request_count"])   # 1500 (this project only)
print(rs_search.stats["request_count"])    # 342 (separate counter)
print(rs_support.stats["total_cost_usd"])  # Project-specific cost
```

Stats are scoped to the instance. There is no central aggregator — you collect stats from each project independently.

---

## Aggregating Across Projects

To get an organization-wide view, iterate over your project instances:

```python
projects = {
    "customer-support": rs_support,
    "internal-search": rs_search,
    "doc-summarizer": rs_summarizer,
}

total_requests = 0
total_cost = 0.0

for name, client in projects.items():
    stats = client.stats
    total_requests += stats["request_count"]
    total_cost += stats["total_cost_usd"]
    print(f"{name}: ${stats['total_cost_usd']:.2f} ({stats['request_count']} requests)")

print(f"\nTotal: ${total_cost:.2f} across {total_requests} requests")
```

---

## Per-Project Budgets

Each project can have its own budget:

```python
support_config = RouteSmithConfig().with_budget(max_cost_per_day=100.0)
rs_support = RouteSmith(config=support_config, project="customer-support")

search_config = RouteSmithConfig().with_budget(max_cost_per_day=20.0)
rs_search = RouteSmith(config=search_config, project="internal-search")
```

When a project's budget is exhausted, only that project is affected — other projects continue normally.

---

## Prometheus Metrics

When using the [proxy server](deployment/docker.md) with Prometheus, project labels appear in metrics:

```
routesmith_requests_total{project="customer-support"} 1500
routesmith_requests_total{project="internal-search"} 342
routesmith_cost_usd_total{project="customer-support"} 12.45
routesmith_cost_usd_total{project="internal-search"} 8.90
routesmith_budget_remaining_usd{project="customer-support"} 87.55
```

This lets you build per-project dashboards in Grafana.

---

## Feedback Storage

When `feedback_storage_path` is set, each project uses a separate SQLite database if you configure distinct paths:

```python
support_config = RouteSmithConfig(
    feedback_storage_path="/var/lib/routesmith/customer-support.db",
)
search_config = RouteSmithConfig(
    feedback_storage_path="/var/lib/routesmith/internal-search.db",
)
```

If projects share the same storage path, records are still tagged with the project name in the `agent_id` and `agent_role` fields.

---

## Best Practices

- **One instance per project** — Don't try to reuse a single RouteSmith for different workloads. Create separate instances.
- **Use descriptive project names** — `"customer-support"` is better than `"project-1"`. These appear in logs and metrics.
- **Set per-project budgets** — Without per-project budgets, one runaway project can't affect others, but setting limits adds safety.
- **Aggregate at the reporting layer** — RouteSmith keeps projects isolated; your dashboard/reporting layer should sum across them.

```python
# Factory pattern for consistent project setup
def create_project_client(project_name: str, daily_budget: float) -> RouteSmith:
    config = RouteSmithConfig().with_budget(max_cost_per_day=daily_budget)
    rs = RouteSmith(config=config, project=project_name)
    rs.register_model("gpt-4o-mini", 0.00015, 0.0006)
    rs.register_model("gpt-4o", 0.005, 0.015)
    return rs

# Usage
rs_support = create_project_client("customer-support", daily_budget=100.0)
rs_search = create_project_client("internal-search", daily_budget=20.0)
```