# Budget Enforcement

RouteSmith supports three behaviors when your budget is exhausted: **FAIL**, **FALLBACK**, and **QUEUE**. Configured via `BudgetBehavior` on `RouteSmithConfig`.

---

## Configuration

```python
from routesmith.config import BudgetBehavior

config = RouteSmithConfig().with_budget(max_cost_per_day=100.0)
config.budget_behavior = BudgetBehavior.FALLBACK
rs = RouteSmith(config=config)
```

---

## FAIL (Default)

Raises `BudgetExceededError` when the budget is exhausted. This is the **safe default** — it prevents any unexpected spend.

```python
from routesmith.exceptions import BudgetExceededError

config = RouteSmithConfig().with_budget(max_cost_per_day=0.50)
config.budget_behavior = BudgetBehavior.FAIL
rs = RouteSmith(config=config)

try:
    response = rs.completion(messages=[{"role": "user", "content": "Hello"}])
except BudgetExceededError as e:
    print(f"Budget exceeded: ${e.current_spend:.2f} / ${e.limit:.2f}")
    print(f"Resets in {int(e.reset_seconds // 60)} minutes")
```

**Use when:** Batch jobs, cost-sensitive experiments, or any scenario where exceeding budget is unacceptable.

---

## FALLBACK

Routes to the **cheapest eligible model** regardless of quality when the budget is exhausted. The service stays available but may use lower-quality models.

```python
config = RouteSmithConfig().with_budget(max_cost_per_day=10.0)
config.budget_behavior = BudgetBehavior.FALLBACK
rs = RouteSmith(config=config)

# When budget is available → routes normally (quality prediction)
# When budget is exhausted → automatically uses cheapest model
response = rs.completion(messages=[{"role": "user", "content": "Hello"}])
```

Fallback still respects:
- **Capability filters** — If the cheapest model lacks `tool_calling`, it won't be selected for tool requests
- **Compliance filters** — If the cheapest model isn't HIPAA-compliant, it won't be selected for HIPAA requests
- **Model availability** — If no eligible model exists, falls through to FAIL

RouteSmith logs a warning when fallback is activated:

```
WARNING routesmith: Budget exceeded, falling back to gpt-4o-mini ($0.00015/1K input)
```

**Use when:** Production services that must stay available even if quality degrades.

---

## QUEUE

Holds requests until the budget window resets (next hour or day). **Sync calls raise `BudgetExceededError`** with guidance to use async:

```python
config.budget_behavior = BudgetBehavior.QUEUE
rs = RouteSmith(config=config)

# This raises BudgetExceededError with a hint:
response = rs.completion(messages=[{"role": "user", "content": "Hello"}])
# "Budget exceeded. Use acompletion() with QUEUE behavior for async queueing."
```

Use async completion instead:

```python
# Request resolves when budget window resets
response = await rs.acompletion(
    messages=[{"role": "user", "content": "Process this document"}],
)
```

**Use when:** Batch jobs that can wait (overnight processing, document analysis, scheduled tasks).

> **Note:** QUEUE is best-effort. If the session is restarted before budget resets, queued requests are lost. For persistent queuing, use an external message queue.

---

## Monitoring Budget Events

Track budget enforcement activity through stats:

```python
events = rs.stats["budget_events"]
# {
#   "failures": 3,     # Times FAIL was raised
#   "fallbacks": 47,   # Times FALLBACK was used
#   "queued": 0,       # Times QUEUE held a request
# }
```

---

## Best Practices

| Environment | Recommended Behavior | Reason |
|-------------|---------------------|--------|
| Production | `FALLBACK` | Keeps service running |
| Batch jobs | `FAIL` | Prevents unexpected costs |
| Development | `FAIL` | Catches budget issues early |
| Nightly jobs | `QUEUE` | Can wait for budget reset |

- **Start with FAIL** during development to catch unexpected costs early
- **Switch to FALLBACK** for production to ensure availability
- **Set alerts** at 50%, 75%, and 90% of your budget threshold:

```python
stats = rs.stats
spend = stats["total_cost_usd"]
daily_limit = config.budget.max_cost_per_day

if daily_limit and spend >= daily_limit * 0.9:
    print(f"Alert: 90% of daily budget consumed (${spend:.2f}/${daily_limit:.2f})")
```

- **Monitor per-project** using the [multi-project setup](multi-project.md) to catch budget issues before they affect all workloads