# Compliance Filtering

RouteSmith lets you tag models with compliance attributes and filter by them on a per-request basis. This is useful for:

- **HIPAA** — Patient health data must stay in compliant environments (e.g., AWS Bedrock)
- **SOC2** — Internal audit requirements
- **Regional** — EU data must stay in EU regions (GDPR)
- **Custom** — Any compliance attribute your organization requires

Compliance filtering is a **hard gate**: models that don't have all required tags are excluded from routing entirely. It runs before quality prediction.

---

## Tagging Models

Add `compliance_tags` when registering models:

```python
# HIPAA-compliant Bedrock model (data stays in VPC)
rs.register_model(
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
    quality_score=0.92,
    compliance_tags={"hipaa", "soc2"},
)

# Public model (no compliance guarantees)
rs.register_model(
    "gpt-4o-mini",
    cost_per_1k_input=0.00015,
    cost_per_1k_output=0.0006,
    compliance_tags={"soc2"},
)

# Unrestricted model
rs.register_model(
    "gpt-4o",
    cost_per_1k_input=0.005,
    cost_per_1k_output=0.015,
    # No compliance_tags — available for all requests
)
```

A model with **no** `compliance_tags` is available for any request, including those with compliance requirements. Tagged models pass the filter when they have **all** requested tags.

---

## Per-Request Filtering

Pass `required_compliance` to restrict routing to compliant models:

```python
# Only route to HIPAA-compliant models
response = rs.completion(
    messages=[{"role": "user", "content": "Patient diagnosis summary"}],
    required_compliance={"hipaa"},
)

# Route to EU-region models for GDPR compliance
response = rs.completion(
    messages=[{"role": "user", "content": "User data export"}],
    required_compliance={"gdpr", "eu-west-1"},
)
```

---

## Error Handling

If no model satisfies the compliance requirement, RouteSmith raises `NoCompliantModelError`:

```python
from routesmith.exceptions import NoCompliantModelError

try:
    response = rs.completion(
        messages=[{"role": "user", "content": patient_question}],
        required_compliance={"hipaa"},
    )
except NoCompliantModelError as e:
    print(f"No HIPAA model available.")
    print(f"Required: {e.required_tags}")
    print(f"Available across all models: {e.available_tags}")
    # "Required: {'hipaa'}. Available: {'soc2', 'pci'}"
```

The error message is self-documenting — it lists what was requested and what's available.

---

## Combining with Capability Filtering

Compliance filtering works alongside existing [capability filtering](concepts/strategies.md). Capabilities are auto-detected from messages and kwargs:

```python
# HIPAA model that also supports tool calling
rs.register_model(
    "bedrock/claude-sonnet",
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
    supports_function_calling=True,
    compliance_tags={"hipaa"},
)

# When HIPAA is required AND tools are passed, both filters apply:
response = rs.completion(
    messages=[{"role": "user", "content": "Schedule an appointment"}],
    required_compliance={"hipaa"},
    tools=[{"type": "function", "function": {"name": "create_event"}}],
)
```

If a HIPAA-compliant model lacks tool support, the request will raise `NoCapableModelError`.

---

## Multi-Region Setup

Tag models with region-specific compliance:

```python
rs.register_model(
    "bedrock/claude-sonnet-us",
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
    compliance_tags={"hipaa", "us-east-1"},
)

rs.register_model(
    "bedrock/claude-sonnet-eu",
    cost_per_1k_input=0.0035,
    cost_per_1k_output=0.017,
    compliance_tags={"hipaa", "eu-west-1"},
)

# Route to EU region only
response = rs.completion(
    messages=[{"role": "user", "content": "European user data"}],
    required_compliance={"hipaa", "eu-west-1"},
)
```

---

## Best Practices

- **Use descriptive tags** — Standardize on a naming convention like `hipaa`, `soc2`, `pci`, `region-xx-yyyy`.
- **Don't over-tag** — Only add compliance tags that have real organizational meaning. Over-tagging makes routing unnecessarily restrictive.
- **Tag all models** — Even models with no compliance guarantees should be explicitly untagged (empty set) rather than forgotten. This makes audit trails clear.
- **Combine with per-project isolation** — Use separate RouteSmith instances for each project, each with its own compliance-tagged models.

```python
# Production project with HIPAA models only
prod = RouteSmith(project="production", config=prod_config)
prod.register_model("bedrock/claude-sonnet", 0.003, 0.015, compliance_tags={"hipaa"})

# Dev project with cheaper, non-compliant models
dev = RouteSmith(project="development", config=dev_config)
dev.register_model("gpt-4o-mini", 0.00015, 0.0006)
```