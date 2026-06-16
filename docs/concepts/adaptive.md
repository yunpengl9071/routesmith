# Adaptive Learning

RouteSmith improves over time by learning from production feedback. The predictor goes through three phases.

## Learning Phases

1. **Cold Start**: Uses initial quality scores from model registration
2. **Warm Up**: Blends initial scores with observed feedback
3. **Learned**: Uses the trained Random Forest model for predictions

## Feedback Signals

RouteSmith automatically collects six implicit quality signals:

- Response length ratio
- Token count (completion / prompt)
- Latency percentile
- Finish reason (stop vs. length)
- Repetition detection
- Empty response detection

## Recording Explicit Feedback

```python
response = rs.completion(messages=[...], include_metadata=True)
request_id = response.routesmith_metadata["request_id"]

# Record feedback
rs.record_outcome(request_id, score=0.95, feedback="good answer")
rs.record_outcome(request_id, success=False, feedback="wrong answer")
```

## Benefits

- **No manual tuning**: The system learns optimal routing for your specific workload
- **Adapts to changes**: If new models are added, the predictor adjusts
- **Personalized**: Different application patterns get different routing behavior