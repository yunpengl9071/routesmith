# DSPy Integration

RouteSmith integrates with DSPy as a custom LM.

```python
import dspy
from routesmith.integrations.dspy import DSPyLM, routesmith_settings

# Configure DSPy with RouteSmith
dspy.settings.configure(lm=DSPyLM())

# Use as normal DSPy
class SimpleQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

qa = dspy.ChainOfThought(SimpleQA)
result = qa(question="What is the capital of France?")
print(result.answer)
```

## Custom Configuration

```python
from routesmith import RouteSmith, RouteSmithConfig

rs = RouteSmith(RouteSmithConfig().with_budget(max_cost_per_day=10.0))
dspy.settings.configure(lm=DSPyLM(routesmith=rs))
```