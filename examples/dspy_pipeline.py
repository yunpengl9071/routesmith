"""
DSPy pipeline with RouteSmith native LM.

Requires: routesmith[dspy]
          pip install dspy-ai
"""

from routesmith.integrations.dspy import RouteSmithLM


def main() -> None:
    try:
        import dspy
    except ImportError:
        print("dspy-ai not installed. Install with: pip install 'routesmith[dspy]'")
        return

    lm = RouteSmithLM()
    lm.register_model(
        "openai/gpt-4o-mini",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        quality_score=0.85,
    )
    dspy.configure(lm=lm)

    classify = dspy.Predict("sentence -> sentiment")
    result = classify(sentence="RouteSmith saves me money on LLM costs!")
    print(f"Sentiment: {result.sentiment}")

    print(f"\nStats: {lm.stats}")


if __name__ == "__main__":
    main()
