"""
Pydantic AI agent with RouteSmith proxy.

Point Pydantic AI's OpenAI provider at the RouteSmith proxy.
Start proxy first: routesmith serve --port 9119

Requires: pip install pydantic-ai
"""


def main() -> None:
    try:
        from pydantic_ai import Agent
    except ImportError:
        print("pydantic-ai not installed.")
        return

    agent = Agent(
        "openai:auto",
        system_prompt="You are a helpful assistant.",
        base_url="http://localhost:9119/v1",
        api_key="dummy",
    )
    result = agent.run_sync("What is the capital of France?")
    print(f"Response: {result.data}")


if __name__ == "__main__":
    main()
