"""
OpenAI Agents SDK with RouteSmith proxy.

Point the SDK's OpenAI client at the RouteSmith proxy.
Start proxy first: routesmith serve --port 9119

Requires: pip install openai-agents
"""

from openai import OpenAI


def main() -> None:
    client = OpenAI(
        base_url="http://localhost:9119/v1",
        api_key="dummy",
    )
    response = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=50,
    )
    print(f"Response: {response.choices[0].message.content}")
    meta = getattr(response, "routesmith_metadata", None)
    if meta:
        print(f"Routed to: {meta.get('model_selected', '?')}")


if __name__ == "__main__":
    main()
