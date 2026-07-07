"""
LlamaIndex with RouteSmith proxy.

Use OpenAI-compatible provider pointed at the RouteSmith proxy.
Start proxy first: routesmith serve --port 9119

Requires: pip install llama-index llama-index-llms-openai-like
"""


def main() -> None:
    try:
        from llama_index.llms.openai_like import OpenAILike
    except ImportError:
        print("llama-index not installed.")
        return

    llm = OpenAILike(
        model="auto",
        api_base="http://localhost:9119/v1",
        api_key="dummy",
        is_chat_model=True,
    )
    response = llm.complete("What is 2+2?")
    print(f"Response: {response.text}")


if __name__ == "__main__":
    main()
