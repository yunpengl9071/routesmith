"""
AutoGen agent pair with RouteSmith.

Uses the RouteSmith proxy server — start it first:
    routesmith serve --port 9119

Requires: routesmith[autogen]
"""

from routesmith.integrations.autogen import routesmith_autogen_agents


def main() -> None:
    try:
        from autogen import ChatResult
    except ImportError:
        print("pyautogen not installed. Install with: pip install 'routesmith[autogen]'")
        return

    assistant, user = routesmith_autogen_agents()
    result: ChatResult = user.initiate_chat(
        assistant,
        message="What are the three laws of robotics?",
        max_turns=2,
    )
    print(f"Chat history: {len(result.chat_history)} messages")

    import requests
    stats = requests.get("http://localhost:9119/v1/stats", timeout=5).json()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
