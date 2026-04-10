"""CrewAI integration for RouteSmith.

Two modes:

1. **Proxy mode** — ``routesmith_crewai_llm()`` returns a ``crewai.LLM``
   pointed at the RouteSmith proxy server (requires proxy to be running).

2. **Native mode** — ``routesmith_crewai_chat_model()`` returns a
   ``ChatRouteSmith`` (LangChain BaseChatModel) that CrewAI agents accept
   directly, with no proxy server needed.

Example (proxy mode):
    >>> from crewai import Agent, Task, Crew
    >>> from routesmith.integrations.crewai import routesmith_crewai_llm
    >>> llm = routesmith_crewai_llm()
    >>> agent = Agent(role="Analyst", goal="...", backstory="...", llm=llm)

Example (native mode):
    >>> from routesmith.integrations.crewai import routesmith_crewai_chat_model
    >>> llm = routesmith_crewai_chat_model()
    >>> llm.routesmith.register_model("openai/gpt-4o-mini", ...)
    >>> agent = Agent(role="Analyst", goal="...", backstory="...", llm=llm)
"""

from __future__ import annotations

from typing import Any

from routesmith.client import RouteSmith
from routesmith.config import RouteSmithConfig


def routesmith_crewai_llm(
    host: str = "http://localhost:9119",
    model: str = "auto",
    **kwargs: Any,
) -> Any:
    """Return a ``crewai.LLM`` configured to route through the RouteSmith proxy.

    Args:
        host: RouteSmith proxy URL (default: http://localhost:9119).
        model: Model name to use (default: auto).
        **kwargs: Extra kwargs forwarded to ``crewai.LLM``.

    Returns:
        A configured ``crewai.LLM`` instance.
    """
    try:
        from crewai import LLM
    except ImportError as e:
        raise ImportError(
            "CrewAI integration requires crewai. "
            "Install with: pip install 'routesmith[crewai]'"
        ) from e

    return LLM(
        model=f"openai/{model}",
        base_url=f"{host.rstrip('/')}/v1",
        api_key="dummy",
        **kwargs,
    )


def routesmith_crewai_chat_model(
    routesmith: RouteSmith | None = None,
    config: RouteSmithConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Return a ``ChatRouteSmith`` for native use as a CrewAI agent LLM.

    CrewAI agents accept any LangChain ``BaseChatModel`` as their ``llm``
    argument. This avoids running the proxy server — RouteSmith routes
    requests directly.

    Args:
        routesmith: Pre-configured RouteSmith instance.
        config: RouteSmithConfig to use when creating a new instance.
        **kwargs: Extra kwargs forwarded to ``ChatRouteSmith``.

    Returns:
        A ``ChatRouteSmith`` instance.

    Example:
        llm = routesmith_crewai_chat_model()
        llm.routesmith.register_model("openai/gpt-4o-mini",
                                      cost_per_1k_input=0.15,
                                      cost_per_1k_output=0.60,
                                      quality_score=0.85)
        agent = Agent(role="Writer", goal="...", backstory="...", llm=llm)
    """
    from routesmith.integrations.langchain import ChatRouteSmith

    rs = routesmith or RouteSmith(config=config)
    return ChatRouteSmith(routesmith=rs, **kwargs)
