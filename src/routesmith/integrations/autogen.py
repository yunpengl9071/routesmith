"""AutoGen integration for RouteSmith.

RouteSmith exposes an OpenAI-compatible proxy, so AutoGen agents (which
natively support custom OpenAI-compatible endpoints) can route through it
with no code changes — just configure the ``base_url``.

Helper functions here generate the AutoGen ``config_list`` and
``llm_config`` dicts so you don't have to remember the exact structure.

Example:
    >>> from autogen import AssistantAgent, UserProxyAgent
    >>> from routesmith.integrations.autogen import routesmith_autogen_llm_config
    >>> llm_config = routesmith_autogen_llm_config()
    >>> assistant = AssistantAgent("assistant", llm_config=llm_config)
    >>> user = UserProxyAgent("user", human_input_mode="NEVER")
    >>> user.initiate_chat(assistant, message="What is 2+2?")
"""

from __future__ import annotations

from typing import Any


def routesmith_config_list(
    host: str = "http://localhost:9119",
    model: str = "auto",
    extra_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return an AutoGen ``config_list`` for routing through RouteSmith.

    Args:
        host: RouteSmith proxy URL (default: http://localhost:9119).
        model: Primary model name (default: auto — RouteSmith picks).
        extra_models: Additional model names to include in the config list.

    Returns:
        A list of config dicts compatible with AutoGen's ``config_list`` format.

    Example:
        config_list = routesmith_config_list()
        # [{"model": "auto", "base_url": "http://localhost:9119/v1",
        #   "api_key": "dummy"}]
    """
    base_url = f"{host.rstrip('/')}/v1"
    models = [model] + (extra_models or [])
    return [
        {"model": m, "base_url": base_url, "api_key": "dummy"}
        for m in models
    ]


def routesmith_autogen_llm_config(
    host: str = "http://localhost:9119",
    model: str = "auto",
    extra_models: list[str] | None = None,
    cache_seed: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a complete AutoGen ``llm_config`` dict for RouteSmith.

    Pass the result directly to ``AssistantAgent`` or ``UserProxyAgent``.

    Args:
        host: RouteSmith proxy URL (default: http://localhost:9119).
        model: Primary model name (default: auto).
        extra_models: Additional model names to include.
        cache_seed: AutoGen cache seed (set to None to disable caching,
            since RouteSmith handles its own semantic cache).
        **kwargs: Extra keys merged into the llm_config dict.

    Returns:
        A dict with ``config_list`` and any extra keys provided.

    Example:
        from autogen import AssistantAgent
        llm_config = routesmith_autogen_llm_config()
        agent = AssistantAgent("assistant", llm_config=llm_config)
    """
    config: dict[str, Any] = {
        "config_list": routesmith_config_list(
            host=host, model=model, extra_models=extra_models
        ),
    }
    if cache_seed is not None:
        config["cache_seed"] = cache_seed
    config.update(kwargs)
    return config


def routesmith_autogen_agents(
    host: str = "http://localhost:9119",
    assistant_name: str = "assistant",
    user_name: str = "user",
    system_message: str = "You are a helpful AI assistant.",
    human_input_mode: str = "NEVER",
    max_consecutive_auto_reply: int = 10,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Create a (AssistantAgent, UserProxyAgent) pair backed by RouteSmith.

    Both agents use the RouteSmith proxy. Requires AutoGen (pyautogen or
    autogen-agentchat) to be installed.

    Args:
        host: RouteSmith proxy URL.
        assistant_name: Name of the assistant agent.
        user_name: Name of the user proxy agent.
        system_message: System message for the assistant.
        human_input_mode: AutoGen human input mode for UserProxyAgent.
        max_consecutive_auto_reply: Max auto-replies before human input.
        **kwargs: Extra kwargs forwarded to AssistantAgent.

    Returns:
        Tuple of (AssistantAgent, UserProxyAgent).

    Example:
        assistant, user = routesmith_autogen_agents()
        user.initiate_chat(assistant, message="Write a haiku about routing.")
    """
    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        try:
            from autogen_agentchat.agents import AssistantAgent, UserProxyAgent  # type: ignore[no-redef]
        except ImportError as e:
            raise ImportError(
                "AutoGen integration requires pyautogen or autogen-agentchat. "
                "Install with: pip install 'routesmith[autogen]'"
            ) from e

    llm_config = routesmith_autogen_llm_config(host=host)

    assistant = AssistantAgent(
        name=assistant_name,
        system_message=system_message,
        llm_config=llm_config,
        **kwargs,
    )
    user = UserProxyAgent(
        name=user_name,
        human_input_mode=human_input_mode,
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config=False,  # UserProxy doesn't need LLM
    )
    return assistant, user
