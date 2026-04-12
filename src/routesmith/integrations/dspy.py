"""DSPy integration for RouteSmith.

Two modes:

1. **Proxy mode** — ``routesmith_lm()`` returns a ``dspy.LM`` pointed at the
   RouteSmith proxy server (requires the proxy to be running).

2. **Native mode** — ``RouteSmithLM`` implements the dspy LM interface and
   routes through RouteSmith directly (no proxy needed).

Example (proxy mode):
    >>> import dspy
    >>> from routesmith.integrations.dspy import routesmith_lm
    >>> lm = routesmith_lm()          # proxy must be running on :9119
    >>> dspy.configure(lm=lm)

Example (native mode):
    >>> from routesmith.integrations.dspy import RouteSmithLM
    >>> lm = RouteSmithLM()
    >>> lm.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
    ...                   cost_per_1k_output=0.60, quality_score=0.85)
    >>> dspy.configure(lm=lm)
    >>> predictor = dspy.Predict("question -> answer")
    >>> result = predictor(question="What is 2+2?")
"""

from __future__ import annotations

from typing import Any

from routesmith.client import RouteSmith
from routesmith.config import PredictorConfig, RouteSmithConfig


def routesmith_lm(
    host: str = "http://localhost:9119",
    model: str = "auto",
    **kwargs: Any,
) -> Any:
    """Return a ``dspy.LM`` configured to route through the RouteSmith proxy.

    Args:
        host: RouteSmith proxy URL (default: http://localhost:9119).
        model: Model name to advertise to DSPy (default: auto).
        **kwargs: Extra kwargs forwarded to ``dspy.LM``.

    Returns:
        A configured ``dspy.LM`` instance.
    """
    try:
        import dspy
    except ImportError as e:
        raise ImportError(
            "DSPy integration requires dspy-ai. "
            "Install with: pip install 'routesmith[dspy]'"
        ) from e

    return dspy.LM(
        f"openai/{model}",
        base_url=f"{host.rstrip('/')}/v1",
        api_key="dummy",
        **kwargs,
    )


class RouteSmithLM:
    """Native DSPy-compatible LM that routes through RouteSmith directly.

    Implements the dspy LM ``__call__`` protocol so it can be passed to
    ``dspy.configure(lm=...)``. Does not require the proxy server.

    Args:
        routesmith: Pre-configured RouteSmith instance.
        config: RouteSmithConfig to use when creating a new instance.
        agent_role: Optional role label for routing heuristics (e.g. "coding").
        conversation_id: Optional ID to associate turns in a conversation.
        track_conversation: If True, create a ConversationTracker to pass
            per-turn RouteContext to completion().
        reward_fn: Optional reward function forwarded to ConversationTracker.
    """

    def __init__(
        self,
        routesmith: RouteSmith | None = None,
        config: RouteSmithConfig | None = None,
        agent_role: str | None = None,
        conversation_id: str | None = None,
        track_conversation: bool = False,
        reward_fn: Any = None,
    ) -> None:
        self._rs = routesmith or RouteSmith(config=config)
        self.history: list[dict[str, Any]] = []
        self.agent_role = agent_role
        self._tracker = None
        if track_conversation:
            from routesmith.feedback.conversation import ConversationTracker
            self._tracker = ConversationTracker(
                agent_role=agent_role,
                conversation_id=conversation_id,
                reward_fn=reward_fn,
            )

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        n: int = 1,
        **kwargs: Any,
    ) -> list[str]:
        """Generate completions, routing through RouteSmith.

        Args:
            prompt: Plain-text prompt (converted to a user message).
            messages: OpenAI-format message list (takes priority over prompt).
            n: Number of completions to return (only 1 is supported).
            **kwargs: Forwarded to RouteSmith.completion().

        Returns:
            List of completion strings (length == n).
        """
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]

        from routesmith.config import RouteContext
        ctx = None
        if self._tracker is not None:
            ctx = self._tracker.next_context(messages)
        elif self.agent_role is not None:
            ctx = RouteContext(agent_role=self.agent_role)

        response = self._rs.completion(messages=messages, context=ctx, **kwargs)
        text = response.choices[0].message.content or ""

        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "response": [text],
        })
        return [text]

    def register_model(self, model_id: str, **kwargs: Any) -> None:
        self._rs.register_model(model_id, **kwargs)

    @property
    def stats(self) -> dict[str, Any]:
        return self._rs.stats

    @classmethod
    def with_openrouter_models(
        cls,
        model_ids: list[str] | None = None,
        predictor_type: str = "lints",
    ) -> RouteSmithLM:
        """Create an LM pre-loaded with models fetched from OpenRouter.

        Args:
            model_ids: Model IDs to use. Defaults to GPT-4o-mini + GPT-4o
                + Claude 3 Haiku.
            predictor_type: Routing algorithm — 'lints' or 'linucb'.
        """
        from routesmith.registry.openrouter import fetch_pricing

        ids = model_ids or [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
        ]
        pricing = fetch_pricing(ids)

        config = RouteSmithConfig(
            predictor_type=predictor_type,
            predictor=PredictorConfig(),
        )
        instance = cls(config=config)
        for mid in ids:
            if mid in pricing:
                m = pricing[mid]
                instance.register_model(
                    mid,
                    cost_per_1k_input=m.cost_per_1k_input,
                    cost_per_1k_output=m.cost_per_1k_output,
                    quality_score=m.quality_score,
                    context_window=m.context_window,
                )
        return instance
