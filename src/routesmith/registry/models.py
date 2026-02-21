"""Model registry and capability mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a registered model."""

    model_id: str
    cost_per_1k_input: float  # USD per 1000 input tokens
    cost_per_1k_output: float  # USD per 1000 output tokens
    quality_score: float = 0.8  # Expected quality 0-1
    latency_p50_ms: float = 500.0  # Median latency
    latency_p95_ms: float = 1500.0  # 95th percentile latency
    latency_p99_ms: float = 2000.0  # 99th percentile latency
    context_window: int = 128000  # Max tokens
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True
    capabilities: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-populate capabilities from boolean flags."""
        if self.supports_function_calling:
            self.capabilities.add("tool_calling")
        if self.supports_vision:
            self.capabilities.add("vision")
        if self.supports_json_mode:
            self.capabilities.add("json_mode")
        if self.supports_streaming:
            self.capabilities.add("streaming")

    @property
    def cost_per_1k_total(self) -> float:
        """Average cost assuming 1:1 input:output ratio."""
        return (self.cost_per_1k_input + self.cost_per_1k_output) / 2


class ModelRegistry:
    """
    Registry of available models with their capabilities and costs.

    Provides lookup and filtering for routing decisions.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}

    def register(
        self,
        model_id: str,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
        quality_score: float = 0.8,
        latency_p50_ms: float = 500.0,
        latency_p99_ms: float = 2000.0,
        context_window: int = 128000,
        capabilities: set[str] | None = None,
        supports_function_calling: bool = True,
        supports_vision: bool = False,
        supports_json_mode: bool = True,
        supports_streaming: bool = True,
        **kwargs: Any,
    ) -> ModelConfig:
        """
        Register a model with its capabilities.

        Args:
            model_id: LiteLLM model identifier
            cost_per_1k_input: Cost per 1000 input tokens (USD)
            cost_per_1k_output: Cost per 1000 output tokens (USD)
            quality_score: Expected quality score 0-1
            latency_p50_ms: Median latency in milliseconds
            latency_p99_ms: 99th percentile latency
            context_window: Maximum context size
            capabilities: Explicit capability set (merged with auto-detected)
            supports_function_calling: Whether the model supports tool/function calling
            supports_vision: Whether the model supports image inputs
            supports_json_mode: Whether the model supports JSON mode
            supports_streaming: Whether the model supports streaming
            **kwargs: Additional model metadata

        Returns:
            The created ModelConfig.
        """
        config = ModelConfig(
            model_id=model_id,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
            quality_score=quality_score,
            latency_p50_ms=latency_p50_ms,
            latency_p99_ms=latency_p99_ms,
            context_window=context_window,
            supports_function_calling=supports_function_calling,
            supports_vision=supports_vision,
            supports_json_mode=supports_json_mode,
            supports_streaming=supports_streaming,
            capabilities=capabilities or set(),
            metadata=kwargs,
        )
        self._models[model_id] = config
        return config

    def get(self, model_id: str) -> ModelConfig | None:
        """Get model config by ID."""
        return self._models.get(model_id)

    def list_models(self) -> list[ModelConfig]:
        """List all registered models."""
        return list(self._models.values())

    def filter_by_quality(self, min_quality: float) -> list[ModelConfig]:
        """Get models meeting minimum quality threshold."""
        return [m for m in self._models.values() if m.quality_score >= min_quality]

    def filter_by_cost(self, max_cost_per_1k: float) -> list[ModelConfig]:
        """Get models under cost threshold."""
        return [m for m in self._models.values() if m.cost_per_1k_total <= max_cost_per_1k]

    def get_cheapest(self, min_quality: float = 0.0) -> ModelConfig | None:
        """Get the cheapest model meeting quality threshold."""
        candidates = self.filter_by_quality(min_quality)
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.cost_per_1k_total)

    def get_best_quality(self, max_cost_per_1k: float | None = None) -> ModelConfig | None:
        """Get the highest quality model under cost threshold."""
        if max_cost_per_1k is not None:
            candidates = self.filter_by_cost(max_cost_per_1k)
        else:
            candidates = list(self._models.values())
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.quality_score)

    def get_by_capability(self, capability: str) -> list[ModelConfig]:
        """Get models supporting a specific capability."""
        return [m for m in self._models.values() if capability in m.capabilities]

    def filter_by_capabilities(self, required: set[str]) -> list[ModelConfig]:
        """Get models supporting all required capabilities."""
        return [m for m in self._models.values() if required.issubset(m.capabilities)]

    def sorted_by_cost(self, descending: bool = False) -> list[ModelConfig]:
        """Get models sorted by cost."""
        return sorted(
            self._models.values(),
            key=lambda m: m.cost_per_1k_total,
            reverse=descending,
        )

    def sorted_by_quality(self, descending: bool = True) -> list[ModelConfig]:
        """Get models sorted by quality."""
        return sorted(
            self._models.values(),
            key=lambda m: m.quality_score,
            reverse=descending,
        )

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models
