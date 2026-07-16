"""Embedding-based quality predictor using matrix factorization (RouteLLM-style)."""

from __future__ import annotations

from typing import Any

import numpy as np

from routesmith.predictor.base import BasePredictor, PredictionResult


class EmbeddingPredictor(BasePredictor):
    """
    Matrix factorization quality predictor based on RouteLLM (Ong et al., ICLR 2025).

    Represents each model as a learned d-dimensional embedding vector.
    Encodes each query as a d-dimensional embedding via sentence-transformers.
    Predicted quality = sigmoid(dot(query_emb, model_emb)), normalized to [0, 1].

    Model embeddings are updated via gradient descent on observed quality signals,
    enabling the predictor to improve from production feedback.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        model_quality_priors: dict[str, float] | None = None,
        embedding_dim: int = 384,
        learning_rate: float = 0.01,
        l2_reg: float = 0.001,
    ) -> None:
        """
        Initialize matrix factorization predictor.

        Args:
            embedding_model: Sentence transformer model for query embeddings.
            model_quality_priors: Prior quality scores for each model (cold start).
            embedding_dim: Dimension of the embedding space.
            learning_rate: SGD learning rate for model embedding updates.
            l2_reg: L2 regularization strength.
        """
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.model_quality_priors = model_quality_priors or {}

        # Learned model embeddings: model_id -> np.ndarray (embedding_dim,)
        self._model_embeddings: dict[str, np.ndarray] = {}

        # Per-model update count (for confidence)
        self._update_counts: dict[str, int] = {}

        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        """Lazy load the sentence transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for EmbeddingPredictor. "
                    "Install with: pip install routesmith-llm[predictor]"
                )
        return self._encoder

    def _ensure_model_embedding(self, model_id: str) -> np.ndarray:
        """Initialize model embedding from prior if not yet learned."""
        if model_id not in self._model_embeddings:
            prior = self.model_quality_priors.get(model_id, 0.5)
            # Initialize near the prior direction
            emb = np.random.default_rng(0).normal(0, 0.1, self.embedding_dim).astype(np.float32)
            # Scale so dot with unit-norm query gives roughly logit(prior)
            logit = np.log(max(prior, 1e-6) / max(1 - prior, 1e-6))
            norm = np.linalg.norm(emb)
            emb = emb / (norm if norm > 1e-8 else 1e-8) * (logit * 0.1)
            self._model_embeddings[model_id] = emb
            self._update_counts.setdefault(model_id, 0)
        return self._model_embeddings[model_id]

    def _encode_query(self, messages: list[dict[str, str]]) -> np.ndarray:
        """Encode messages into a query embedding vector."""
        text = " ".join(m.get("content", "") for m in messages)
        if not text.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        encoder = self._get_encoder()
        emb = encoder.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def predict(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> list[PredictionResult]:
        """
        Predict quality using matrix factorization: sigmoid(query_emb @ model_emb).

        Falls back to prior quality scores when no query embedding is available
        (sentence-transformers not installed) or on cold start.
        """
        try:
            query_emb = self._encode_query(messages)
        except ImportError:
            query_emb = None

        results = []
        for model_id in model_ids:
            if query_emb is not None and np.any(query_emb):
                emb = self._ensure_model_embedding(model_id)
                score = float(np.dot(query_emb, emb))
                # Sigmoid to [0, 1]
                quality = 1.0 / (1.0 + np.exp(-score))
                # Confidence increases with update count
                updates = self._update_counts.get(model_id, 0)
                confidence = updates / (updates + 20.0)
            else:
                # Fallback to prior
                quality = self.model_quality_priors.get(model_id, 0.5)
                confidence = 0.1

            results.append(
                PredictionResult(
                    model_id=model_id,
                    predicted_quality=quality,
                    confidence=confidence,
                )
            )

        return sorted(results, key=lambda r: r.predicted_quality, reverse=True)

    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
    ) -> None:
        """
        Update model embedding via gradient descent on observed quality.

        Gradient: (sigmoid - target) * query_emb + l2_reg * model_emb
        """
        target = reward_override if reward_override is not None else actual_quality
        self._update_counts[model_id] = self._update_counts.get(model_id, 0) + 1

        try:
            query_emb = self._encode_query(messages)
        except ImportError:
            # Without embeddings, fall back to prior-weighted EMA
            alpha = 0.1
            current = self.model_quality_priors.get(model_id, 0.5)
            self.model_quality_priors[model_id] = alpha * target + (1 - alpha) * current
            return

        if not np.any(query_emb):
            return

        emb = self._ensure_model_embedding(model_id)
        score = float(np.dot(query_emb, emb))
        pred = 1.0 / (1.0 + np.exp(-score))

        # Gradient: (pred - target) * query_emb + l2_reg * emb
        grad = (pred - target) * query_emb + self.l2_reg * emb
        self._model_embeddings[model_id] = emb - self.learning_rate * grad

    def add_arm(self, model_id: str, quality_score: float | None = None) -> None:
        """Register a new model arm."""
        if quality_score is not None:
            self.model_quality_priors[model_id] = quality_score
        self._ensure_model_embedding(model_id)

    def remove_arm(self, model_id: str) -> None:
        """Remove a model arm."""
        self._model_embeddings.pop(model_id, None)
        self._update_counts.pop(model_id, None)
        self.model_quality_priors.pop(model_id, None)
