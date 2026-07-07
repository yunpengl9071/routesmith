"""Semantic caching layer for LLM responses."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A cached response entry."""

    query_hash: str
    query_embedding: list[float] | None
    response: Any
    model_id: str
    created_at: float
    ttl_seconds: int
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.created_at + self.ttl_seconds


class SemanticCache:
    """
    Semantic cache for LLM responses.

    Supports:
    - Exact match caching (hash-based)
    - Semantic similarity caching (embedding-based)
    - TTL-based expiration
    - Multi-namespace isolation

    Requires sentence-transformers for semantic matching.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries: int = 10000,
        embedding_model: str = "all-MiniLM-L6-v2",
        lock: bool = False,
        namespace: str = "default",
    ) -> None:
        """
        Initialize semantic cache.

        Args:
            similarity_threshold: Minimum similarity for semantic match (0-1).
            ttl_seconds: Time-to-live for cache entries.
            max_entries: Maximum number of cached entries.
            embedding_model: Model for computing embeddings.
            namespace: Isolation namespace for multi-tenant use.
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.embedding_model = embedding_model
        self.namespace = namespace

        self._exact_cache: dict[str, CacheEntry] = {}
        self._semantic_entries: list[CacheEntry] = []
        self._encoder: Any = None
        self._lock: Any = None
        if lock:
            import threading
            self._lock = threading.Lock()

    def _get_encoder(self) -> Any:
        """Lazy load sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic caching. "
                    "Install with: pip install routesmith[cache]"
                )
        return self._encoder

    def _hash_messages(self, messages: list[dict[str, str]]) -> str:
        """Create hash key from messages."""
        content = str(messages)
        return hashlib.sha256(content.encode()).hexdigest()

    def _composite_key(self, messages: list[dict[str, str]], model_id: str) -> str:
        """Create a composite cache key from messages, namespace, and model_id."""
        return f"{self.namespace}|{self._hash_messages(messages)}|{model_id}"

    def _compute_embedding(self, messages: list[dict[str, str]]) -> list[float]:
        """Compute embedding for messages."""
        encoder = self._get_encoder()
        # Concatenate message contents
        text = " ".join(m.get("content", "") for m in messages)
        embedding = encoder.encode(text)
        return embedding.tolist()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def get(
        self,
        messages: list[dict[str, str]],
        model_id: str | None = None,
        semantic: bool = True,
    ) -> CacheEntry | None:
        """
        Look up cached response.

        Args:
            messages: Query messages.
            model_id: If provided, only return entries from this model.
                      Prevents model-mismatch cache returns that would
                      poison predictor feedback.
            semantic: Whether to try semantic matching.

        Returns:
            CacheEntry if found, None otherwise.
        """
        def _lookup() -> CacheEntry | None:
            # Try exact match first
            query_hash = self._hash_messages(messages)
            if model_id is not None:
                composite_key = self._composite_key(messages, model_id)
                if composite_key in self._exact_cache:
                    entry = self._exact_cache[composite_key]
                    if not entry.is_expired:
                        entry.hit_count += 1
                        return entry
                    else:
                        del self._exact_cache[composite_key]
            else:
                # Backward-compatible: model-unaware exact match.
                # Iterate composite keys to find any entry for this query hash.
                # Keys may be "hash|model_id" (legacy) or "ns|hash|model_id" (namespaced).
                for key, entry in list(self._exact_cache.items()):
                    sep_hash = "|" + query_hash + "|"
                    if key.startswith(query_hash + "|") or sep_hash in key:
                        if not entry.is_expired:
                            entry.hit_count += 1
                            return entry
                        else:
                            del self._exact_cache[key]

            # Try semantic match
            if semantic and self._semantic_entries:
                query_embedding = self._compute_embedding(messages)
                best_match: CacheEntry | None = None
                best_similarity = 0.0

                for entry in self._semantic_entries:
                    if entry.is_expired:
                        continue
                    if entry.query_embedding is None:
                        continue
                    if model_id is not None and entry.model_id != model_id:
                        continue
                    if entry.namespace != self.namespace:
                        continue

                    similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = entry

                if best_match:
                    best_match.hit_count += 1
                    return best_match

            return None

        if self._lock is not None:
            with self._lock:
                return _lookup()
        return _lookup()

    def put(
        self,
        messages: list[dict[str, str]],
        response: Any,
        model_id: str,
        semantic: bool = True,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> CacheEntry:
        """
        Store response in cache.

        Args:
            messages: Query messages.
            response: Response to cache.
            model_id: Model that generated the response.
            semantic: Whether to enable semantic matching.
            metadata: Additional metadata to store.
            namespace: Override namespace (defaults to self.namespace).

        Returns:
            The created cache entry.
        """
        ns = namespace or self.namespace
        query_hash = self._hash_messages(messages)
        composite_key = self._composite_key(messages, model_id)
        query_embedding = self._compute_embedding(messages) if semantic else None

        entry = CacheEntry(
            query_hash=query_hash,
            query_embedding=query_embedding,
            response=response,
            model_id=model_id,
            created_at=time.time(),
            ttl_seconds=self.ttl_seconds,
            metadata=metadata or {},
            namespace=ns,
        )

        def _store() -> None:
            # Store in exact cache with model-aware composite key
            self._exact_cache[composite_key] = entry

            # Store in semantic index if enabled
            if semantic and query_embedding:
                self._semantic_entries.append(entry)

            # Evict if over capacity
            self._evict_if_needed()

        if self._lock is not None:
            with self._lock:
                _store()
        else:
            _store()

        return entry

    def _evict_if_needed(self) -> None:
        """Evict entries if over capacity."""
        # Remove expired entries
        self._exact_cache = {
            k: v for k, v in self._exact_cache.items() if not v.is_expired
        }
        self._semantic_entries = [e for e in self._semantic_entries if not e.is_expired]

        # Evict oldest if still over capacity
        while len(self._exact_cache) > self.max_entries:
            oldest_key = min(self._exact_cache, key=lambda k: self._exact_cache[k].created_at)
            del self._exact_cache[oldest_key]

        while len(self._semantic_entries) > self.max_entries:
            oldest_idx = min(
                range(len(self._semantic_entries)),
                key=lambda i: self._semantic_entries[i].created_at,
            )
            self._semantic_entries.pop(oldest_idx)

    def invalidate(self, messages: list[dict[str, str]], model_id: str | None = None,
                   namespace: str | None = None) -> bool:
        """
        Invalidate cache entry for messages.

        Args:
            messages: Query messages to invalidate.
            model_id: If provided, only invalidate entries for this model.
            namespace: If provided, only invalidate entries in this namespace.

        Returns:
            True if entry was found and removed.
        """
        ns = namespace or self.namespace
        query_hash = self._hash_messages(messages)
        ns_prefix = f"{ns}|{query_hash}|"
        found = False

        if model_id is not None:
            composite_key = f"{ns_prefix}{model_id}"
            if composite_key in self._exact_cache:
                del self._exact_cache[composite_key]
                found = True
            self._semantic_entries = [
                e for e in self._semantic_entries
                if not (e.query_hash == query_hash and e.model_id == model_id and e.namespace == ns)
            ]
        else:
            keys_to_remove = [k for k in self._exact_cache if k.startswith(ns_prefix)]
            for k in keys_to_remove:
                del self._exact_cache[k]
                found = True
            self._semantic_entries = [
                e for e in self._semantic_entries
                if not (e.query_hash == query_hash and e.namespace == ns)
            ]

        return found

    def clear(self, namespace: str | None = None) -> None:
        """Clear cache entries, optionally for a specific namespace."""
        if namespace is None:
            self._exact_cache.clear()
            self._semantic_entries.clear()
        else:
            ns_prefix = f"{namespace}|"
            self._exact_cache = {k: v for k, v in self._exact_cache.items() if not k.startswith(ns_prefix)}
            self._semantic_entries = [e for e in self._semantic_entries if e.namespace != namespace]

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self._exact_cache.values())
        return {
            "exact_entries": len(self._exact_cache),
            "semantic_entries": len(self._semantic_entries),
            "total_hits": total_hits,
        }

    def __len__(self) -> int:
        return len(self._exact_cache)
