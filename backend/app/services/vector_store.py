"""MongoDB Atlas vector store stubs."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple

from ..core import SpeakerEmbedding, VectorSimilarityResult

logger = logging.getLogger("webrtc.vector_store")


class MongoDBVectorStore:
    """In-memory stub mimicking MongoDB Atlas vector search collection."""

    def __init__(self, uri: str, database: str, collection: str) -> None:
        self.uri = uri
        self.database = database
        self.collection = collection
        self._store: DefaultDict[str, list[SpeakerEmbedding]] = defaultdict(list)
        self._metrics_lock = asyncio.Lock()
        self._query_count = 0
        self._unique_match_ids: Set[str] = set()
        self._embedding_total = 0

    async def connect(self) -> None:
        """Pretend to establish a connection to MongoDB Atlas."""

        logger.info(
            "[stub] Connecting to MongoDB Atlas at %s/%s.%s",
            self.uri,
            self.database,
            self.collection,
        )
        await asyncio.sleep(0)

    async def upsert_identity_embedding(
        self, person_id: str, embedding: SpeakerEmbedding
    ) -> None:
        """Persist embedding under a global identity."""

        stored = embedding.model_copy()
        self._store[person_id].append(stored)
        async with self._metrics_lock:
            self._embedding_total += 1
        logger.info(
            "[stub] Stored embedding for identity=%s segment=%s vector_dim=%d",
            person_id,
            embedding.segment_id,
            len(embedding.vector),
        )

    async def query_similar_global(
        self, embedding: SpeakerEmbedding, limit: int = 3
    ) -> List[VectorSimilarityResult]:
        """Return cosine similarity scores against all stored identities."""

        results: list[VectorSimilarityResult] = []
        for person_id, embeddings in self._store.items():
            for stored in embeddings:
                score = self._cosine_proxy(embedding.vector, stored.vector)
                results.append(
                    VectorSimilarityResult(
                        matched_person_id=person_id,
                        score=score,
                        embedding=stored,
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        trimmed = results[:limit]
        async with self._metrics_lock:
            self._query_count += 1
            self._unique_match_ids.update(
                r.matched_person_id for r in trimmed if r.matched_person_id
            )
        logger.info(
            "[stub] query returned %d candidates (limit=%d)", len(trimmed), limit
        )
        return trimmed

    async def snapshot_metrics(self) -> Tuple[int, int, int]:
        """Return and reset lookup metrics.

        Returns a tuple of (lookup_count, unique_matches, total_embeddings).
        """

        async with self._metrics_lock:
            lookups = self._query_count
            unique = len(self._unique_match_ids)
            total = self._embedding_total
            self._query_count = 0
            self._unique_match_ids.clear()
        return lookups, unique, total

    @staticmethod
    def _cosine_proxy(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        numerator = sum(x * y for x, y in zip(a, b))
        denom_a = sum(x * x for x in a) ** 0.5
        denom_b = sum(y * y for y in b) ** 0.5
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / (denom_a * denom_b)
