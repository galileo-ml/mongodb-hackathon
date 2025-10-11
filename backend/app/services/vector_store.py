"""MongoDB Atlas vector store stubs."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import DefaultDict, List

from ..core import SpeakerEmbedding, VectorSimilarityResult

logger = logging.getLogger("webrtc.vector_store")


class MongoDBVectorStore:
    """In-memory stub mimicking MongoDB Atlas vector search collection."""

    def __init__(self, uri: str, database: str, collection: str) -> None:
        self.uri = uri
        self.database = database
        self.collection = collection
        self._store: DefaultDict[str, list[SpeakerEmbedding]] = defaultdict(list)

    async def connect(self) -> None:
        """Pretend to establish a connection to MongoDB Atlas."""

        logger.info(
            "[stub] Connecting to MongoDB Atlas at %s/%s.%s",
            self.uri,
            self.database,
            self.collection,
        )
        await asyncio.sleep(0)

    async def upsert_embedding(self, embedding: SpeakerEmbedding) -> None:
        """Persist the embedding in the stub store."""

        self._store[embedding.session_id].append(embedding)
        logger.info(
            "[stub] Stored embedding for session=%s segment=%s vector_dim=%d",
            embedding.session_id,
            embedding.segment_id,
            len(embedding.vector),
        )

    async def query_similar(
        self, embedding: SpeakerEmbedding, limit: int = 3
    ) -> List[VectorSimilarityResult]:
        """Return naive cosine-like similarity scores from stub memory."""

        existing = self._store.get(embedding.session_id, [])
        results: list[VectorSimilarityResult] = []
        for stored in existing:
            score = self._cosine_proxy(embedding.vector, stored.vector)
            results.append(
                VectorSimilarityResult(
                    matched_person_id=None,
                    score=score,
                    embedding=stored,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        trimmed = results[:limit]
        logger.info(
            "[stub] query returned %d candidates (limit=%d)", len(trimmed), limit
        )
        return trimmed

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
