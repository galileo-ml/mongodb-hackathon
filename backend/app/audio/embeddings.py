"""Speaker embedding extraction stubs."""

from __future__ import annotations

import logging

from ..core import AudioSegment, SpeakerEmbedding

logger = logging.getLogger("webrtc.audio.embedder")


class SpeakerEmbedder:
    """Placeholder embedder using deterministic mock vectors."""

    def __init__(self, vector_dim: int = 256) -> None:
        self.vector_dim = vector_dim

    async def embed(self, segment: AudioSegment) -> SpeakerEmbedding:
        """Return a deterministic pseudo-embedding for the provided segment."""

        logger.debug(
            "Generating embedding for session=%s segment=(%s,%s)",
            segment.session_id,
            segment.start_ms,
            segment.end_ms,
        )
        # Pseudo-random but deterministic vector using simple byte hashing.
        vector = [
            (segment.payload[i % len(segment.payload)] / 255.0 if segment.payload else 0.0)
            for i in range(self.vector_dim)
        ]
        segment_id = f"{segment.session_id}:{segment.start_ms}-{segment.end_ms}"
        return SpeakerEmbedding(
            session_id=segment.session_id,
            segment_id=segment_id,
            vector=vector,
        )
