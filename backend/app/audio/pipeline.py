"""Composable audio processing pipeline for diarization."""

from __future__ import annotations

import logging
from typing import List

from ..core import AudioChunk, VectorSimilarityResult
from .denoiser import AdaptiveDenoiser
from .embeddings import SpeakerEmbedder
from .segmenter import VoiceActivitySegmenter
from ..services.vector_store import MongoDBVectorStore

logger = logging.getLogger("webrtc.audio.pipeline")


class AudioPipeline:
    """Orchestrates denoising, segmentation, embedding and storage."""

    def __init__(
        self,
        denoiser: AdaptiveDenoiser,
        segmenter: VoiceActivitySegmenter,
        embedder: SpeakerEmbedder,
        vector_store: MongoDBVectorStore,
    ) -> None:
        self.denoiser = denoiser
        self.segmenter = segmenter
        self.embedder = embedder
        self.vector_store = vector_store

    async def process_chunk(self, chunk: AudioChunk) -> List[VectorSimilarityResult]:
        logger.info(
            "Processing audio chunk session=%s sample_rate=%d size=%d",
            chunk.session_id,
            chunk.sample_rate,
            len(chunk.data),
        )
        segment = await self.denoiser.denoise(chunk)
        segments = await self.segmenter.segment(segment)
        all_matches: list[VectorSimilarityResult] = []
        for seg in segments:
            embedding = await self.embedder.embed(seg)
            await self.vector_store.upsert_embedding(embedding)
            matches = await self.vector_store.query_similar(embedding, limit=3)
            all_matches.extend(matches)
        return all_matches
