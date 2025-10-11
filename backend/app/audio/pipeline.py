"""Composable audio processing pipeline for diarization."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from ..core import AudioChunk, AudioSegment, VectorSimilarityResult
from .denoiser import AdaptiveDenoiser
from .segmenter import VoiceActivitySegmenter
from ..services.vector_store import MongoDBVectorStore

logger = logging.getLogger("webrtc.audio.pipeline")


@dataclass
class PipelineConfig:
    """Configuration options for the audio pipeline."""

    min_segment_seconds: float = 1.5
    tail_keep_seconds: float = 0.5
    similarity_limit: int = 3


class AudioPipeline:
    """Orchestrates denoising, segmentation, embedding and storage."""

    def __init__(
        self,
        denoiser: AdaptiveDenoiser,
        segmenter: VoiceActivitySegmenter,
        embedder,
        vector_store: MongoDBVectorStore,
        config: PipelineConfig | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.segmenter = segmenter
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or PipelineConfig()
        self._buffers: defaultdict[str, bytearray] = defaultdict(bytearray)

    async def process_chunk(self, chunk: AudioChunk) -> List[VectorSimilarityResult]:
        logger.debug(
            "Buffering audio chunk session=%s sample_rate=%d size=%d",
            chunk.session_id,
            chunk.sample_rate,
            len(chunk.data),
        )
        buf = self._buffers[chunk.session_id]
        buf.extend(chunk.data)

        required_samples = int(
            chunk.sample_rate * self.config.min_segment_seconds
        )
        current_samples = len(buf) // 2
        if current_samples < required_samples:
            return []

        payload = bytes(buf)
        duration_ms = int(current_samples / chunk.sample_rate * 1000)
        segment = AudioSegment(
            session_id=chunk.session_id,
            sample_rate=chunk.sample_rate,
            start_ms=0,
            end_ms=duration_ms,
            energy=0.0,
            payload=payload,
        )
        all_matches = await self._analyze(segment)

        tail_samples = int(chunk.sample_rate * self.config.tail_keep_seconds)
        if tail_samples <= 0:
            buf.clear()
        else:
            retain_bytes = tail_samples * 2
            self._buffers[chunk.session_id] = bytearray(buf[-retain_bytes:])

        return all_matches

    async def flush_session(
        self, session_id: str, sample_rate: int
    ) -> List[VectorSimilarityResult]:
        buf = self._buffers.get(session_id)
        if not buf:
            return []
        payload = bytes(buf)
        if not payload:
            return []

        samples = len(payload) // 2
        duration_ms = int(samples / sample_rate * 1000)
        segment = AudioSegment(
            session_id=session_id,
            sample_rate=sample_rate,
            start_ms=0,
            end_ms=duration_ms,
            energy=0.0,
            payload=payload,
        )
        self._buffers.pop(session_id, None)
        return await self._analyze(segment)

    async def _analyze(
        self, segment: AudioSegment
    ) -> List[VectorSimilarityResult]:
        segments = await self.segmenter.segment(segment)
        all_matches: list[VectorSimilarityResult] = []
        for seg in segments:
            embedding = await self.embedder.embed(seg)
            await self.vector_store.upsert_embedding(embedding)
            matches = await self.vector_store.query_similar(
                embedding, limit=self.config.similarity_limit
            )
            all_matches.extend(matches)
        logger.info(
            "Processed segment session=%s duration_ms=%d matches=%d",
            segment.session_id,
            segment.end_ms - segment.start_ms,
            len(all_matches),
        )
        return all_matches
