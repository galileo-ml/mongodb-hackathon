"""Composable audio processing pipeline for diarization."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np

from ..core import AudioChunk, AudioSegment, VectorSimilarityResult
from .denoiser import AdaptiveDenoiser
from .segmenter import WebRTCVADSegmenter
from ..services.vector_store import MongoDBVectorStore

logger = logging.getLogger("webrtc.audio.pipeline")


@dataclass
class PipelineConfig:
    """Configuration options for the audio pipeline."""

    min_segment_seconds: float = 1.5
    tail_keep_seconds: float = 0.5
    similarity_limit: int = 3
    min_segment_rms: float = 800.0
    min_embedding_ms: int = 600
    vad_mode: int = 3
    vad_min_speech_ms: int = 400
    vad_min_silence_ms: int = 400


class AudioPipeline:
    """Orchestrates denoising, segmentation, embedding and storage."""

    def __init__(
        self,
        denoiser: AdaptiveDenoiser,
        segmenter: WebRTCVADSegmenter,
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
        self._consumed_samples: defaultdict[str, int] = defaultdict(int)
        self._last_segment_end_ms: defaultdict[str, float] = defaultdict(float)

    async def process_chunk(self, chunk: AudioChunk) -> List[VectorSimilarityResult]:
        logger.debug(
            "Buffering audio chunk session=%s sample_rate=%d size=%d",
            chunk.session_id,
            chunk.sample_rate,
            len(chunk.data),
        )
        session_id = chunk.session_id
        denoised = await self.denoiser.denoise(chunk)
        buf = self._buffers[session_id]
        buf.extend(denoised.payload)

        sample_rate = denoised.sample_rate
        required_samples = int(sample_rate * self.config.min_segment_seconds)
        current_samples = len(buf) // 2
        if current_samples < required_samples:
            return []

        payload = bytes(buf)
        start_offset_samples = self._consumed_samples[session_id]
        start_offset_ms = int(start_offset_samples / sample_rate * 1000)
        duration_ms = int(current_samples / sample_rate * 1000)
        segment = AudioSegment(
            session_id=session_id,
            sample_rate=sample_rate,
            start_ms=start_offset_ms,
            end_ms=start_offset_ms + duration_ms,
            energy=0.0,
            payload=payload,
        )
        all_matches = await self._analyze(segment)

        tail_samples = int(sample_rate * self.config.tail_keep_seconds)
        if tail_samples <= 0:
            buf.clear()
            self._buffers[session_id] = bytearray()
        else:
            retain_bytes = tail_samples * 2
            retain_bytes = min(retain_bytes, len(buf))
            self._buffers[session_id] = bytearray(buf[-retain_bytes:])

        consumed_increment = max(current_samples - tail_samples, 0)
        self._consumed_samples[session_id] += consumed_increment
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
        start_offset_samples = self._consumed_samples[session_id]
        start_offset_ms = int(start_offset_samples / sample_rate * 1000)
        duration_ms = int(samples / sample_rate * 1000)
        segment = AudioSegment(
            session_id=session_id,
            sample_rate=sample_rate,
            start_ms=start_offset_ms,
            end_ms=start_offset_ms + duration_ms,
            energy=0.0,
            payload=payload,
        )
        self._buffers.pop(session_id, None)
        self._consumed_samples.pop(session_id, None)
        return await self._analyze(segment)

    async def _analyze(
        self, segment: AudioSegment
    ) -> List[VectorSimilarityResult]:
        sample_ranges = await self.segmenter.segment(segment)
        all_matches: list[VectorSimilarityResult] = []
        sr = segment.sample_rate
        raw = memoryview(segment.payload)
        last_end = self._last_segment_end_ms[segment.session_id]

        min_samples_for_embed = int(sr * (self.config.min_embedding_ms / 1000))

        for start_sample, end_sample in sample_ranges:
            start_ms = segment.start_ms + int(start_sample / sr * 1000)
            end_ms = segment.start_ms + int(end_sample / sr * 1000)
            if end_ms <= last_end:
                continue
            if start_ms < last_end:
                # Trim overlap to avoid re-embedding the same speech
                overlap_samples = max(
                    int((last_end - segment.start_ms) / 1000 * sr),
                    0,
                )
                trimmed_start = max(overlap_samples, start_sample)
                start_sample = trimmed_start
                start_ms = segment.start_ms + int(start_sample / sr * 1000)
                if start_sample >= end_sample:
                    continue

            payload_bytes = raw[start_sample * 2 : end_sample * 2].tobytes()
            segment_samples = np.frombuffer(payload_bytes, dtype=np.int16)
            rms = float(np.sqrt(np.mean(np.square(segment_samples.astype(np.float32))))) if segment_samples.size else 0.0
            if rms < self.config.min_segment_rms:
                logger.debug(
                    "Skipping low-RMS segment session=%s rms=%.2f start_ms=%d end_ms=%d",
                    segment.session_id,
                    rms,
                    start_ms,
                    end_ms,
                )
                last_end = max(last_end, end_ms)
                continue

            if (end_sample - start_sample) < min_samples_for_embed:
                logger.debug(
                    "Skipping short segment session=%s length_ms=%d",
                    segment.session_id,
                    end_ms - start_ms,
                )
                last_end = max(last_end, end_ms)
                continue

            payload = payload_bytes
            child_segment = AudioSegment(
                session_id=segment.session_id,
                sample_rate=sr,
                start_ms=start_ms,
                end_ms=end_ms,
                energy=0.0,
                payload=payload,
            )

            embedding = await self.embedder.embed(child_segment)
            await self.vector_store.upsert_embedding(embedding)
            matches = await self.vector_store.query_similar(
                embedding, limit=self.config.similarity_limit
            )
            all_matches.extend(matches)
            last_end = max(last_end, end_ms)

        self._last_segment_end_ms[segment.session_id] = last_end
        logger.info(
            "Processed segment session=%s window_ms=%d speech_segments=%d matches=%d",
            segment.session_id,
            segment.end_ms - segment.start_ms,
            len(sample_ranges),
            len(all_matches),
        )
        return all_matches
