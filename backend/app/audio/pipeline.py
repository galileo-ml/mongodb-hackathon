"""Composable audio processing pipeline for diarization."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..core import AudioChunk, AudioSegment, SpeakerEmbedding, VectorSimilarityResult
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
    conversation_timeout_seconds: float = 10.0
    local_similarity_threshold: float = 0.8
    global_similarity_threshold: float = 0.85


@dataclass
class LocalSpeakerState:
    local_id: str
    centroid: np.ndarray
    count: int = 1
    global_id: str | None = None
    last_seen_ts: float = field(default_factory=time.time)


@dataclass
class ConversationState:
    conversation_id: str
    started_ts: float
    last_speech_ts: float | None
    last_chunk_ts: float
    local_speakers: Dict[str, LocalSpeakerState] = field(default_factory=dict)
    next_local_idx: int = 1


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
        self._conversation_states: Dict[str, ConversationState] = {}
        self._conversation_seq: defaultdict[str, int] = defaultdict(int)

    async def process_chunk(self, chunk: AudioChunk) -> List[VectorSimilarityResult]:
        logger.debug(
            "Buffering audio chunk session=%s sample_rate=%d size=%d",
            chunk.session_id,
            chunk.sample_rate,
            len(chunk.data),
        )
        session_id = chunk.session_id
        state = self._ensure_conversation_state(session_id, chunk.timestamp)
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
        event_ts = chunk.timestamp.timestamp()
        all_matches, speech_detected = await self._analyze(segment, state, event_ts)
        if speech_detected:
            state.last_speech_ts = event_ts
        state.last_chunk_ts = event_ts

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
        self._last_segment_end_ms.pop(session_id, None)
        state = self._conversation_states.get(session_id)
        event_ts = time.time()
        if state is None:
            return []
        matches, speech_detected = await self._analyze(segment, state, event_ts)
        if speech_detected:
            state.last_speech_ts = event_ts
        await self._end_conversation(session_id, state, event_ts, reason="session flush")
        return matches

    async def _analyze(
        self, segment: AudioSegment, state: ConversationState, event_ts: float
    ) -> Tuple[List[VectorSimilarityResult], bool]:
        sample_ranges = await self.segmenter.segment(segment)
        all_matches: list[VectorSimilarityResult] = []
        speech_detected = False
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
            matches = await self._handle_embedding(state, embedding, event_ts)
            all_matches.extend(matches)
            speech_detected = True
            last_end = max(last_end, end_ms)

        self._last_segment_end_ms[segment.session_id] = last_end
        logger.info(
            "Processed segment session=%s window_ms=%d speech_segments=%d matches=%d",
            segment.session_id,
            segment.end_ms - segment.start_ms,
            len(sample_ranges),
            len(all_matches),
        )
        return all_matches, speech_detected

    async def _handle_embedding(
        self,
        state: ConversationState,
        embedding: SpeakerEmbedding,
        event_ts: float,
    ) -> List[VectorSimilarityResult]:

        vector = np.asarray(embedding.vector, dtype=np.float32).flatten()
        local_state, local_score = self._match_local_speaker(state, vector)

        if local_state:
            self._update_local_speaker(local_state, vector, event_ts)
            if local_state.global_id:
                await self.vector_store.upsert_identity_embedding(
                    local_state.global_id, embedding
                )
                logger.info(
                    "Conversation %s speaker %s reused identity %s (local similarity=%.3f)",
                    state.conversation_id,
                    local_state.local_id,
                    local_state.global_id,
                    local_score,
                )
                return []

        if local_state is None:
            if local_score >= 0:
                logger.debug(
                    "Conversation %s no local match (best similarity=%.3f < %.3f)",
                    state.conversation_id,
                    local_score,
                    self.config.local_similarity_threshold,
                )
            local_state = self._create_local_speaker(state, vector, event_ts)
        elif local_state.global_id is None:
            logger.debug(
                "Conversation %s speaker %s has no global identity; resolving",
                state.conversation_id,
                local_state.local_id,
            )

        global_id, assignment, matched = await self._resolve_global_identity(
            state, local_state, embedding
        )
        local_state.global_id = global_id
        local_state.last_seen_ts = event_ts

        if matched:
            logger.info(
                "Conversation %s speaker %s matched identity %s score=%.3f",
                state.conversation_id,
                local_state.local_id,
                global_id,
                assignment[0].score,
            )
        else:
            logger.info(
                "Conversation %s speaker %s registered new identity %s",
                state.conversation_id,
                local_state.local_id,
                global_id,
            )
        return assignment

    def _match_local_speaker(
        self, state: ConversationState, vector: np.ndarray
    ) -> Tuple[LocalSpeakerState | None, float]:
        best_state: LocalSpeakerState | None = None
        best_score = -1.0
        for speaker in state.local_speakers.values():
            score = self._cosine_similarity(vector, speaker.centroid)
            if score > best_score:
                best_score = score
                best_state = speaker
        if best_state and best_score >= self.config.local_similarity_threshold:
            return best_state, best_score
        return None, best_score

    def _update_local_speaker(
        self, speaker: LocalSpeakerState, vector: np.ndarray, event_ts: float
    ) -> None:
        speaker.centroid = (
            speaker.centroid * speaker.count + vector
        ) / (speaker.count + 1)
        speaker.centroid = speaker.centroid.astype(np.float32)
        speaker.count += 1
        speaker.last_seen_ts = event_ts

    def _create_local_speaker(
        self, state: ConversationState, vector: np.ndarray, event_ts: float
    ) -> LocalSpeakerState:
        local_id = f"{state.conversation_id}-sp{state.next_local_idx}"
        state.next_local_idx += 1
        speaker = LocalSpeakerState(
            local_id=local_id,
            centroid=vector.astype(np.float32).copy(),
            last_seen_ts=event_ts,
        )
        state.local_speakers[local_id] = speaker
        logger.info(
            "Conversation %s detected new local speaker %s",
            state.conversation_id,
            local_id,
        )
        return speaker

    async def _resolve_global_identity(
        self,
        state: ConversationState,
        local_state: LocalSpeakerState,
        embedding: SpeakerEmbedding,
    ) -> Tuple[str, List[VectorSimilarityResult], bool]:
        results = await self.vector_store.query_similar_global(
            embedding, limit=self.config.similarity_limit
        )
        matched_existing = False
        if (
            results
            and results[0].matched_person_id
            and results[0].score >= self.config.global_similarity_threshold
        ):
            global_id = results[0].matched_person_id  # type: ignore[assignment]
            score = results[0].score
            matched_existing = True
        elif results and results[0].matched_person_id:
            candidate_id = results[0].matched_person_id
            score = results[0].score
            logger.info(
                "Conversation %s candidate identity %s score=%.3f below threshold %.2f",
                state.conversation_id,
                candidate_id,
                score,
                self.config.global_similarity_threshold,
            )
            global_id = self._generate_global_identity()
        else:
            global_id = self._generate_global_identity()
            score = 0.0

        await self.vector_store.upsert_identity_embedding(global_id, embedding)
        assignment = VectorSimilarityResult(
            matched_person_id=global_id,
            score=score,
            embedding=embedding,
        )
        return global_id, [assignment], matched_existing

    def _ensure_conversation_state(
        self, session_id: str, chunk_timestamp
    ) -> ConversationState:
        ts = chunk_timestamp.timestamp()
        state = self._conversation_states.get(session_id)
        if state is None:
            return self._start_new_conversation(session_id, ts, reason="session start")

        state.last_chunk_ts = ts
        if (
            state.last_speech_ts is not None
            and ts - state.last_speech_ts >= self.config.conversation_timeout_seconds
        ):
            self._end_conversation(session_id, state, ts, reason="silence timeout")
            return self._start_new_conversation(
                session_id, ts, reason="silence timeout"
            )

        return state

    def _start_new_conversation(
        self, session_id: str, ts: float, reason: str
    ) -> ConversationState:
        self._conversation_seq[session_id] += 1
        conv_id = f"{session_id}-conv{self._conversation_seq[session_id]}"
        state = ConversationState(
            conversation_id=conv_id,
            started_ts=ts,
            last_speech_ts=None,
            last_chunk_ts=ts,
        )
        self._conversation_states[session_id] = state
        self._buffers[session_id] = bytearray()
        self._consumed_samples[session_id] = 0
        self._last_segment_end_ms[session_id] = 0
        logger.info(
            "Starting conversation %s for session=%s (reason=%s)",
            conv_id,
            session_id,
            reason,
        )
        return state

    def _end_conversation(
        self,
        session_id: str,
        state: ConversationState,
        ts: float,
        reason: str,
    ) -> None:
        duration = 0.0
        if state.last_speech_ts is not None:
            duration = state.last_speech_ts - state.started_ts
        idle_gap = ts - (state.last_speech_ts or state.last_chunk_ts)
        speaker_count = len(state.local_speakers)
        logger.info(
            "Ending conversation %s for session=%s (reason=%s, duration=%.2fs, idle_gap=%.2fs, speakers=%d)",
            state.conversation_id,
            session_id,
            reason,
            duration,
            idle_gap,
            speaker_count,
        )
        self._conversation_states.pop(session_id, None)
        self._buffers.pop(session_id, None)
        self._consumed_samples.pop(session_id, None)
        self._last_segment_end_ms.pop(session_id, None)

    @staticmethod
    def _generate_global_identity() -> str:
        return f"person-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
