"""Audio pipeline that buffers conversations and transcribes them with Whisper."""

from __future__ import annotations

import asyncio
import logging
import uuid
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core import AudioChunk, ConversationEvent, ConversationUtterance
from ..services.conversation_stream import ConversationEventBus
from .denoiser import AdaptiveDenoiser

try:  # pragma: no cover - optional dependency
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

try:  # pragma: no cover - optional dependency
    from torchaudio.functional import resample as ta_resample
except ImportError:  # pragma: no cover
    ta_resample = None

try:  # pragma: no cover - optional dependency
    import webrtcvad
except ImportError:  # pragma: no cover
    webrtcvad = None

try:  # pragma: no cover - optional dependency
    from pyannote.audio import Inference as PyannoteInference, Model as PyannoteModel
except ImportError:  # pragma: no cover
    PyannoteInference = None
    PyannoteModel = None

logger = logging.getLogger("webrtc.audio.pipeline")


@dataclass
class PipelineConfig:
    target_sample_rate: int = 16_000
    silence_timeout_seconds: float = 2.0
    min_conversation_seconds: float = 2.0
    vad_aggressiveness: int = 3
    transcription_model: str = "large-v3"
    min_speech_rms: float = 0.01
    noise_floor_smoothing: float = 0.9
    noise_gate_margin: float = 0.005
    embedding_model: str = "pyannote/embedding"
    speaker_match_threshold: float = 0.25
    embedding_window_seconds: float = 0.8


@dataclass
class ConversationState:
    conversation_id: str
    started_ts: float
    last_audio_ts: float
    last_speech_ts: float | None
    noise_floor_rms: float | None = None
    has_speech: bool = False
    chunks: List[np.ndarray] | None = None
    last_speaker_id: str | None = None

    def __post_init__(self) -> None:
        if self.chunks is None:
            self.chunks = []


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class SpeakerProfile:
    speaker_id: str
    embedding: np.ndarray
    count: int = 1


class AudioPipeline:
    """Buffers audio per WebRTC session and transcribes completed conversations."""

    def __init__(
        self,
        denoiser: AdaptiveDenoiser,
        config: PipelineConfig | None = None,
        conversation_bus: ConversationEventBus | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.config = config or PipelineConfig()
        self._conversations: Dict[str, ConversationState] = {}
        self._whisper_model = None
        self._whisper_lock = asyncio.Lock()
        self._vad = webrtcvad.Vad(self.config.vad_aggressiveness) if webrtcvad else None
        self._pyannote_inference = None
        self._speaker_profiles: List[SpeakerProfile] = []
        self._speaker_lock = asyncio.Lock()
        self._next_speaker_index = 10
        self._pyannote_auth_token = os.getenv("PYANNOTE_AUTH_TOKEN")
        self._conversation_bus = conversation_bus

        if PyannoteInference is None:
            logger.warning(
                "pyannote.audio is not installed; speaker attribution will remain disabled"
            )
        else:
            try:
                if PyannoteModel is not None:
                    if not self._pyannote_auth_token:
                        logger.warning(
                            "PYANNOTE_AUTH_TOKEN not set; gated models may fail to download"
                        )
                    model = PyannoteModel.from_pretrained(
                        self.config.embedding_model,
                        use_auth_token=self._pyannote_auth_token,
                    )
                    self._pyannote_inference = PyannoteInference(
                        model=model,
                        window="whole",
                    )
                else:
                    kwargs = {
                        "pretrained": self.config.embedding_model,
                        "window": "whole",
                    }
                    if self._pyannote_auth_token:
                        kwargs["use_auth_token"] = self._pyannote_auth_token
                    self._pyannote_inference = PyannoteInference(**kwargs)
                logger.info(
                    "Loaded pyannote embedding model '%s'",
                    self.config.embedding_model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to initialize pyannote embedding model '%s': %s",
                    self.config.embedding_model,
                    exc,
                )
                self._pyannote_inference = None

    async def process_chunk(self, chunk: AudioChunk) -> None:
        session_id = chunk.session_id
        state = self._ensure_conversation(session_id, chunk.timestamp.timestamp())

        denoised = await self.denoiser.denoise(chunk)
        audio = self._convert_to_target_sr(denoised.payload, denoised.sample_rate)
        if audio.size == 0:
            logger.debug("Session %s chunk had no audio data", session_id)
            return

        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        has_speech = self._chunk_has_speech(audio)
        effective_threshold = self.config.min_speech_rms
        if state.noise_floor_rms is not None:
            effective_threshold = max(
                effective_threshold,
                state.noise_floor_rms + self.config.noise_gate_margin,
            )
        if has_speech and rms < effective_threshold:
            logger.debug(
                "Session %s chunk suppressed by RMS gate (rms=%.5f threshold=%.5f)",
                session_id,
                rms,
                effective_threshold,
            )
            has_speech = False
        logger.debug(
            "Session %s chunk rms=%.5f has_speech=%s", session_id, rms, has_speech
        )

        state.chunks.append(audio)
        state.last_audio_ts = chunk.timestamp.timestamp()

        now = chunk.timestamp.timestamp()
        if not has_speech:
            smoothing = min(max(self.config.noise_floor_smoothing, 0.0), 0.999)
            if state.noise_floor_rms is None:
                state.noise_floor_rms = rms
            else:
                state.noise_floor_rms = (
                    state.noise_floor_rms * smoothing
                    + rms * (1.0 - smoothing)
                )
        if has_speech:
            state.last_speech_ts = now
            state.has_speech = True
            logger.info(
                "Session %s conversation %s detected speech (rms=%.4f)",
                session_id,
                state.conversation_id,
                rms,
            )
        elif state.has_speech and state.last_speech_ts is not None:
            elapsed = now - state.last_speech_ts
            if elapsed >= self.config.silence_timeout_seconds:
                logger.info(
                    "Session %s conversation %s reached silence timeout (%.2fs)",
                    session_id,
                    state.conversation_id,
                    elapsed,
                )
                await self._finalize_conversation(session_id, "silence timeout")

    async def flush_session(self, session_id: str, sample_rate: int) -> None:  # noqa: ARG002
        if session_id in self._conversations:
            await self._finalize_conversation(session_id, "session flush")

    async def warm_whisper(self) -> None:
        if whisper is None:
            logger.info("Whisper not installed; skipping warm-up")
            return
        await self._load_whisper_model()

    def _ensure_conversation(self, session_id: str, ts: float) -> ConversationState:
        state = self._conversations.get(session_id)
        if state is None:
            conv_id = f"{session_id}-conv{uuid.uuid4().hex[:6]}"
            state = ConversationState(
                conversation_id=conv_id,
                started_ts=ts,
                last_audio_ts=ts,
                last_speech_ts=None,
            )
            self._conversations[session_id] = state
            logger.info(
                "Starting conversation %s for session=%s",
                conv_id,
                session_id,
            )
        return state

    async def _finalize_conversation(self, session_id: str, reason: str) -> None:
        state = self._conversations.pop(session_id, None)
        if state is None:
            return

        duration = state.last_audio_ts - state.started_ts
        if not state.has_speech or duration < self.config.min_conversation_seconds:
            logger.info(
                "Discarding conversation %s for session=%s (reason=%s, duration=%.2fs, has_speech=%s)",
                state.conversation_id,
                session_id,
                reason,
                duration,
                state.has_speech,
            )
            return

        audio = np.concatenate(state.chunks) if state.chunks else np.array([], dtype=np.float32)
        if audio.size == 0:
            logger.info(
                "Conversation %s for session=%s had no samples after buffering",
                state.conversation_id,
                session_id,
            )
            return

        transcript = await self._transcribe_audio(audio)
        await self._assign_speakers(state, session_id, audio, transcript)
        await self._publish_conversation_event(state, session_id, transcript)
        if transcript:
            self._print_transcript(state.conversation_id, transcript)
        else:
            logger.info(
                "Conversation %s for session=%s produced no transcription",
                state.conversation_id,
                session_id,
            )
        logger.info(
            "Ending conversation %s for session=%s (reason=%s, duration=%.2fs)",
            state.conversation_id,
            session_id,
            reason,
            duration,
        )

    async def _transcribe_audio(self, audio: np.ndarray) -> List[TranscriptSegment]:
        if whisper is None:
            logger.warning("Whisper not installed; skipping transcription")
            return []

        model = await self._load_whisper_model()
        if model is None:
            return []

        try:
            result = await asyncio.to_thread(
                model.transcribe,
                audio,
                fp16=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper transcription failed: %s", exc)
            return []

        segments = result.get("segments") or []
        snippets: List[TranscriptSegment] = []
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if end <= start:
                duration = float(seg.get("duration", 0.0))
                end = start + max(duration, 0.0)
            text = (seg.get("text") or "").strip()
            if text:
                snippets.append(TranscriptSegment(start=start, end=end, text=text))
        if not snippets and result.get("text"):
            total_duration = audio.size / float(self.config.target_sample_rate)
            snippets.append(
                TranscriptSegment(
                    start=0.0,
                    end=total_duration,
                    text=result["text"].strip(),
                )
            )
        return snippets

    def _print_transcript(
        self, conversation_id: str, snippets: List[TranscriptSegment]
    ) -> None:
        logger.info(
            "Publishing conversation summary for %s (%d snippets)",
            conversation_id,
            len(snippets),
        )
        for segment in snippets:
            minutes, seconds = divmod(segment.start, 60.0)
            millis = int(round((seconds - int(seconds)) * 100))
            timestamp = f"{int(minutes):02d}:{int(seconds)%60:02d}.{millis:02d}"
            speaker_label = segment.speaker or "speaker_unknown"
            line = f"[{timestamp}] {speaker_label}: {segment.text}"
            print(f"\033[31m{line}\033[0m", flush=True)

    async def _publish_person_detected(
        self,
        session_id: str,
        conversation_id: str,
        speaker_id: str,
        utterance: str | None = None,
        is_new: bool = False,
    ) -> None:
        if self._conversation_bus is None:
            return

        conversation: list[ConversationUtterance] = []
        if utterance:
            conversation.append(
                ConversationUtterance(
                    speaker=speaker_id,
                    text=utterance,
                )
            )

        event = ConversationEvent(
            event_type="PERSON_DETECTED",
            person_id=speaker_id,
            conversation_id=conversation_id,
            session_id=session_id,
            conversation=conversation,
        )

        try:
            await self._conversation_bus.publish(event)
            logger.info(
                "Published PERSON_DETECTED for %s (session=%s, conversation=%s, new=%s)",
                speaker_id,
                session_id,
                conversation_id,
                is_new,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to publish PERSON_DETECTED for %s: %s",
                speaker_id,
                exc,
            )

    async def _publish_conversation_event(
        self,
        state: ConversationState,
        session_id: str,
        snippets: List[TranscriptSegment],
    ) -> None:
        if not snippets or self._conversation_bus is None:
            return

        conversation = [
            ConversationUtterance(
                speaker=segment.speaker or "speaker_unknown",
                text=segment.text,
            )
            for segment in snippets
        ]

        primary_speaker = next(
            (entry.speaker for entry in conversation if entry.speaker != "speaker_unknown"),
            state.last_speaker_id,
        ) or "speaker_unknown"

        event = ConversationEvent(
            event_type="CONVERSATION_END",
            conversation_id=state.conversation_id,
            session_id=session_id,
            person_id=primary_speaker,
            conversation=conversation,
        )

        try:
            await self._conversation_bus.publish(event)
            logger.info(
                "Published conversation event %s with %d utterances",
                state.conversation_id,
                len(conversation),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Failed to publish conversation event %s: %s",
                state.conversation_id,
                exc,
            )

    async def _load_whisper_model(self):
        async with self._whisper_lock:
            if self._whisper_model is not None:
                return self._whisper_model
            try:
                model = await asyncio.to_thread(
                    whisper.load_model, self.config.transcription_model
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load Whisper model '%s': %s",
                    self.config.transcription_model,
                    exc,
                )
                self._whisper_model = None
                return None
            self._whisper_model = model
            logger.info(
                "Loaded Whisper model '%s' for transcription",
                self.config.transcription_model,
            )
            return self._whisper_model

    def _convert_to_target_sr(self, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        if not audio_bytes:
            return np.array([], dtype=np.float32)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        if sample_rate == self.config.target_sample_rate:
            return audio
        if ta_resample is None:
            duration = audio.shape[0] / sample_rate
            target_length = int(duration * self.config.target_sample_rate)
            if target_length <= 0:
                return np.array([], dtype=np.float32)
            x_old = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
            x_new = np.linspace(0, duration, num=target_length, endpoint=False)
            resampled = np.interp(x_new, x_old, audio)
            return resampled.astype(np.float32)
        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = ta_resample(tensor, sample_rate, self.config.target_sample_rate)
        return tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def _chunk_has_speech(self, audio: np.ndarray) -> bool:
        if self._vad is None or audio.size == 0:
            return True
        frame_samples = int(self.config.target_sample_rate * 0.02)
        pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        if pcm16.size < frame_samples:
            return False
        for start in range(0, pcm16.size - frame_samples + 1, frame_samples):
            frame = pcm16[start : start + frame_samples].tobytes()
            try:
                if self._vad.is_speech(frame, self.config.target_sample_rate):
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.debug("WebRTC VAD error: %s", exc)
                return True
        return False

    async def _assign_speakers(
        self,
        state: ConversationState,
        session_id: str,
        audio: np.ndarray,
        snippets: List[TranscriptSegment],
    ) -> None:
        if not snippets or audio.size == 0:
            return

        publish_queue: List[Tuple[str, Optional[str], bool]] = []

        if self._pyannote_inference is None:
            logger.warning(
                "pyannote embedding unavailable; using fallback speaker attribution"
            )
            async with self._speaker_lock:
                speaker_id = state.last_speaker_id
                is_new = False
                if speaker_id is None:
                    speaker_id = self._register_new_speaker(np.zeros(1, dtype=np.float32))
                    state.last_speaker_id = speaker_id
                    is_new = True

                for segment in snippets:
                    segment.speaker = speaker_id
                    publish_queue.append((speaker_id, segment.text, is_new))
                    is_new = False

            for speaker_id, utterance, is_new in publish_queue:
                await self._publish_person_detected(
                    session_id=session_id,
                    conversation_id=state.conversation_id,
                    speaker_id=speaker_id,
                    utterance=utterance,
                    is_new=is_new,
                )
            return

        sr = self.config.target_sample_rate
        windows: List[
            Tuple[TranscriptSegment, List[Tuple[np.ndarray, float]]]
        ] = []
        for segment in snippets:
            start_idx = max(int(segment.start * sr), 0)
            end_idx = max(int(segment.end * sr), start_idx + 1)
            if start_idx >= audio.size:
                continue
            end_idx = min(end_idx, audio.size)
            segment_audio = audio[start_idx:end_idx]
            if segment_audio.size == 0:
                continue
            prepared = self._prepare_embedding_windows(segment_audio)
            if not prepared:
                continue
            windows.append((segment, prepared))

        if not windows:
            return

        embeddings: List[Tuple[TranscriptSegment, np.ndarray]] = []
        for segment, prepared_windows in windows:
            vectors: List[Tuple[np.ndarray, float]] = []
            for window, weight in prepared_windows:
                embedding = await self._embed_audio(window)
                if embedding is None or embedding.size == 0:
                    continue
                vectors.append((embedding, weight))

            if not vectors:
                logger.info(
                    "No embedding generated for segment %.2f-%.2f in %s",
                    segment.start,
                    segment.end,
                    state.conversation_id,
                )
                continue

            if len(vectors) > 1:
                embeddings_array = np.vstack([vec for vec, _ in vectors])
                weights = np.array([max(weight, 1e-6) for _, weight in vectors])
                weights /= weights.sum()
                averaged = (embeddings_array * weights[:, None]).sum(axis=0)
                logger.info(
                    "Averaged %d embedding windows for segment %.2f-%.2f in %s",
                    len(vectors),
                    segment.start,
                    segment.end,
                    state.conversation_id,
                )
            else:
                averaged = vectors[0][0]
            embeddings.append((segment, averaged))

        if not embeddings:
            logger.info(
                "pyannote produced no embeddings for conversation %s", state.conversation_id
            )
            return

        async with self._speaker_lock:
            for segment, vector in embeddings:
                speaker_id, is_new = self._match_speaker(vector, state.last_speaker_id)
                segment.speaker = speaker_id
                state.last_speaker_id = speaker_id
                publish_queue.append((speaker_id, segment.text, is_new))

        for speaker_id, utterance, is_new in publish_queue:
            await self._publish_person_detected(
                session_id=session_id,
                conversation_id=state.conversation_id,
                speaker_id=speaker_id,
                utterance=utterance,
                is_new=is_new,
            )

    def _prepare_embedding_windows(
        self, segment_audio: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        window_size = int(self.config.embedding_window_seconds * self.config.target_sample_rate)
        if window_size <= 0:
            return []
        if segment_audio.size <= window_size:
            rms = float(np.sqrt(np.mean(np.square(segment_audio))))
            return [(segment_audio, rms)]

        step = max(window_size // 6, 1)
        windows: List[Tuple[float, np.ndarray]] = []
        for start in range(0, segment_audio.size - window_size + 1, step):
            window = segment_audio[start : start + window_size]
            rms = float(np.sqrt(np.mean(np.square(window))))
            windows.append((rms, window))

        if not windows:
            tail = segment_audio[-window_size:]
            rms = float(np.sqrt(np.mean(np.square(tail))))
            return [(tail, rms)]

        windows.sort(key=lambda item: item[0], reverse=True)
        top_k = min(5, len(windows))
        return [(windows[i][1], windows[i][0]) for i in range(top_k)]

    async def _embed_audio(self, audio_window: np.ndarray) -> Optional[np.ndarray]:
        if self._pyannote_inference is None or audio_window.size == 0:
            return None
        waveform = torch.from_numpy(audio_window.astype(np.float32)).unsqueeze(0)

        def _infer() -> Optional[np.ndarray]:
            inference = self._pyannote_inference
            if inference is None:
                return None
            try:
                result = inference(
                    {"waveform": waveform, "sample_rate": self.config.target_sample_rate}
                )
            except TypeError:
                # Older pyannote versions expect numpy waveform without dict wrapper
                logger.debug("pyannote fallback to numpy waveform invocation")
                result = inference(
                    waveform.squeeze(0).numpy(), self.config.target_sample_rate
                )
            return result

        try:
            result = await asyncio.to_thread(_infer)
        except Exception as exc:  # noqa: BLE001
            logger.exception("pyannote embedding failed: %s", exc)
            return None

        if result is None:
            logger.info("pyannote inference returned None")
            return None
        if hasattr(result, "cpu"):
            vector = result.detach().cpu().numpy()
        elif isinstance(result, np.ndarray):
            vector = result
        else:
            vector = np.asarray(result)

        return vector.astype(np.float32, copy=False)

    def _match_speaker(
        self, vector: np.ndarray, previous_speaker: Optional[str]
    ) -> Tuple[str, bool]:
        normalized = self._normalize_vector(vector)
        if normalized.size == 0:
            fallback = previous_speaker or self._register_new_speaker(normalized)
            return fallback, fallback != previous_speaker

        scores: List[Tuple[SpeakerProfile, float]] = []
        for profile in self._speaker_profiles:
            score = float(np.dot(normalized, profile.embedding))
            scores.append((profile, score))

        if scores:
            similarity_map = ", ".join(
                f"{profile.speaker_id}:{score:.3f}" for profile, score in scores
            )
            logger.info("Speaker similarity map: %s", similarity_map)

        prev_profile = self._find_profile(previous_speaker)
        prev_score = float("-inf")
        if prev_profile is not None:
            prev_score = next(
                (score for profile, score in scores if profile is prev_profile),
                float(np.dot(normalized, prev_profile.embedding)),
            )
            logger.info(
                "Speaker similarity with previous %s: %.3f",
                prev_profile.speaker_id,
                prev_score,
            )

        best_profile: Optional[SpeakerProfile] = None
        best_score = float("-inf")
        for profile, score in scores:
            if profile is prev_profile:
                continue
            if score > best_score:
                best_score = score
                best_profile = profile

        if best_profile is not None:
            logger.info(
                "Best speaker candidate %s score=%.3f (threshold=%.3f)",
                best_profile.speaker_id,
                best_score,
                self.config.speaker_match_threshold,
            )

        candidate: Optional[Tuple[SpeakerProfile, float]] = None
        for profile, score in scores:
            if score >= self.config.speaker_match_threshold:
                if candidate is None or score > candidate[1]:
                    candidate = (profile, score)

        if candidate is not None:
            profile, score = candidate
            logger.info(
                "Selecting speaker %s with score %.3f", profile.speaker_id, score
            )
            self._update_profile(profile, normalized)
            return profile.speaker_id, False

        new_id = self._register_new_speaker(normalized)
        return new_id, True

    def _find_profile(self, speaker_id: Optional[str]) -> Optional[SpeakerProfile]:
        if speaker_id is None:
            return None
        for profile in self._speaker_profiles:
            if profile.speaker_id == speaker_id:
                return profile
        return None

    def _update_profile(self, profile: SpeakerProfile, vector: np.ndarray) -> None:
        weight = 1.0 / (profile.count + 1)
        updated = profile.embedding * (1.0 - weight) + vector * weight
        profile.embedding = self._normalize_vector(updated)
        profile.count += 1

    def _register_new_speaker(self, vector: np.ndarray) -> str:
        speaker_id = f"speaker_{self._next_speaker_index:03d}"
        self._next_speaker_index += 1
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            embedding=self._normalize_vector(vector),
            count=1,
        )
        self._speaker_profiles.append(profile)
        logger.info(
            "Registered new speaker profile %s (total=%d)",
            speaker_id,
            len(self._speaker_profiles),
        )
        return speaker_id

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        if vector.size == 0:
            return vector
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.zeros_like(vector)
        return (vector / norm).astype(np.float32)
