"""Audio pipeline that buffers conversations and transcribes them with Whisper."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ..core import AudioChunk
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

logger = logging.getLogger("webrtc.audio.pipeline")


@dataclass
class PipelineConfig:
    target_sample_rate: int = 16_000
    silence_timeout_seconds: float = 4.0
    min_conversation_seconds: float = 2.0
    vad_aggressiveness: int = 3
    transcription_model: str = "large-v3"
    min_speech_rms: float = 0.01
    noise_floor_smoothing: float = 0.9
    noise_gate_margin: float = 0.005


@dataclass
class ConversationState:
    conversation_id: str
    started_ts: float
    last_audio_ts: float
    last_speech_ts: float | None
    noise_floor_rms: float | None = None
    has_speech: bool = False
    chunks: List[np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.chunks is None:
            self.chunks = []


class AudioPipeline:
    """Buffers audio per WebRTC session and transcribes completed conversations."""

    def __init__(
        self,
        denoiser: AdaptiveDenoiser,
        config: PipelineConfig | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.config = config or PipelineConfig()
        self._conversations: Dict[str, ConversationState] = {}
        self._whisper_model = None
        self._whisper_lock = asyncio.Lock()
        self._vad = webrtcvad.Vad(self.config.vad_aggressiveness) if webrtcvad else None

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

    async def _transcribe_audio(self, audio: np.ndarray) -> List[tuple[str, str]]:
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
        snippets: List[tuple[str, str]] = []
        for seg in segments:
            start = float(seg.get("start", 0.0))
            minutes, seconds = divmod(start, 60.0)
            millis = int(round((seconds - int(seconds)) * 100))
            timestamp = f"{int(minutes):02d}:{int(seconds)%60:02d}.{millis:02d}"
            text = (seg.get("text") or "").strip()
            if text:
                snippets.append((timestamp, text))
        if not snippets and result.get("text"):
            snippets.append(("00:00.00", result["text"].strip()))
        return snippets

    def _print_transcript(self, conversation_id: str, snippets: List[tuple[str, str]]) -> None:
        logger.info(
            "Publishing conversation summary for %s (%d snippets)",
            conversation_id,
            len(snippets),
        )
        for ts, text in snippets:
            line = f"[{ts}] Speaker 1: {text}"
            print(f"\033[31m{line}\033[0m", flush=True)

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
