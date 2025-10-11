"""Voice activity detection using WebRTC VAD."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import webrtcvad

from ..core import AudioSegment

logger = logging.getLogger("webrtc.audio.segmenter")


@dataclass
class SegmenterConfig:
    mode: int = 3  # 0=quality, 3=aggressive
    frame_duration_ms: int = 30
    target_sample_rate: int = 16000
    min_speech_ms: int = 600
    min_silence_ms: int = 700


class WebRTCVADSegmenter:
    """Detects speech regions using WebRTC VAD."""

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        self.vad = webrtcvad.Vad(self.config.mode)
        self._resampler = None
        self._load_resampler()

    def _load_resampler(self) -> None:
        try:
            from torchaudio.functional import resample
        except ImportError:  # pragma: no cover - runtime dependency
            logger.warning(
                "torchaudio is required for resampling; speech detection may be limited"
            )
            self._resampler = None
        else:
            self._resampler = resample

    async def segment(self, segment: AudioSegment) -> List[Tuple[int, int]]:
        raw = np.frombuffer(segment.payload, dtype=np.int16)
        if raw.size == 0:
            return []

        sr = segment.sample_rate
        target_sr = self.config.target_sample_rate

        waveform = torch.from_numpy(raw.astype(np.float32)).unsqueeze(0)
        if sr != target_sr:
            if self._resampler is None:
                logger.debug(
                    "Skipping VAD for session=%s due to missing resampler",
                    segment.session_id,
                )
                return []
            waveform = self._resampler(waveform, sr, target_sr)
            sr = target_sr

        waveform = waveform.squeeze(0).clamp(-32768, 32767)
        pcm16 = waveform.numpy().astype(np.int16)

        frame_samples = int(sr * self.config.frame_duration_ms / 1000)
        if frame_samples == 0:
            return []

        bytes_pcm = pcm16.tobytes()
        total_frames = len(bytes_pcm) // (frame_samples * 2)
        speech_frames: list[Tuple[int, int]] = []

        current_start: int | None = None
        silence_count = 0

        for idx in range(total_frames):
            start = idx * frame_samples * 2
            end = start + frame_samples * 2
            frame_bytes = bytes_pcm[start:end]
            if len(frame_bytes) < frame_samples * 2:
                break
            is_speech = self.vad.is_speech(frame_bytes, sr)
            if is_speech:
                if current_start is None:
                    current_start = idx
                silence_count = 0
            elif current_start is not None:
                silence_count += 1
                if (
                    silence_count * self.config.frame_duration_ms
                    >= self.config.min_silence_ms
                ):
                    end_frame = idx - silence_count + 1
                    duration_ms = (
                        (end_frame - current_start) * self.config.frame_duration_ms
                    )
                    if duration_ms >= self.config.min_speech_ms:
                        speech_frames.append((current_start, end_frame))
                    current_start = None
                    silence_count = 0

        if current_start is not None:
            end_frame = total_frames
            duration_ms = (end_frame - current_start) * self.config.frame_duration_ms
            if duration_ms >= self.config.min_speech_ms:
                speech_frames.append((current_start, end_frame))

        if not speech_frames:
            logger.debug(
                "VAD detected no speech for session=%s window_ms=%d",
                segment.session_id,
                segment.end_ms - segment.start_ms,
            )
            return []

        sample_ranges: list[tuple[int, int]] = []
        orig_sr = segment.sample_rate
        ratio = orig_sr / sr

        total_samples = len(raw)
        for start_frame, end_frame in speech_frames:
            start_sample = int(start_frame * frame_samples * ratio)
            end_sample = int(end_frame * frame_samples * ratio)
            start_sample = max(start_sample, 0)
            end_sample = min(end_sample, total_samples)
            if end_sample <= start_sample:
                continue
            sample_ranges.append((start_sample, end_sample))

        logger.debug(
            "VAD detected %d segments for session=%s", len(sample_ranges), segment.session_id
        )
        return sample_ranges
