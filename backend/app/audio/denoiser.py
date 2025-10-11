"""Noise suppression primitives for audio chunks."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ..core import AudioChunk, AudioSegment

logger = logging.getLogger("webrtc.audio.denoiser")

try:  # pragma: no cover - optional dependency
    from rnnoise import RNNoise  # type: ignore
except ImportError:  # pragma: no cover
    RNNoise = None

try:  # pragma: no cover - optional dependency
    from torchaudio.functional import resample
except ImportError:  # pragma: no cover
    resample = None


class AdaptiveDenoiser:
    """Applies RNNoise when available, falling back to identity."""

    def __init__(self, aggressiveness: float = 0.4) -> None:
        self.aggressiveness = aggressiveness
        self._rnnoise: Optional[RNNoise] = None
        if RNNoise is not None:
            try:
                self._rnnoise = RNNoise()
                logger.info("RNNoise initialized for audio denoising")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize RNNoise: %s", exc)
                self._rnnoise = None
        else:
            logger.info("RNNoise not installed; denoiser will pass through audio")

    async def denoise(self, chunk: AudioChunk) -> AudioSegment:
        logger.debug(
            "Denoising chunk for session=%s ts=%s", chunk.session_id, chunk.timestamp
        )
        pcm = np.frombuffer(chunk.data, dtype=np.int16)
        if pcm.size == 0:
            payload = chunk.data
        else:
            payload = await self._apply_rnnoise(pcm, chunk.sample_rate)

        duration_ms = int(len(payload) / 2 / chunk.sample_rate * 1000)
        return AudioSegment(
            session_id=chunk.session_id,
            sample_rate=chunk.sample_rate,
            start_ms=0,
            end_ms=duration_ms,
            energy=0.0,
            payload=payload,
        )

    async def _apply_rnnoise(self, samples: np.ndarray, sample_rate: int) -> bytes:
        if self._rnnoise is None or samples.size < 480:
            return samples.tobytes()

        target_sr = 48000
        tensor = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0) / 32768.0
        if sample_rate != target_sr:
            if resample is None:
                logger.debug(
                    "torchaudio missing; skipping RNNoise resample for sr=%d",
                    sample_rate,
                )
                return samples.tobytes()
            tensor = resample(tensor, sample_rate, target_sr)

        pcm48 = np.clip(
            (tensor.squeeze(0).numpy() * 32768.0), -32768, 32767
        ).astype(np.int16)
        frame_size = 480
        if pcm48.size % frame_size != 0:
            pad = frame_size - (pcm48.size % frame_size)
            pcm48 = np.pad(pcm48, (0, pad), mode="constant")

        denoised_frames = []
        for i in range(0, pcm48.size, frame_size):
            frame = pcm48[i : i + frame_size]
            try:
                den = self._rnnoise.process_frame(frame.tobytes())
                den_frame = np.frombuffer(den, dtype=np.int16)
            except AttributeError:
                den_frame = self._rnnoise.filter(frame)  # type: ignore[attr-defined]
                den_frame = np.asarray(den_frame, dtype=np.int16)
            except Exception as exc:  # noqa: BLE001
                logger.debug("RNNoise frame processing failed: %s", exc)
                den_frame = frame
            denoised_frames.append(den_frame)

        denoised = np.concatenate(denoised_frames) if denoised_frames else pcm48

        if sample_rate != target_sr and resample is not None:
            tensor = (
                torch.from_numpy(denoised.astype(np.float32)).unsqueeze(0) / 32768.0
            )
            tensor = resample(tensor, target_sr, sample_rate)
            denoised = np.clip(
                tensor.squeeze(0).numpy() * 32768.0, -32768, 32767
            ).astype(np.int16)

        denoised = denoised[: samples.size]
        return denoised.tobytes()
