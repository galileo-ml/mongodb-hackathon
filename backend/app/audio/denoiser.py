"""Noise suppression primitives for audio chunks."""

from __future__ import annotations

import logging

from ..core import AudioChunk, AudioSegment

logger = logging.getLogger("webrtc.audio.denoiser")


class AdaptiveDenoiser:
    """Stub adaptive denoiser that will host a real denoising model later."""

    def __init__(self, aggressiveness: float = 0.4) -> None:
        self.aggressiveness = aggressiveness

    async def denoise(self, chunk: AudioChunk) -> AudioSegment:
        """Return a voice-active segment placeholder.

        This currently acts as an identity function and simply wraps the raw audio
        chunk in an `AudioSegment`. Future revisions can plug a WebRTC VAD or deep
        denoiser here.
        """

        logger.debug(
            "Denoising chunk for session=%s ts=%s", chunk.session_id, chunk.timestamp
        )
        duration_ms = int(len(chunk.data) / 2 / chunk.sample_rate * 1000)
        return AudioSegment(
            session_id=chunk.session_id,
            sample_rate=chunk.sample_rate,
            start_ms=0,
            end_ms=duration_ms,
            energy=0.0,
            payload=chunk.data,
        )
