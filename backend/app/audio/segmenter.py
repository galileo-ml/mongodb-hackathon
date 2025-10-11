"""Audio segmentation stubs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from ..core import AudioSegment

logger = logging.getLogger("webrtc.audio.segmenter")


@dataclass
class SegmenterConfig:
    window_ms: int = 1500
    hop_ms: int = 500
    energy_threshold: float = 0.1


class VoiceActivitySegmenter:
    """Stub segmenter that yields the input segment unchanged.

    Later this will integrate pyannote or WebRTC VAD to produce precise speaker
    turns. For now it simply emits the segment that was provided.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()

    async def segment(self, segment: AudioSegment) -> List[AudioSegment]:
        logger.debug(
            "Segmenting audio for session=%s window=%sms",
            segment.session_id,
            self.config.window_ms,
        )
        return [segment]
