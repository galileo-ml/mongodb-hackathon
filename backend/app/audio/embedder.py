"""Pyannote-based speaker embedding extraction."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime


import numpy as np
import torch
from ..core import AudioSegment, SpeakerEmbedding

logger = logging.getLogger("webrtc.audio.embedder")


class PyannoteSpeakerEmbedder:
    """Generate speaker embeddings using pyannote.audio."""

    def __init__(
        self,
        model_name: str = "pyannote/embedding",
        auth_token: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.auth_token = auth_token or os.getenv("PYANNOTE_AUTH_TOKEN")
        self.device = device
        self._inference = None

    def _load_model(self):
        if self._inference is not None:
            return self._inference
        try:
            from pyannote.audio import Inference, Model
        except ImportError:  # pragma: no cover - runtime dependency
            logger.error(
                "pyannote.audio is not installed. Add it to dependencies and provide a Hugging Face token."
            )
            raise

        if not self.auth_token:
            logger.warning(
                "No PYANNOTE_AUTH_TOKEN provided. Hugging Face gated models may fail to download."
            )

        logger.info(
            "Loading pyannote embedding model '%s' (device=%s)",
            self.model_name,
            self.device or "auto",
        )

        model = Model.from_pretrained(
            self.model_name,
            use_auth_token=self.auth_token,
        )

        kwargs = {"model": model, "window": "whole"}
        if self.device:
            kwargs["device"] = self.device

        self._inference = Inference(**kwargs)
        return self._inference

    async def embed(self, segment: AudioSegment) -> SpeakerEmbedding:
        if not segment.payload:
            raise ValueError("Empty audio payload; cannot extract embedding")

        waveform = np.frombuffer(segment.payload, dtype=np.int16)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform /= torch.iinfo(torch.int16).max

        def _infer() -> np.ndarray:
            inference = self._load_model()
            result = inference(
                {"waveform": waveform, "sample_rate": segment.sample_rate}
            )
            return result

        embedding_vector = await asyncio.to_thread(_infer)
        if hasattr(embedding_vector, "cpu"):
            embedding_vector = embedding_vector.cpu().numpy()

        vector_list = embedding_vector.astype(float).tolist()
        segment_id = f"{segment.session_id}:{datetime.utcnow().timestamp():.3f}"
        return SpeakerEmbedding(
            session_id=segment.session_id,
            segment_id=segment_id,
            vector=vector_list,
            model=self.model_name,
        )
