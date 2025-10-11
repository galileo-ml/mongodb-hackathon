"""Shared data models for audio processing and identity management."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AudioChunk(BaseModel):
    """Raw audio payload emitted by the capture stack."""

    session_id: str
    data: bytes = Field(description="PCM16 audio payload")
    sample_rate: int = Field(default=16000, description="Samples per second")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AudioSegment(BaseModel):
    """Represents a denoised, voice-active slice of audio."""

    session_id: str
    start_ms: int
    end_ms: int
    energy: float
    payload: bytes


class SpeakerEmbedding(BaseModel):
    """Speaker vector embedding suitable for similarity search."""

    session_id: str
    segment_id: str
    vector: list[float]
    model: Literal["resemblyzer", "xvector", "custom"] = "resemblyzer"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VectorSimilarityResult(BaseModel):
    """Result row returned from the vector store lookup."""

    matched_person_id: str | None
    score: float
    embedding: SpeakerEmbedding
