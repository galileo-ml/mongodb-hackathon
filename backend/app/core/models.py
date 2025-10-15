"""Shared data models for audio processing and identity management."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

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
    sample_rate: int
    start_ms: int
    end_ms: int
    energy: float
    payload: bytes


class SpeakerEmbedding(BaseModel):
    """Speaker vector embedding suitable for similarity search."""

    session_id: str
    segment_id: str
    vector: list[float]
    model: str = "pyannote/embedding"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VectorSimilarityResult(BaseModel):
    """Result row returned from the vector store lookup."""

    matched_person_id: str | None
    score: float
    embedding: SpeakerEmbedding


class ConversationUtterance(BaseModel):
    """Single utterance in a conversation transcript."""

    speaker: str
    text: str


class ConversationEvent(BaseModel):
    """Event emitted by the audio pipeline for downstream consumers."""

    event_type: Literal["PERSON_DETECTED", "CONVERSATION_END"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    person_id: Optional[str] = None
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation: list[ConversationUtterance] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "event_type": "PERSON_DETECTED",
                    "person_id": "speaker_010",
                    "session_id": "sess-1234",
                    "timestamp": "2025-10-11T14:30:00Z",
                    "conversation": [
                        {
                            "speaker": "speaker_010",
                            "text": "Hey there, it's me again!",
                        }
                    ],
                },
                {
                    "event_type": "CONVERSATION_END",
                    "person_id": "speaker_010",
                    "conversation_id": "sess-1234-conv01",
                    "session_id": "sess-1234",
                    "timestamp": "2025-10-11T14:35:00Z",
                    "conversation": [
                        {
                            "speaker": "speaker_010",
                            "text": "Hi there, how are you today?",
                        },
                        {
                            "speaker": "speaker_patient",
                            "text": "I'm doing well, thank you!",
                        },
                    ],
                },
            ]
        }
