"""Data models for the inference service."""

from datetime import datetime
from pydantic import BaseModel, Field


class ConversationEvent(BaseModel):
    """Event from speaker diarization metadata service."""

    event_id: str = Field(..., description="Unique event identifier")
    person_id: str = Field(..., description="Speaker/person identifier from diarization")
    text: str = Field(..., description="Transcribed conversation text")
    timestamp: datetime = Field(..., description="Event timestamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Diarization confidence score")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_001",
                "person_id": "person_123",
                "text": "Hello, how can I help you today?",
                "timestamp": "2025-10-11T14:30:00Z",
                "confidence": 0.95
            }
        }


class InferenceResult(BaseModel):
    """Processed inference result to be streamed to downstream services."""

    result_id: str = Field(..., description="Unique result identifier")
    event_id: str = Field(..., description="Source event ID this result is based on")
    person_id: str = Field(..., description="Speaker/person identifier")
    original_text: str = Field(..., description="Original conversation text")
    analysis: str = Field(..., description="Analysis result (hardcoded for prototype)")
    sentiment: str = Field(..., description="Detected sentiment")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    timestamp: datetime = Field(..., description="Processing timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "result_id": "res_001",
                "event_id": "evt_001",
                "person_id": "person_123",
                "original_text": "Hello, how can I help you today?",
                "analysis": "Greeting detected - customer service interaction",
                "sentiment": "positive",
                "keywords": ["help", "today"],
                "timestamp": "2025-10-11T14:30:01Z"
            }
        }
