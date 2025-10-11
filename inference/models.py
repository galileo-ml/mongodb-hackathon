"""Data models for the inference service - Dementia care focus."""

from datetime import datetime
from pydantic import BaseModel, Field


class ConversationEvent(BaseModel):
    """Event from speaker diarization metadata service."""

    person_id: str = Field(..., description="Speaker/person identifier from diarization")
    text: str = Field(..., description="Transcribed conversation text")
    timestamp: datetime = Field(..., description="Event timestamp")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Diarization confidence score")
    event_id: str | None = Field(None, description="Optional unique event identifier")
    conversation_id: str | None = Field(None, description="Optional conversation session identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "person_id": "person_sarah",
                "text": "Hi dad, how are you feeling today?",
                "timestamp": "2025-10-11T14:30:00Z",
                "confidence": 0.95,
                "event_id": "evt_001",
                "conversation_id": "conv_abc123"
            }
        }


class InferenceResult(BaseModel):
    """Simple inference result for AR glasses display - shows who the person is and recent context."""

    person_id: str = Field(..., description="Person identifier for internal tracking")
    name: str = Field(..., description="Person's name to display")
    relationship: str = Field(..., description="Relationship to patient")
    description: str = Field(..., description="One-line context for AR display")

    class Config:
        json_schema_extra = {
            "example": {
                "person_id": "person_001",
                "name": "Sarah",
                "relationship": "Your daughter",
                "description": "Last spoke 3 days ago about her promotion and the grandchildren visiting"
            }
        }
