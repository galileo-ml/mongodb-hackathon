"""Mock metadata service that simulates real-time speaker diarization events via SSE."""

import asyncio
import logging
import random
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from models import ConversationEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_metadata")

app = FastAPI(title="Mock Metadata Service")

# Mock conversation data
MOCK_CONVERSATIONS = [
    "Hello, how can I help you today?",
    "I'm looking for information about your product.",
    "That sounds great, can you tell me more?",
    "What are the pricing options available?",
    "I'd like to schedule a demo please.",
    "Thank you for your help!",
    "Could you clarify that for me?",
    "I have a question about the features.",
    "How long does the setup take?",
    "I'm interested in the enterprise plan.",
]

PERSON_IDS = ["person_001", "person_002", "person_003"]


async def generate_conversation_events():
    """Generate mock conversation events with realistic timing."""
    event_count = 0

    while True:
        try:
            # Simulate variable timing between utterances (2-5 seconds)
            await asyncio.sleep(random.uniform(2.0, 5.0))

            event = ConversationEvent(
                event_id=f"evt_{uuid4().hex[:8]}",
                person_id=random.choice(PERSON_IDS),
                text=random.choice(MOCK_CONVERSATIONS),
                timestamp=datetime.utcnow(),
                confidence=random.uniform(0.85, 0.99)
            )

            event_count += 1
            logger.info(f"Event #{event_count}: {event.person_id} - {event.text[:50]}...")

            yield {
                "event": "conversation",
                "data": event.model_dump_json(),
            }

        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            break
        except Exception as e:
            logger.error(f"Error generating event: {e}")
            break


@app.get("/stream/conversation")
async def stream_conversation():
    """SSE endpoint that streams mock conversation events."""
    logger.info("New client connected to conversation stream")
    return EventSourceResponse(generate_conversation_events())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "mock_metadata_service"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
