"""Mock metadata service that simulates two event types via SSE."""

import asyncio
import logging
import random
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from models import ConversationEvent, ConversationUtterance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_metadata")

app = FastAPI(title="Mock Metadata Service - Two Event Types")

# Mock conversation transcripts for each person - structured arrays
MOCK_CONVERSATIONS = {
    "person_004": [
        [
            ConversationUtterance(speaker="person_004", text="Hi Mom, it's Lisa! How are you doing today?"),
            ConversationUtterance(speaker="patient", text="I'm okay... who are you?"),
            ConversationUtterance(speaker="person_004", text="It's Lisa, your daughter. I just got promoted to head nurse at the hospital!"),
            ConversationUtterance(speaker="patient", text="That's wonderful news!"),
            ConversationUtterance(speaker="person_004", text="I'll be working the day shift now, so I can visit more often."),
            ConversationUtterance(speaker="patient", text="I'd like that very much."),
        ],
    ],
    "person_001": [
        [
            ConversationUtterance(speaker="person_001", text="Hi dad, how are you feeling today?"),
            ConversationUtterance(speaker="patient", text="I'm doing well, thanks for asking."),
            ConversationUtterance(speaker="person_001", text="I got that promotion at work I mentioned!"),
            ConversationUtterance(speaker="patient", text="That's wonderful news! Congratulations!"),
            ConversationUtterance(speaker="person_001", text="The kids are so excited to visit you next weekend."),
            ConversationUtterance(speaker="patient", text="I can't wait to see them."),
        ],
        [
            ConversationUtterance(speaker="person_001", text="How has your week been?"),
            ConversationUtterance(speaker="patient", text="Pretty good, I've been reading."),
            ConversationUtterance(speaker="person_001", text="Do you remember we talked about the garden last time?"),
            ConversationUtterance(speaker="patient", text="Yes, I need to water the plants."),
            ConversationUtterance(speaker="person_001", text="I'll bring your favorite cookies when I visit."),
            ConversationUtterance(speaker="patient", text="That would be lovely, thank you dear."),
        ],
    ],
    "person_002": [
        [
            ConversationUtterance(speaker="person_002", text="Hey dad, I brought some groceries for you."),
            ConversationUtterance(speaker="patient", text="Thank you son, you're always so thoughtful."),
            ConversationUtterance(speaker="person_002", text="My car is finally fixed after that issue."),
            ConversationUtterance(speaker="patient", text="That's good to hear."),
            ConversationUtterance(speaker="person_002", text="I'm planning a camping trip next month."),
            ConversationUtterance(speaker="patient", text="Sounds like fun, be safe out there."),
        ],
        [
            ConversationUtterance(speaker="person_002", text="Want to watch the game together this Sunday?"),
            ConversationUtterance(speaker="patient", text="Sure, that sounds nice."),
            ConversationUtterance(speaker="person_002", text="The weather has been great for walking."),
            ConversationUtterance(speaker="patient", text="Yes, I've been going out more often."),
            ConversationUtterance(speaker="person_002", text="How are you feeling today?"),
            ConversationUtterance(speaker="patient", text="Much better, thanks for checking."),
        ],
    ],
    "person_003": [
        [
            ConversationUtterance(speaker="person_003", text="Have you finished that mystery novel yet?"),
            ConversationUtterance(speaker="patient", text="Almost done, it's quite good."),
            ConversationUtterance(speaker="person_003", text="Book club is meeting next Tuesday."),
            ConversationUtterance(speaker="patient", text="I'll be there."),
            ConversationUtterance(speaker="person_003", text="Remember when we used to play chess in college?"),
            ConversationUtterance(speaker="patient", text="Those were good times."),
            ConversationUtterance(speaker="person_003", text="I found an old photo of us from graduation."),
            ConversationUtterance(speaker="patient", text="I'd love to see that."),
        ],
        [
            ConversationUtterance(speaker="person_003", text="The new library downtown is wonderful."),
            ConversationUtterance(speaker="patient", text="I haven't been yet."),
            ConversationUtterance(speaker="person_003", text="What have you been reading lately?"),
            ConversationUtterance(speaker="patient", text="Some classic mysteries."),
            ConversationUtterance(speaker="person_003", text="You always did love those."),
            ConversationUtterance(speaker="patient", text="They never get old."),
        ],
    ]
}


async def generate_conversation_events():
    """Generate mock conversation events - alternating between detection and conversation end."""
    event_count = 0

    while True:
        try:
            # Pick a person (including the new person_004)
            person_id = random.choice(["person_001", "person_002", "person_003", "person_004"])

            # First: Send PERSON_DETECTED event
            person_detected_event = ConversationEvent(
                event_type="PERSON_DETECTED",
                person_id=person_id,
                timestamp=datetime.utcnow()
            )

            event_count += 1
            logger.info(f"Event #{event_count}: PERSON_DETECTED - {person_id}")

            yield {
                "event": "conversation",
                "data": person_detected_event.model_dump_json(),
            }

            # Wait a bit (simulating conversation happening)
            await asyncio.sleep(random.uniform(8.0, 15.0))

            # Then: Send CONVERSATION_END event with structured conversation array
            conversation_array = random.choice(MOCK_CONVERSATIONS[person_id])

            conversation_end_event = ConversationEvent(
                event_type="CONVERSATION_END",
                person_id=person_id,
                conversation=conversation_array,
                timestamp=datetime.utcnow()
            )

            event_count += 1
            logger.info(f"Event #{event_count}: CONVERSATION_END - {person_id} ({len(conversation_array)} utterances)")

            yield {
                "event": "conversation",
                "data": conversation_end_event.model_dump_json(),
            }

            # Pause before next person detection
            await asyncio.sleep(random.uniform(5.0, 10.0))

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
    return {
        "status": "ok",
        "service": "mock_metadata_service"
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "mock_metadata_service",
        "version": "0.3.0",
        "focus": "two_event_types",
        "event_types": ["PERSON_DETECTED", "CONVERSATION_END"],
        "endpoints": {
            "conversation_stream": "/stream/conversation",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
