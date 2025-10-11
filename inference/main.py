"""Main inference service for AR glasses - handles two event types."""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from models import ConversationEvent, InferenceResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_service")

app = FastAPI(title="Inference Service - AR Glasses")

# Queue to hold processed results for streaming to clients
result_queue: asyncio.Queue[InferenceResult] = asyncio.Queue()

# Mock "database" of people (mutable so we can update last interactions)
MOCK_PERSON_DATA = {
    "person_001": {
        "name": "Sarah",
        "relationship": "Your daughter",
        "last_interaction": "Last spoke 3 days ago about her promotion and the grandchildren visiting"
    },
    "person_002": {
        "name": "Michael",
        "relationship": "Your son",
        "last_interaction": "Visited yesterday with groceries and talked about his camping trip"
    },
    "person_003": {
        "name": "Robert",
        "relationship": "Your friend from book club",
        "last_interaction": "Last week you discussed the mystery novel and college memories"
    }
}

# Configuration
METADATA_SERVICE_URL = "http://localhost:8001/stream/conversation"


def handle_person_detected(event: ConversationEvent) -> InferenceResult:
    """
    Handle PERSON_DETECTED event - return AR display information.
    """
    # Get person data (name, relationship, last interaction)
    person_data = MOCK_PERSON_DATA.get(event.person_id, {
        "name": "Unknown Person",
        "relationship": "Unknown relationship",
        "last_interaction": "No previous interactions"
    })

    result = InferenceResult(
        person_id=event.person_id,
        name=person_data["name"],
        relationship=person_data["relationship"],
        description=person_data["last_interaction"]
    )

    logger.info(f"Person detected: {person_data['name']} ({event.person_id})")
    return result


def handle_conversation_end(event: ConversationEvent) -> None:
    """
    Handle CONVERSATION_END event - store conversation for future reference.
    Updates the person's last_interaction field.
    """
    if not event.conversation:
        logger.warning(f"CONVERSATION_END event for {event.person_id} has no conversation data")
        return

    # Update person's last interaction in mock database
    if event.person_id in MOCK_PERSON_DATA:
        # In production, this would use LLM to generate a summary
        # For now, create simple summary from conversation structure
        num_utterances = len(event.conversation)

        # Extract some topic keywords from conversation
        all_text = " ".join([u.text.lower() for u in event.conversation])
        topics = []
        if any(word in all_text for word in ["promotion", "job", "work"]):
            topics.append("work")
        if any(word in all_text for word in ["kids", "children", "grandchildren"]):
            topics.append("family")
        if any(word in all_text for word in ["visit", "coming", "see you"]):
            topics.append("upcoming visit")

        if topics:
            topic_str = " and ".join(topics)
            summary = f"Just talked about {topic_str}"
        else:
            summary = f"Just had a conversation ({num_utterances} messages)"

        MOCK_PERSON_DATA[event.person_id]["last_interaction"] = summary

        logger.info(f"Stored conversation for {MOCK_PERSON_DATA[event.person_id]['name']} ({event.person_id})")
        logger.info(f"Conversation: {num_utterances} utterances")
    else:
        logger.warning(f"Unknown person {event.person_id} - conversation not stored")


async def consume_metadata_stream():
    """Background task to consume SSE from metadata service and process events."""
    logger.info(f"Starting metadata stream consumer from {METADATA_SERVICE_URL}")

    retry_delay = 5
    max_retry_delay = 60

    while True:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", METADATA_SERVICE_URL) as response:
                    logger.info("Connected to metadata stream")
                    retry_delay = 5  # Reset retry delay on successful connection

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            try:
                                event_data = json.loads(data)
                                event = ConversationEvent(**event_data)

                                # Auto-generate timestamp if not provided
                                if not event.timestamp:
                                    event.timestamp = datetime.utcnow()

                                logger.info(f"Received {event.event_type} event for {event.person_id}")

                                # Route to appropriate handler based on event type
                                if event.event_type == "PERSON_DETECTED":
                                    result = handle_person_detected(event)
                                    # Put result in queue for streaming to AR glasses
                                    await result_queue.put(result)

                                elif event.event_type == "CONVERSATION_END":
                                    handle_conversation_end(event)
                                    # No result to stream - just storage

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse event data: {e}")
                            except Exception as e:
                                logger.error(f"Error processing event: {e}")

        except httpx.ConnectError:
            logger.error(f"Cannot connect to metadata service. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error in metadata consumer: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)


async def generate_inference_results() -> AsyncGenerator[dict, None]:
    """Generate SSE events from processed inference results."""
    logger.info("New client connected to inference stream")

    try:
        while True:
            # Wait for next result with timeout to send keepalive
            try:
                result = await asyncio.wait_for(result_queue.get(), timeout=30.0)
                yield {
                    "event": "inference",
                    "data": result.model_dump_json(),
                }
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield {
                    "comment": "keepalive"
                }
    except asyncio.CancelledError:
        logger.info("Client disconnected from inference stream")
        raise


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    logger.info("Starting inference service for AR glasses...")
    asyncio.create_task(consume_metadata_stream())


@app.get("/stream/inference")
async def stream_inference():
    """SSE endpoint that streams processed inference results (PERSON_DETECTED only)."""
    return EventSourceResponse(generate_inference_results())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "inference_service",
        "queue_size": result_queue.qsize()
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "inference_service",
        "version": "0.3.0",
        "focus": "ar_glasses_two_event_types",
        "endpoints": {
            "inference_stream": "/stream/inference",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
