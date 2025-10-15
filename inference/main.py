"""Main inference service for AR glasses - handles two event types."""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from database import create_person, get_person_by_id, update_person_context
from fireworks_client import (
    aggregate_conversation_context,
    generate_ar_description,
    infer_new_person_details,
)
from models import ConversationEvent, InferenceResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_service")

app = FastAPI(title="Inference Service - AR Glasses")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Queue to hold processed results for streaming to clients
result_queue: asyncio.Queue[InferenceResult] = asyncio.Queue()

# Configuration
METADATA_SERVICE_URL = "http://localhost:8000/stream/conversation"


def safe_get_person(person_id: str) -> Optional[dict]:
    try:
        return get_person_by_id(person_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database lookup failed for %s: %s", person_id, exc)
        return None


def handle_person_detected(event: ConversationEvent) -> InferenceResult:
    """
    Handle PERSON_DETECTED event - return AR display information from MongoDB.
    """
    # Query MongoDB for person data
    person_doc = safe_get_person(event.person_id)

    latest_utterance = None
    if event.conversation:
        try:
            latest_utterance = event.conversation[-1].text
        except Exception:  # noqa: BLE001
            latest_utterance = None

    if person_doc:
        result = InferenceResult(
            person_id=event.person_id,
            name=person_doc["name"],
            relationship=person_doc["relationship"],
            description=person_doc["cached_description"]
        )
        logger.info(f"Person detected: {person_doc['name']} ({event.person_id})")
    else:
        person_label = event.person_id or "speaker_unknown"
        friendly_name = person_label.replace("_", " ").title()
        if person_label and person_label.startswith("speaker_"):
            suffix = person_label[8:]
            if suffix.isdigit():
                friendly_name = f"Speaker {suffix}"
        description = latest_utterance or "No previous interactions"
        # Person not found in database
        result = InferenceResult(
            person_id=person_label,
            name=friendly_name,
            relationship="Unidentified speaker",
            description=description,
        )
        logger.warning(f"Person not found in database: {event.person_id}")

    return result


async def handle_conversation_end(event: ConversationEvent) -> None:
    """
    Handle CONVERSATION_END event - store conversation for future reference.

    Two scenarios:
    1. Existing person: Update their context and description
    2. New person: Infer their details from conversation and create entry

    Uses Fireworks.ai models for both scenarios.
    """
    if not event.conversation:
        logger.warning(f"CONVERSATION_END event for {event.person_id} has no conversation data")
        return

    # Get current person data from MongoDB
    person_doc = safe_get_person(event.person_id)

    # Scenario 1: NEW PERSON - Infer details from conversation
    if not person_doc:
        logger.info(f"🆕 NEW PERSON DETECTED: {event.person_id}")
        logger.info(f"Analyzing first conversation ({len(event.conversation)} utterances) to infer details...")

        try:
            # Call Fireworks Model #3: Infer person details
            inferred_details = await infer_new_person_details(event.conversation)

            try:
                create_person(
                    person_id=event.person_id,
                    name=inferred_details["name"],
                    relationship=inferred_details["relationship"],
                    aggregated_context=inferred_details["aggregated_context"],
                    cached_description=inferred_details["cached_description"],
                )

                logger.info(
                    f"✓ Created new person: {inferred_details['name']} ({inferred_details['relationship']})"
                )
                logger.info(f"  Description: {inferred_details['cached_description']}")
            except Exception as create_exc:  # noqa: BLE001
                logger.warning(
                    "Could not persist new person %s: %s", event.person_id, create_exc
                )
            return

        except Exception as e:
            logger.error(f"Error inferring new person details: {e}")
            logger.error(f"Conversation will not be stored for {event.person_id}")
            return

    # Scenario 2: EXISTING PERSON - Update with new conversation
    logger.info(
        "Processing conversation end for %s (%s)",
        person_doc["name"],
        event.person_id,
    )
    logger.info(f"Conversation: {len(event.conversation)} utterances")

    try:
        # Call Fireworks Model #1: Aggregate conversation context
        updated_context = await aggregate_conversation_context(
            person_name=person_doc["name"],
            current_context=person_doc["aggregated_context"],
            new_conversation=event.conversation,
        )

        # Call Fireworks Model #2: Generate AR description
        new_description = await generate_ar_description(
            person_name=person_doc["name"],
            relationship=person_doc["relationship"],
            aggregated_context=updated_context,
        )

        try:
            updated = update_person_context(
                person_id=event.person_id,
                aggregated_context=updated_context,
                cached_description=new_description,
            )
        except Exception as update_exc:  # noqa: BLE001
            logger.warning(
                "Could not update person %s: %s", event.person_id, update_exc
            )
            return

        if updated:
            logger.info(
                f"✓ Successfully updated {person_doc['name']} with AI-generated content"
            )
            logger.info(f"  New context: {updated_context[:100]}...")
            logger.info(f"  New description: {new_description}")
        else:
            logger.error(f"Failed to update MongoDB for person {event.person_id}")

    except Exception as e:
        logger.error(f"Error processing conversation with Fireworks.ai: {e}")
        logger.error(f"Conversation will not be stored for {event.person_id}")


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
                                    await handle_conversation_end(event)
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
