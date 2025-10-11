"""Main inference service for AR glasses - handles two event types."""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
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

# Queue to hold processed results for streaming to clients
result_queue: asyncio.Queue[InferenceResult] = asyncio.Queue()

# Configuration
METADATA_SERVICE_URL = "http://localhost:8001/stream/conversation"


def handle_person_detected(event: ConversationEvent) -> InferenceResult:
    """
    Handle PERSON_DETECTED event - return AR display information from MongoDB.
    """
    # Query MongoDB for person data
    person_doc = get_person_by_id(event.person_id)

    if person_doc:
        result = InferenceResult(
            person_id=event.person_id,
            name=person_doc["name"],
            relationship=person_doc["relationship"],
            description=person_doc["cached_description"]
        )
        logger.info(f"Person detected: {person_doc['name']} ({event.person_id})")
    else:
        # Person not found in database
        result = InferenceResult(
            person_id=event.person_id,
            name="Unknown Person",
            relationship="Unknown relationship",
            description="No previous interactions"
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
    person_doc = get_person_by_id(event.person_id)

    # Scenario 1: NEW PERSON - Infer details from conversation
    if not person_doc:
        logger.info(f"🆕 NEW PERSON DETECTED: {event.person_id}")
        logger.info(f"Analyzing first conversation ({len(event.conversation)} utterances) to infer details...")

        try:
            # Call Fireworks Model #3: Infer person details
            inferred_details = await infer_new_person_details(event.conversation)

            # Create new person in MongoDB
            new_person = create_person(
                person_id=event.person_id,
                name=inferred_details["name"],
                relationship=inferred_details["relationship"],
                aggregated_context=inferred_details["aggregated_context"],
                cached_description=inferred_details["cached_description"]
            )

            logger.info(f"✓ Created new person: {inferred_details['name']} ({inferred_details['relationship']})")
            logger.info(f"  Description: {inferred_details['cached_description']}")
            return

        except Exception as e:
            logger.error(f"Error inferring new person details: {e}")
            logger.error(f"Conversation will not be stored for {event.person_id}")
            return

    # Scenario 2: EXISTING PERSON - Update with new conversation
    logger.info(f"Processing conversation end for {person_doc['name']} ({event.person_id})")
    logger.info(f"Conversation: {len(event.conversation)} utterances")

    try:
        # Call Fireworks Model #1: Aggregate conversation context
        updated_context = await aggregate_conversation_context(
            person_name=person_doc["name"],
            current_context=person_doc["aggregated_context"],
            new_conversation=event.conversation
        )

        # Call Fireworks Model #2: Generate AR description
        new_description = await generate_ar_description(
            person_name=person_doc["name"],
            relationship=person_doc["relationship"],
            aggregated_context=updated_context
        )

        # Update MongoDB with AI-generated results
        updated = update_person_context(
            person_id=event.person_id,
            aggregated_context=updated_context,
            cached_description=new_description
        )

        if updated:
            logger.info(f"✓ Successfully updated {person_doc['name']} with AI-generated content")
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
