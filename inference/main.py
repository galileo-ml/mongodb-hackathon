"""Main inference service that processes conversation data in real-time."""

import asyncio
import json
import logging
from datetime import datetime
from uuid import uuid4
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from models import ConversationEvent, InferenceResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_service")

app = FastAPI(title="Inference Service")

# Queue to hold processed results for streaming to clients
result_queue: asyncio.Queue[InferenceResult] = asyncio.Queue()

# Configuration
METADATA_SERVICE_URL = "http://localhost:8001/stream/conversation"


def hardcoded_inference_logic(event: ConversationEvent) -> InferenceResult:
    """
    Hardcoded inference logic for prototype.
    Replace this with actual ML models or business logic.
    """
    text_lower = event.text.lower()

    # Simple hardcoded sentiment analysis
    if any(word in text_lower for word in ["great", "thank", "interested", "please"]):
        sentiment = "positive"
    elif any(word in text_lower for word in ["question", "clarify", "help"]):
        sentiment = "neutral"
    else:
        sentiment = "positive"

    # Simple keyword extraction (words longer than 4 chars)
    keywords = [word.strip(",.!?") for word in event.text.split() if len(word) > 4]

    # Hardcoded analysis based on text patterns
    if "help" in text_lower or "question" in text_lower:
        analysis = "Customer inquiry detected - requires assistance"
    elif "pricing" in text_lower or "plan" in text_lower:
        analysis = "Sales opportunity - discussing pricing/plans"
    elif "demo" in text_lower or "schedule" in text_lower:
        analysis = "Demo request - high intent"
    elif "thank" in text_lower:
        analysis = "Positive closing - customer satisfied"
    else:
        analysis = "General conversation - information exchange"

    result = InferenceResult(
        result_id=f"res_{uuid4().hex[:8]}",
        event_id=event.event_id,
        person_id=event.person_id,
        original_text=event.text,
        analysis=analysis,
        sentiment=sentiment,
        keywords=keywords[:5],  # Limit to top 5
        timestamp=datetime.utcnow()
    )

    logger.info(f"Processed {event.event_id}: {sentiment} sentiment, {len(keywords)} keywords")
    return result


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

                                logger.info(f"Received event: {event.event_id} from {event.person_id}")

                                # Process with inference logic
                                result = hardcoded_inference_logic(event)

                                # Put result in queue for streaming to clients
                                await result_queue.put(result)

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
    logger.info("Starting inference service...")
    asyncio.create_task(consume_metadata_stream())


@app.get("/stream/inference")
async def stream_inference():
    """SSE endpoint that streams processed inference results."""
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
        "version": "0.1.0",
        "endpoints": {
            "inference_stream": "/stream/inference",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
