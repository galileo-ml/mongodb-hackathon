"""FastAPI application providing WebRTC ingress and audio pipeline stubs."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Set
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .audio import (
    AdaptiveDenoiser,
    AudioPipeline,
    PipelineConfig,
    PyannoteSpeakerEmbedder,
    SegmenterConfig,
    WebRTCVADSegmenter,
)
from .core import AudioChunk
from .services.vector_store import MongoDBVectorStore

logger = logging.getLogger("webrtc")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

app = FastAPI(title="Multimodal Ingress Prototype")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: Set[RTCPeerConnection] = set()
event_queues: dict[str, asyncio.Queue] = {}  # session_id -> queue for SSE
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "dummy_frontend" / "index.html"

# Load environment variables from project root .env (if present)
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("TORCHAUDIO_PYTHON_ONLY", "1")

vector_store = MongoDBVectorStore(
    uri="mongodb+srv://stub.example.com",
    database="diarization",
    collection="speaker_embeddings",
)

pipeline_config = PipelineConfig()
segmenter = WebRTCVADSegmenter(
    SegmenterConfig(
        mode=pipeline_config.vad_mode,
        min_speech_ms=pipeline_config.vad_min_speech_ms,
        min_silence_ms=pipeline_config.vad_min_silence_ms,
    )
)

audio_pipeline = AudioPipeline(
    denoiser=AdaptiveDenoiser(),
    segmenter=segmenter,
    embedder=PyannoteSpeakerEmbedder(),
    vector_store=vector_store,
    config=pipeline_config,
)

METRIC_INTERVAL_SECONDS = 30
_metrics_task: asyncio.Task | None = None
_STAT_COLOR = "\033[38;5;45m"
_VALUE_COLOR = "\033[38;5;214m"
_RESET_COLOR = "\033[0m"


async def log_vector_metrics_periodically() -> None:
    try:
        while True:
            await asyncio.sleep(METRIC_INTERVAL_SECONDS)
            lookups, unique_matches, total_embeddings = await vector_store.snapshot_metrics()
            logger.info(
                "%sVector store stats%s lookups=%s%d%s unique_speakers=%s%d%s total_embeddings=%s%d%s",
                _STAT_COLOR,
                _RESET_COLOR,
                _VALUE_COLOR,
                lookups,
                _RESET_COLOR,
                _VALUE_COLOR,
                unique_matches,
                _RESET_COLOR,
                _VALUE_COLOR,
                total_embeddings,
                _RESET_COLOR,
            )
    except asyncio.CancelledError:
        lookups, unique_matches, total_embeddings = await vector_store.snapshot_metrics()
        if lookups or unique_matches:
            logger.info(
                "%sVector store stats (final)%s lookups=%s%d%s unique_speakers=%s%d%s total_embeddings=%s%d%s",
                _STAT_COLOR,
                _RESET_COLOR,
                _VALUE_COLOR,
                lookups,
                _RESET_COLOR,
                _VALUE_COLOR,
                unique_matches,
                _RESET_COLOR,
                _VALUE_COLOR,
                total_embeddings,
                _RESET_COLOR,
            )
        raise

class SDPModel(BaseModel):
    sdp: str
    type: str


class PersonData(BaseModel):
    """Person information sent to frontend via SSE."""
    name: str
    description: str
    relationship: str
    person_id: str | None = None


async def broadcast_person(person: PersonData):
    """Broadcast person detection to all SSE connections."""
    dead_queues = []
    for session_id, queue in event_queues.items():
        try:
            await queue.put(person)
        except Exception as e:
            logger.error("Failed to send to SSE queue %s: %s", session_id, e)
            dead_queues.append(session_id)
    
    for session_id in dead_queues:
        event_queues.pop(session_id, None)


@app.on_event("startup")
async def on_startup() -> None:
    global _metrics_task
    await vector_store.connect()
    logger.info("Audio pipeline initialized with MongoDB vector store stub")
    try:
        await audio_pipeline.warm_whisper()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper warm-up failed: %s", exc)
    if _metrics_task is None or _metrics_task.done():
        _metrics_task = asyncio.create_task(log_vector_metrics_periodically())


@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=500, detail="Frontend bundle missing")
    return FileResponse(FRONTEND_INDEX)


@app.post("/offer", response_model=SDPModel)
async def offer(session: SDPModel) -> SDPModel:
    logger.info("Received SDP offer payload length=%d", len(session.sdp))
    pc = RTCPeerConnection()
    pcs.add(pc)
    session_id = f"sess-{uuid4().hex[:8]}"


    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio":
            logger.info("Session %s audio track ready", session_id)

        if track.kind == "audio":

            async def consume_audio() -> None:
                last_sample_rate: int | None = None
                while True:
                    try:
                        frame = await track.recv()
                        if not frame.planes:
                            continue
                        plane = frame.planes[0]
                        try:
                            data = plane.to_bytes()
                        except AttributeError:
                            data = bytes(plane)
                        sample_rate = getattr(frame, "sample_rate", None) or 16000
                        last_sample_rate = sample_rate
                        chunk = AudioChunk(
                            session_id=session_id,
                            data=data,
                            sample_rate=sample_rate,
                            timestamp=datetime.utcnow(),
                        )
                        try:
                            await audio_pipeline.process_chunk(chunk)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception(
                                "Audio pipeline error for session %s: %s",
                                session_id,
                                exc,
                            )
                            continue
                    except Exception:  # noqa: BLE001
                        break

                if last_sample_rate:
                    try:
                        await audio_pipeline.flush_session(session_id, last_sample_rate)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Error flushing audio buffer for session %s: %s",
                            session_id,
                            exc,
                        )

            asyncio.create_task(consume_audio())

        else:

            async def consume_video() -> None:
                while True:
                    try:
                        await track.recv()
                    except Exception:  # noqa: BLE001
                        break

            asyncio.create_task(consume_video())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            pcs.discard(pc)

    logger.info("Processing SDP offer for %s", session_id)
    offer_sdp = RTCSessionDescription(sdp=session.sdp, type=session.type)
    await pc.setRemoteDescription(offer_sdp)

    for transceiver in pc.getTransceivers():
        if transceiver.kind in {"audio", "video"}:
            transceiver.direction = "recvonly"

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    logger.info("Returning answer for PeerConnection %s", id(pc))
    return SDPModel(sdp=pc.localDescription.sdp, type=pc.localDescription.type)


@app.get("/events")
async def events_stream(session_id: str = "default"):
    """
    SSE endpoint for streaming person detection events to the frontend.
    
    When the audio pipeline identifies a speaker, person data will be
    pushed through this stream.
    
    Usage in frontend:
    ```js
    const eventSource = new EventSource('http://localhost:8000/events?session_id=abc123');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Person detected:', data);
    };
    ```
    """
    queue = asyncio.Queue()
    event_queues[session_id] = queue
    
    async def event_generator():
        try:
            logger.info("📡 SSE client connected: %s", session_id)
            
            # Stream person data from queue
            while True:
                person_data = await queue.get()
                yield f"data: {person_data.model_dump_json()}\n\n"
                
        except asyncio.CancelledError:
            logger.info("📡 SSE client disconnected: %s", session_id)
            event_queues.pop(session_id, None)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    global _metrics_task
    if _metrics_task is not None:
        _metrics_task.cancel()
        try:
            await _metrics_task
        except asyncio.CancelledError:  # noqa: PERF203 - expected
            pass
        _metrics_task = None
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    logger.info("Shutdown complete; closed all peer connections")
