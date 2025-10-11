"""FastAPI application providing WebRTC ingress and audio pipeline stubs."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Set
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .audio import AdaptiveDenoiser, AudioPipeline, PipelineConfig
from .core import AudioChunk
from .services.conversation_stream import ConversationEventBus

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pcs: Set[RTCPeerConnection] = set()
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "dummy_frontend" / "index.html"

# Load environment variables from project root .env (if present)
load_dotenv(ROOT_DIR / ".env")
os.environ.setdefault("TORCHAUDIO_PYTHON_ONLY", "1")

pipeline_config = PipelineConfig()
conversation_bus = ConversationEventBus()
audio_pipeline = AudioPipeline(
    denoiser=AdaptiveDenoiser(),
    config=pipeline_config,
    conversation_bus=conversation_bus,
)

class SDPModel(BaseModel):
    sdp: str
    type: str


@app.on_event("startup")
async def on_startup() -> None:
    try:
        await audio_pipeline.warm_whisper()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper warm-up failed: %s", exc)


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


@app.on_event("shutdown")
async def on_shutdown() -> None:
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    logger.info("Shutdown complete; closed all peer connections")


@app.get("/stream/conversation")
async def stream_conversation() -> StreamingResponse:
    """Server-Sent Events stream of completed conversations."""

    async def event_generator():
        queue = await conversation_bus.subscribe()
        try:
            while True:
                try:
                    event = await queue.get()
                except asyncio.CancelledError:
                    break
                payload = event.model_dump_json()
                yield f"event: conversation\ndata: {payload}\n\n"
        finally:
            await conversation_bus.unsubscribe(queue)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
