"""FastAPI application providing WebRTC ingress and audio pipeline stubs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Set
from uuid import uuid4

from aiortc import RTCPeerConnection, RTCSessionDescription
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .audio import (
    AdaptiveDenoiser,
    AudioPipeline,
    PipelineConfig,
    PyannoteSpeakerEmbedder,
    VoiceActivitySegmenter,
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

pcs: Set[RTCPeerConnection] = set()
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "dummy_frontend" / "index.html"

# Load environment variables from project root .env (if present)
load_dotenv(ROOT_DIR / ".env")

vector_store = MongoDBVectorStore(
    uri="mongodb+srv://stub.example.com",
    database="diarization",
    collection="speaker_embeddings",
)

audio_pipeline = AudioPipeline(
    denoiser=AdaptiveDenoiser(),
    segmenter=VoiceActivitySegmenter(),
    embedder=PyannoteSpeakerEmbedder(),
    vector_store=vector_store,
    config=PipelineConfig(),
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


@app.on_event("startup")
async def on_startup() -> None:
    global _metrics_task
    await vector_store.connect()
    logger.info("Audio pipeline initialized with MongoDB vector store stub")
    if _metrics_task is None or _metrics_task.done():
        _metrics_task = asyncio.create_task(log_vector_metrics_periodically())


@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=500, detail="Frontend bundle missing")
    return FileResponse(FRONTEND_INDEX)


@app.post("/offer", response_model=SDPModel)
async def offer(session: SDPModel) -> SDPModel:
    pc = RTCPeerConnection()
    pcs.add(pc)
    session_id = f"sess-{uuid4().hex[:8]}"
    logger.info("Created PeerConnection %s (%s)", id(pc), session_id)

    @pc.on("track")
    def on_track(track) -> None:
        logger.info("Track %s received on %s (%s)", track.kind, id(pc), session_id)

        if track.kind == "audio":

            async def consume_audio() -> None:
                frame_count = 0
                last_logged = monotonic()
                last_sample_rate: int | None = None
                while True:
                    try:
                        frame = await track.recv()
                        frame_count += 1
                        if not frame.planes:
                            logger.warning(
                                "Audio frame without planes on %s", session_id
                            )
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
                            matches = await audio_pipeline.process_chunk(chunk)
                        except Exception as exc:  # noqa: BLE001
                            logger.exception(
                                "Audio pipeline error for session %s: %s",
                                session_id,
                                exc,
                            )
                            continue
                        if matches:
                            best = matches[0]
                            logger.info(
                                "Session %s best similarity score=%.3f segment=%s",
                                session_id,
                                best.score,
                                best.embedding.segment_id,
                            )
                        now = monotonic()
                        elapsed = now - last_logged
                        if elapsed >= 1:
                            logger.info(
                                "Peer %s audio: %d frames in %.2fs (last pts=%s)",
                                id(pc),
                                frame_count,
                                elapsed,
                                getattr(frame, "pts", "?"),
                            )
                            frame_count = 0
                            last_logged = now
                    except Exception as exc:  # noqa: BLE001
                        logger.info("Audio track on %s ended: %s", id(pc), exc)
                        break

                if last_sample_rate:
                    try:
                        tail_matches = await audio_pipeline.flush_session(
                            session_id, last_sample_rate
                        )
                        if tail_matches:
                            best = tail_matches[0]
                            logger.info(
                                "Session %s tail similarity score=%.3f segment=%s",
                                session_id,
                                best.score,
                                best.embedding.segment_id,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "Error flushing audio buffer for session %s: %s",
                            session_id,
                            exc,
                        )

            asyncio.create_task(consume_audio())

        else:

            async def consume_video() -> None:
                frame_count = 0
                last_logged = monotonic()
                while True:
                    try:
                        frame = await track.recv()
                        frame_count += 1
                        now = monotonic()
                        elapsed = now - last_logged
                        if elapsed >= 1:
                            fps = frame_count / elapsed
                            logger.info(
                                "Peer %s video: ~%.1f fps, frame pts=%s, size=%sx%s",
                                id(pc),
                                fps,
                                getattr(frame, "pts", "?"),
                                getattr(frame, "width", "?"),
                                getattr(frame, "height", "?"),
                            )
                            frame_count = 0
                            last_logged = now
                    except Exception as exc:  # noqa: BLE001
                        logger.info("Video track on %s ended: %s", id(pc), exc)
                        break

            asyncio.create_task(consume_video())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("PeerConnection %s state %s", id(pc), pc.connectionState)
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
