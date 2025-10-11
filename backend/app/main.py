"""FastAPI application providing WebRTC ingress for webcam streams."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from pathlib import Path
from typing import Set

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc")
logger.setLevel(logging.INFO)

app = FastAPI(title="Multimodal Ingress Prototype")

pcs: Set[RTCPeerConnection] = set()
ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "dummy_frontend" / "index.html"


class SDPModel(BaseModel):
    sdp: str
    type: str


@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=500, detail="Frontend bundle missing")
    return FileResponse(FRONTEND_INDEX)


@app.post("/offer", response_model=SDPModel)
async def offer(session: SDPModel) -> SDPModel:
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info("Created PeerConnection %s", id(pc))

    @pc.on("track")
    def on_track(track) -> None:
        logger.info("Track %s received on %s", track.kind, id(pc))

        frame_count = 0
        last_logged = monotonic()

        async def consume() -> None:
            nonlocal frame_count, last_logged
            while True:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    now = monotonic()
                    elapsed = now - last_logged
                    if elapsed >= 1:
                        if track.kind == "video":
                            fps = frame_count / elapsed
                            logger.info(
                                "Peer %s video: ~%.1f fps, frame pts=%s, size=%sx%s",
                                id(pc),
                                fps,
                                getattr(frame, "pts", "?"),
                                getattr(frame, "width", "?"),
                                getattr(frame, "height", "?"),
                            )
                        else:
                            logger.info(
                                "Peer %s audio: %d chunks received in %.2fs (last pts=%s)",
                                id(pc),
                                frame_count,
                                elapsed,
                                getattr(frame, "pts", "?"),
                            )
                        frame_count = 0
                        last_logged = now
                except Exception as exc:  # noqa: BLE001
                    logger.info("Track %s on %s ended: %s", track.kind, id(pc), exc)
                    break

        asyncio.create_task(consume())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("PeerConnection %s state %s", id(pc), pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            await pc.close()
            pcs.discard(pc)

    offer_sdp = RTCSessionDescription(sdp=session.sdp, type=session.type)
    await pc.setRemoteDescription(offer_sdp)

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
