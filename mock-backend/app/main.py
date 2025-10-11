"""FastAPI mock backend with WebRTC ingress + SSE/WebSocket for AI events."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from pathlib import Path
from typing import Set
from datetime import datetime
from contextlib import asynccontextmanager

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc")
logger.setLevel(logging.INFO)

# Store active connections and event queues
pcs: Set[RTCPeerConnection] = set()
event_queues: dict[str, asyncio.Queue] = {}  # session_id -> queue for SSE
active_websockets: Set[WebSocket] = set()

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_INDEX = ROOT_DIR / "dummy_frontend" / "index.html"


class SDPModel(BaseModel):
    sdp: str
    type: str


class AIEvent(BaseModel):
    type: str  # "ai-response", "face-detected", "speaker-identified", etc.
    data: dict
    timestamp: str


# ==========================================
# Helper: Broadcast AI Events
# ==========================================

async def broadcast_event(event: AIEvent):
    """Broadcast event to all SSE and WebSocket connections."""
    event_json = event.model_dump_json()
    
    # Send to all SSE queues
    dead_queues = []
    for session_id, queue in event_queues.items():
        try:
            await queue.put(event)
        except Exception as e:
            logger.error(f"Failed to send to SSE queue {session_id}: {e}")
            dead_queues.append(session_id)
    
    # Cleanup dead queues
    for session_id in dead_queues:
        event_queues.pop(session_id, None)
    
    # Send to all WebSocket connections
    dead_ws = []
    for ws in active_websockets:
        try:
            await ws.send_text(event_json)
        except Exception as e:
            logger.error(f"Failed to send to WebSocket: {e}")
            dead_ws.append(ws)
    
    # Cleanup dead WebSockets
    for ws in dead_ws:
        active_websockets.discard(ws)


# ==========================================
# Lifespan & App Setup
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("🚀 Starting Mock FastAPI backend...")
    yield
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    event_queues.clear()
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="Mock Multimodal Ingress Backend",
    description="Mock backend with WebRTC, SSE, and WebSocket for testing",
    lifespan=lifespan
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Routes
# ==========================================

@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=500, detail="Frontend bundle missing")
    return FileResponse(FRONTEND_INDEX)


@app.post("/offer", response_model=SDPModel)
async def offer(session: SDPModel) -> SDPModel:
    """WebRTC offer/answer negotiation."""
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
                    
                    # TODO: Process frames here and generate AI events
                    # Example: Every 30 frames, send a mock AI response
                    if frame_count % 30 == 0:
                        mock_event = AIEvent(
                            type="ai-response",
                            data={
                                "llmResponse": f"Processed {frame_count} frames from {track.kind} track",
                                "frameCount": frame_count,
                                "trackKind": track.kind
                            },
                            timestamp=datetime.utcnow().isoformat()
                        )
                        await broadcast_event(mock_event)
                    
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


# ==========================================
# SSE Route (Server-Sent Events)
# ==========================================

@app.get("/events")
async def events_stream(session_id: str = "default"):
    """
    SSE endpoint for streaming AI events to the frontend.
    
    Usage in frontend:
    ```js
    const eventSource = new EventSource('http://localhost:8000/events?session_id=abc123');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('AI Event:', data);
    };
    ```
    """
    queue = asyncio.Queue()
    event_queues[session_id] = queue
    
    async def event_generator():
        try:
            logger.info(f"📡 SSE client connected: {session_id}")
            
            # Send initial connection event
            init_event = AIEvent(
                type="connection",
                data={"message": "Connected to event stream"},
                timestamp=datetime.utcnow().isoformat()
            )
            yield f"data: {init_event.model_dump_json()}\n\n"
            
            # Stream events from queue
            while True:
                event = await queue.get()
                yield f"data: {event.model_dump_json()}\n\n"
                
        except asyncio.CancelledError:
            logger.info(f"📡 SSE client disconnected: {session_id}")
            event_queues.pop(session_id, None)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ==========================================
# WebSocket Route
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for bidirectional communication.
    
    Usage in frontend:
    ```js
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('AI Event:', data);
    };
    ```
    """
    await websocket.accept()
    active_websockets.add(websocket)
    logger.info(f"🔌 WebSocket client connected (total: {len(active_websockets)})")
    
    try:
        # Send connection confirmation
        init_event = AIEvent(
            type="connection",
            data={"message": "Connected to WebSocket"},
            timestamp=datetime.utcnow().isoformat()
        )
        await websocket.send_text(init_event.model_dump_json())
        
        # Listen for incoming messages (optional)
        while True:
            data = await websocket.receive_text()
            logger.info(f"📩 Received from client: {data}")
            
            # Echo or process client messages if needed
            # For now, just acknowledge
            response = AIEvent(
                type="ack",
                data={"message": "Message received"},
                timestamp=datetime.utcnow().isoformat()
            )
            await websocket.send_text(response.model_dump_json())
            
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected")
    except Exception as e:
        logger.error(f"🔌 WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_peer_connections": len(pcs),
        "active_sse_clients": len(event_queues),
        "active_websockets": len(active_websockets),
    }

