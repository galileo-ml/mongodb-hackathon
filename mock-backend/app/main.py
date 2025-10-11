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

from app.contracts import PersonData

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


# ==========================================
# Helper: Broadcast Person Data
# ==========================================

async def broadcast_person(person: PersonData):
    """Broadcast person detection to all SSE and WebSocket connections."""
    person_json = person.model_dump_json()

    # Send to all SSE queues
    dead_queues = []
    for session_id, queue in event_queues.items():
        try:
            await queue.put(person)
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
            await ws.send_text(person_json)
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
        last_notification = monotonic()
        notification_index = 0

        # Mock person detection notifications with structured data
        mock_people = [
            {
                "name": "Sarah",
                "description": "Last spoke 3 days ago about her promotion and the grandchildren visiting",
                "relationship": "Your daughter"
            },
            {
                "name": "Michael",
                "description": "Visited yesterday with groceries and talked about his camping trip",
                "relationship": "Your son"
            },
            {
                "name": "Robert",
                "description": "Last week you discussed the mystery novel and college memories",
                "relationship": "Your friend from book club"
            },
            {
                "name": "Emily",
                "description": "Called last month about the family reunion plans",
                "relationship": "Your niece"
            },
            {
                "name": "Dr. Patricia Chen",
                "description": "Your last appointment was two weeks ago for the checkup",
                "relationship": "Your doctor"
            },
        ]

        async def consume() -> None:
            nonlocal frame_count, last_logged, last_notification, notification_index
            while True:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    now = monotonic()
                    elapsed = now - last_logged
                    notification_elapsed = now - last_notification

                    # Send mock person notifications every 5 seconds (only for video track)
                    if track.kind == "video" and notification_elapsed >= 5:
                        person_dict = mock_people[notification_index % len(mock_people)]
                        person_data = PersonData(
                            name=person_dict["name"],
                            description=person_dict["description"],
                            relationship=person_dict["relationship"],
                            person_id=f"person_{(notification_index % len(mock_people)) + 1:03d}"
                        )
                        await broadcast_person(person_data)
                        logger.info("📢 Sent person notification: %s (%s)", person_data.name, person_data.relationship)
                        notification_index += 1
                        last_notification = now

                    # Log data reception every 5 seconds
                    if elapsed >= 5:
                        if track.kind == "video":
                            fps = frame_count / elapsed
                            logger.info(
                                "✅ Receiving video data from Peer %s: ~%.1f fps, %d frames in %.1fs (size=%sx%s)",
                                id(pc),
                                fps,
                                frame_count,
                                elapsed,
                                getattr(frame, "width", "?"),
                                getattr(frame, "height", "?"),
                            )
                        else:
                            logger.info(
                                "✅ Receiving audio data from Peer %s: %d chunks in %.1fs",
                                id(pc),
                                frame_count,
                                elapsed,
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

            # Stream person data from queue
            while True:
                person_data = await queue.get()
                yield f"data: {person_data.model_dump_json()}\n\n"

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
        # Listen for incoming messages (optional)
        while True:
            data = await websocket.receive_text()
            logger.info(f"📩 Received from client: {data}")

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

