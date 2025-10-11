# Mock Backend

This is a mock FastAPI backend with WebRTC, SSE (Server-Sent Events), and WebSocket support for testing the frontend.

## Features

- **WebRTC** - Accepts video/audio streams via `/offer` endpoint
- **SSE** - Streams AI events to frontend via `/events` endpoint
- **WebSocket** - Bidirectional communication via `/ws` endpoint
- **Mock AI Responses** - Sends mock responses every 30 frames

## Setup

```bash
# Install dependencies
uv sync

# Run the server (on port 8001 to avoid conflict with real backend)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

## Endpoints

### WebRTC
- `POST /offer` - WebRTC offer/answer negotiation

### SSE (Recommended for AI responses)
- `GET /events?session_id=<id>` - Server-Sent Events stream

### WebSocket (Alternative)
- `WebSocket /ws` - Bidirectional WebSocket connection

### Other
- `GET /` - Serves dummy frontend
- `GET /health` - Health check

## Testing

### Test health endpoint
```bash
curl http://localhost:8001/health
```

### Test SSE endpoint
```bash
curl -N http://localhost:8001/events?session_id=test123
```

### Full integration test
1. Run this mock backend: `uv run uvicorn app.main:app --reload --port 8001`
2. Run the Next.js frontend: `cd ../frontend && npm run dev`
3. Update frontend to point to `http://localhost:8001` (or use real backend on 8000)
4. Open browser to `http://localhost:3000`
5. Allow camera/microphone access
6. Watch for "Connected (WebRTC)" status
7. Mock AI responses will appear every ~5 seconds

## Mock AI Response Logic

Currently sends a mock response every 30 video frames (line ~159 in `main.py`):

```python
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
```

## Architecture

```
Frontend (Next.js)
    │
    ├─── WebRTC ────────> Mock Backend (port 8000)
    │    (video+audio)    POST /offer
    │                     ↓
    │                  Process frames
    │                     ↓
    └─── SSE <────────  Mock AI events
         (responses)    GET /events
```

## Note

This is a **mock backend** for testing. The real backend should be in the `backend/` directory and implement actual AI processing, face detection, speaker diarization, etc.

