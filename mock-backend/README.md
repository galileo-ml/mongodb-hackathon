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

# Run the server
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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
curl http://localhost:8000/health
```

### Test SSE endpoint
```bash
curl -N http://localhost:8000/events?session_id=test123
```

### Full integration test
1. Run this mock backend: `uv run uvicorn app.main:app --reload --port 8000`
2. Run the Next.js frontend: `cd ../frontend && npm run dev`
3. Open browser to `http://localhost:3000`
4. Allow camera/microphone access
5. Watch for "Connected (WebRTC)" status
6. Mock AI responses will appear every ~1 second

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

