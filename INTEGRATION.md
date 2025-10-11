# Backend Integration Guide

## Overview
The real backend now has SSE support to stream person detection events to the frontend, matching the mock-backend API.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Frontend (port 3000)                   │
│  ┌────────────────┐         ┌──────────────────────────┐    │
│  │  WebcamStream  │────────▶│  Face Detection (local)  │    │
│  └────────────────┘         └──────────────────────────┘    │
└────────┬─────────────────────────────────────┬───────────────┘
         │ WebRTC (video+audio)                │ SSE (EventSource)
         │                                      │
         ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Real Backend (port 8000)                        │
│  ┌─────────────┐   ┌────────────────┐   ┌────────────────┐│
│  │ /offer      │──▶│ Audio Pipeline │──▶│ /events (SSE)  ││
│  │ (WebRTC)    │   │ - Denoise      │   │ Person Data    ││
│  │             │   │ - VAD Segment  │   │ Broadcast      ││
│  │             │   │ - Speaker ID   │   │                ││
│  │             │   │ - Whisper      │   │                ││
│  └─────────────┘   └────────────────┘   └────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Changes Made

### 1. Real Backend (`backend/app/main.py`)

#### Added Imports
```python
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
```

#### Added CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Added Event Queue Storage
```python
event_queues: dict[str, asyncio.Queue] = {}  # session_id -> queue for SSE
```

#### Added PersonData Model
```python
class PersonData(BaseModel):
    """Person information sent to frontend via SSE."""
    name: str
    description: str
    relationship: str
    person_id: str | None = None
```

#### Added Broadcast Helper
```python
async def broadcast_person(person: PersonData):
    """Broadcast person detection to all SSE connections."""
    # Sends person data to all connected SSE clients
```

#### Added SSE Endpoint
```python
@app.get("/events")
async def events_stream(session_id: str = "default"):
    """SSE endpoint for streaming person detection events."""
    # Returns StreamingResponse with text/event-stream
```

### 2. Mock Backend Port Change

- **Old**: Port 8000
- **New**: Port 8001 (to avoid conflicts)
- **Updated**: README.md, QUICKSTART.md

### 3. Frontend (No Changes Needed!)

Frontend already points to `localhost:8000` and uses:
- `/offer` for WebRTC
- `/events` for SSE

## Running the Stack

### Option 1: Real Backend (Recommended)

```bash
# Terminal 1: Real Backend
cd backend
source venv/bin/activate
export PYANNOTE_AUTH_TOKEN=your_token
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

**What happens:**
1. Frontend sends audio/video via WebRTC
2. Backend processes audio through diarization pipeline
3. When speaker identified, `broadcast_person()` sends data via SSE
4. Frontend shows notification next to detected face

### Option 2: Mock Backend (Testing)

```bash
# Terminal 1: Mock Backend
cd mock-backend
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Terminal 2: Frontend (update BACKEND_URL to port 8001)
cd frontend
npm run dev
```

**What happens:**
1. Frontend sends audio/video via WebRTC
2. Mock backend sends hardcoded person data every 5 seconds
3. Frontend shows notification next to detected face

## API Endpoints

### Real Backend (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve dummy frontend |
| `/offer` | POST | WebRTC offer/answer negotiation |
| `/events` | GET | SSE stream for person detection events |

### Mock Backend (port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve dummy frontend |
| `/offer` | POST | WebRTC offer/answer negotiation |
| `/events` | GET | SSE stream for mock person data |
| `/ws` | WebSocket | Alternative to SSE (not used) |
| `/health` | GET | Health check |

## SSE Event Format

Both backends send the same JSON structure:

```json
{
  "name": "Sarah",
  "description": "Last spoke 3 days ago about her promotion and the grandchildren visiting",
  "relationship": "Your daughter",
  "person_id": "person_001"
}
```

## Integration Points

### When to Broadcast Person Data

In the real backend, call `broadcast_person()` when:

1. **Speaker diarization identifies someone**
2. **Vector similarity match found**
3. **Whisper transcription completes**

Example integration in `consume_audio()`:

```python
# After audio pipeline processes chunk
result = await audio_pipeline.process_chunk(chunk)

# If speaker identified, broadcast to SSE clients
if result and hasattr(result, 'person_id'):
    person = PersonData(
        name=result.name,
        description=result.description,
        relationship=result.relationship,
        person_id=result.person_id
    )
    await broadcast_person(person)
```

## Testing

### Test SSE Connection
```bash
# Real backend
curl -N http://localhost:8000/events?session_id=test

# Mock backend
curl -N http://localhost:8001/events?session_id=test
```

### Test WebRTC
1. Open browser to `http://localhost:3000`
2. Allow camera/microphone access
3. Watch for "Connected (WebRTC)" indicator
4. Speak or wait for mock data
5. Notification should appear next to face

## Next Steps

1. **Integrate with audio pipeline**: Add `broadcast_person()` calls when speakers are identified
2. **Connect to MongoDB**: Replace stub vector store with real Atlas instance
3. **Add person metadata**: Enhance `PersonData` with more context fields
4. **Add error handling**: Graceful degradation when SSE disconnects

## Troubleshooting

### SSE Not Connecting
- Check CORS headers are set
- Verify backend is running on correct port
- Check browser console for connection errors

### WebRTC Not Connecting
- Ensure camera/microphone permissions granted
- Check network tab for failed `/offer` request
- Verify aiortc is installed in backend

### No Person Data Appearing
- Check audio pipeline is processing chunks
- Verify `broadcast_person()` is being called
- Check SSE connection is established
- Look for errors in backend logs

