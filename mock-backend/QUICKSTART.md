# Quick Start - Mock Backend

## What is this?

This is a **mock backend** for testing the frontend WebRTC + SSE integration. It simulates the real backend that you'll build in the `backend/` directory.

## Quick Setup (2 steps)

### 1. Start Mock Backend

```bash
cd mock-backend
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
🚀 Starting Mock FastAPI backend...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Start Frontend

```bash
cd ../frontend
npm install  # if you haven't already
npm run dev
```

### 3. Test in Browser

1. Open http://localhost:3000
2. Allow camera/microphone access
3. Watch for "Connected (WebRTC)" indicator (top-right)
4. Mock AI responses will appear every ~1 second

## What's Happening?

```
Browser                Mock Backend (port 8000)
  │                           │
  ├──── WebRTC ──────────────>│  Receives video+audio
  │     (video+audio)          │
  │                            │  Processes every 30 frames
  │                            │
  │<────── SSE ────────────────│  Sends mock AI response
        (AI events)
```

## Endpoints

- **POST /offer** - WebRTC negotiation
- **GET /events** - SSE stream for AI responses
- **WebSocket /ws** - Alternative to SSE
- **GET /health** - Health check

## Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# SSE stream (Ctrl+C to stop)
curl -N http://localhost:8000/events?session_id=test
```

## Next Steps

This mock backend is for **testing only**. Build your real backend in the `backend/` directory with:

1. Real AI processing
2. Face detection
3. Speaker diarization
4. MongoDB integration
5. Person identification
6. Context retrieval

See `../SETUP.md` for full architecture details.

