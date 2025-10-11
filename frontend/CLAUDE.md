# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ForgetMeNot** - An AI-powered real-time webcam streaming application built for a MongoDB hackathon. The app captures webcam video/audio and streams them via WebRTC to a backend for AI analysis, then receives person detection notifications via Server-Sent Events (SSE).

## Development Commands

```bash
# Install dependencies
pnpm install

# Start development server (runs custom Node.js server with WebSocket support)
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start

# Run linter
pnpm lint
```

**Important**: This project uses standard Next.js dev server. The backend is a **separate Python FastAPI server** that handles WebRTC and SSE.

## Architecture

### Two-Server Architecture

This app uses a **decoupled architecture** with two servers:

1. **Frontend Server** (Next.js on port 3000)
   - Standard Next.js app
   - No custom server needed
   - Handles UI rendering only

2. **Backend Server** (FastAPI on port 8000)
   - Python FastAPI server (see `mock-backend/` or `backend/`)
   - Handles WebRTC video/audio ingress
   - Processes frames for AI analysis
   - Streams notifications via SSE

### Video Streaming Flow (WebRTC + SSE)

1. **Frontend** (`components/webcam-stream.tsx`):
   - Accesses user's webcam + microphone via `getUserMedia()`
   - Establishes WebRTC connection to backend
   - Sends offer to `http://localhost:8000/offer`
   - Streams video + audio tracks continuously via WebRTC
   - Subscribes to SSE endpoint for AI notifications
   - Displays person detection events in AI overlay

2. **Backend** (`mock-backend/app/main.py`):
   - Accepts WebRTC connections via `/offer` endpoint
   - Receives video + audio tracks in real-time
   - Processes frames (mock: every 5 seconds sends notification)
   - Broadcasts events to all SSE clients
   - Logs data reception every 5 seconds

3. **AI Overlay** (`components/ai-overlay.tsx`):
   - Displays person detection notifications over video feed
   - Shows context about detected people
   - Provides tabs for suggestions and follow-up questions
   - Dismissible modal interface

### Communication Protocols

**WebRTC (Frontend → Backend)**:
- Video + audio streams sent continuously
- Peer connection established via offer/answer negotiation
- Backend receives `RTCTrack` objects for processing

**SSE (Backend → Frontend)**:
- Server-Sent Events for one-way push notifications
- Endpoint: `GET http://localhost:8000/events?session_id=<id>`
- Event format:
```json
{
  "type": "person-detected" | "ai-response" | "connection",
  "data": {
    "llmResponse": "Person detected: John Doe...",
    "personId"?: "person_0",
    "confidence"?: 0.95
  },
  "timestamp": "2025-10-11T12:00:00Z"
}
```

### Component Structure

- **Main page**: `app/page.tsx` - Simple container for WebcamStream component
- **WebcamStream**: `components/webcam-stream.tsx` - Core component handling:
  - Webcam + microphone access
  - WebRTC peer connection setup and management
  - SSE subscription for AI notifications
  - Connection status indicator (top-right)
- **AiOverlay**: `components/ai-overlay.tsx` - Modal overlay for displaying person detection notifications
- **UI Components**: `components/ui/*` - shadcn/ui component library (New York style)

### Tech Stack

**Frontend:**
- **Framework**: Next.js 15 (App Router)
- **WebRTC**: Browser native `RTCPeerConnection`
- **SSE**: Browser native `EventSource`
- **UI Library**: shadcn/ui (Radix UI primitives)
- **Styling**: Tailwind CSS 4
- **Icons**: Lucide React
- **Fonts**: Inter (sans), Geist Mono
- **Package Manager**: pnpm

**Backend (Mock):**
- **Framework**: FastAPI (Python)
- **WebRTC**: `aiortc` library
- **Server**: Uvicorn ASGI server
- **Package Manager**: `uv`

### Path Aliases

TypeScript and imports use `@/*` alias:
- `@/components` → `components/`
- `@/lib` → `lib/`
- `@/hooks` → `hooks/`
- `@/app` → `app/`

## Configuration Notes

- **TypeScript**: Strict mode enabled, but build errors are ignored (`ignoreBuildErrors: true`)
- **ESLint**: Ignored during builds (`ignoreDuringBuilds: true`)
- **Images**: Unoptimized for faster development
- **Dark mode**: Enabled by default in layout
- **shadcn/ui**: Configured with "new-york" style variant, CSS variables enabled

## Testing with Mock Backend

### Quick Start (2 Terminal Windows)

**Terminal 1 - Start Mock Backend:**
```bash
cd mock-backend
uv sync  # Install dependencies (first time only)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
pnpm install  # First time only
pnpm dev
```

**Browser:**
1. Open http://localhost:3000
2. Allow camera + microphone access
3. Watch for "Connected (WebRTC)" indicator (top-right)
4. Person notifications will appear every 5 seconds
5. Backend logs will show data reception every 5 seconds

### Mock Backend Behavior

- **Data logging**: Every 5 seconds, logs video FPS and frame count
- **Person notifications**: Every 5 seconds, broadcasts mock person detection to SSE clients
- **Mock people**: Cycles through 5 mock people (John Doe, Jane Smith, etc.)

## Integration Points for Real Backend

When building the real backend:

1. **Replace mock-backend with real backend**:
   - Build in `backend/` directory instead of `mock-backend/`
   - Implement same endpoints: `/offer` (WebRTC) and `/events` (SSE)
   - Add actual AI processing:
     - Face detection and recognition
     - Speaker diarization
     - MongoDB Atlas Vector Search for context retrieval
     - LLM integration for response generation

2. **Backend URL configuration**:
   - Current: `BACKEND_URL = "http://localhost:8000"` in `webcam-stream.tsx:18`
   - Change to production backend URL when deploying

3. **Notification timing**:
   - Mock: Every 5 seconds
   - Real: Event-driven when person detected/identified

## Known Limitations

- Mock backend only (no real AI processing yet)
- No video recording/playback functionality
- No frame buffering or quality adjustment based on connection
- Mute button is UI-only (audio is always captured for WebRTC)
