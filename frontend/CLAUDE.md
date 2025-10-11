# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ForgetMeNot** - An AI-powered real-time webcam streaming application built for a MongoDB hackathon. The app captures webcam video frames and streams them via WebSocket to a backend for AI analysis and response generation.

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

**Important**: This project uses a **custom Next.js server** (`server.js`) instead of the default Next.js dev server. The custom server is required for WebSocket support.

## Architecture

### Custom Server Architecture

This app uses a custom Node.js server (`server.js` in root) that:
- Runs the Next.js app
- Hosts a WebSocket server at `/api/ws`
- Handles WebSocket upgrade requests
- Processes binary video frame data from clients
- Returns mock AI responses (currently placeholder logic for backend integration)

**Key point**: Standard Next.js API routes coexist with the WebSocket endpoint. The WebSocket server intercepts upgrade requests at `/api/ws`, while all other routes are handled by Next.js.

### Video Streaming Flow

1. **Frontend** (`components/webcam-stream.tsx`):
   - Accesses user's webcam via `navigator.mediaDevices.getUserMedia()`
   - Captures frames at ~10 FPS using Canvas API
   - Converts frames to JPEG blobs
   - Sends binary frame data via WebSocket to `/api/ws`
   - Receives AI responses via WebSocket messages
   - Displays responses in AI overlay component

2. **WebSocket Server** (`server.js`):
   - Accepts WebSocket connections
   - Receives binary frame data (JPEG images as ArrayBuffers)
   - Currently sends mock AI responses
   - **TODO**: Replace mock responses with actual backend AI/LLM integration

3. **AI Overlay** (`components/ai-overlay.tsx`):
   - Displays AI-generated responses over the video feed
   - Provides tabs for suggestions and follow-up questions
   - Dismissible modal interface

### WebSocket Protocol

**Client → Server** (binary):
- Raw JPEG image data as ArrayBuffer

**Server → Client** (JSON):
```json
{
  "type": "connection" | "ai-response",
  "data": {
    "message"?: string,
    "llmResponse"?: string,
    "timestamp": string
  }
}
```

### Component Structure

- **Main page**: `app/page.tsx` - Simple container for WebcamStream component
- **WebcamStream**: `components/webcam-stream.tsx` - Core component handling:
  - Webcam access and streaming
  - WebSocket connection management
  - Frame capture and transmission
  - Automatic reconnection on disconnect
  - Connection status indicator (top-right)
- **AiOverlay**: `components/ai-overlay.tsx` - Modal overlay for displaying AI responses
- **UI Components**: `components/ui/*` - shadcn/ui component library (New York style)

### Tech Stack

- **Framework**: Next.js 15 (App Router)
- **WebSocket**: `ws` library
- **UI Library**: shadcn/ui (Radix UI primitives)
- **Styling**: Tailwind CSS 4
- **Icons**: Lucide React
- **Fonts**: Inter (sans), Geist Mono
- **Package Manager**: pnpm

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

## Integration Points for Backend

When connecting to a real backend AI service:

1. **Replace mock responses** in `server.js` (lines 42-76):
   - Remove the mock response array
   - Add actual AI/LLM API calls
   - Process the received JPEG binary data
   - Return structured responses

2. **Alternative**: Point WebSocket URL to external backend:
   - Modify `connectWebSocket()` in `components/webcam-stream.tsx` (line 33)
   - Change `wsUrl` to point to your backend WebSocket server
   - Ensure backend expects binary JPEG data and returns JSON with `type` and `data` fields

3. **Frame rate adjustment**:
   - Current: ~10 FPS (100ms interval) in `webcam-stream.tsx:162`
   - Adjust interval based on backend processing capabilities

## Known Limitations

- No audio streaming implemented (mute button is UI-only)
- No video recording/playback functionality
- WebSocket errors don't show user-friendly messages
- No frame buffering or quality adjustment based on connection
