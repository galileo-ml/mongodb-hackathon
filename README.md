# mongodb-hackathon

## Getting Started

1. Ensure you have Python 3.10+ and [uv](https://github.com/astral-sh/uv) installed.
2. Install dependencies (Whisper downloads a model the first time and requires FFmpeg on your system) and launch the backend:
   ```bash
   uv sync
   uv run -- uvicorn backend.app.main:app --reload
   ```
3. Visit `http://localhost:8000/`, allow camera/mic access, and monitor the server logs. After each conversation (8 s of silence), Whisper prints a red transcript snippet.

The stack now uses FastAPI + aiortc for WebRTC ingress, optional RNNoise denoising, and Whisper for speech-to-text on conversation boundaries.
