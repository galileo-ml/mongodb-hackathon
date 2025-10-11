# mongodb-hackathon

## Getting Started

1. Ensure you have Python 3.10+ and [uv](https://github.com/astral-sh/uv) installed.
2. Create a `.env` file at the project root (same folder as this README) and set the Hugging Face token required by pyannote (only used for speaker embeddings). WebRTC VAD handles denoising locally; optional RNNoise support is auto-detected if installed:
   ```env
   PYANNOTE_AUTH_TOKEN=hf_...
   ```
   The application automatically loads this file on startup.
3. Install dependencies and launch the backend:
   ```bash
   uv sync
   uv run -- uvicorn backend.app.main:app --reload
   ```
4. Visit `http://localhost:8000/`, allow camera/mic access, and monitor the server logs for diarization output. You should see periodic vector-store stats and WebRTC-VAD-filtered segment logs every ~30 seconds, along with RNNoise/energy gating skips when the environment is noisy.

The stack uses FastAPI + aiortc for WebRTC ingest, pyannote.audio for speaker embeddings, and stubs for the future MongoDB Atlas vector store.
