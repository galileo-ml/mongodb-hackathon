# mongodb-hackathon

## Getting Started

1. Ensure you have Python 3.10+ and [uv](https://github.com/astral-sh/uv) installed.
<<<<<<< HEAD
2. Install dependencies (Whisper downloads a model the first time and requires FFmpeg on your system) and launch the backend:
=======
2. Create a `.env` file at the project root (same folder as this README) and set the Hugging Face token required by pyannote (only used for speaker embeddings). WebRTC VAD + local speaker tracking thin out background noise; optional RNNoise support is auto-detected if installed:
   ```env
   PYANNOTE_AUTH_TOKEN=hf_...
   ```
   The application automatically loads this file on startup.
3. Install dependencies (Whisper requires FFmpeg on your system) and launch the backend:
>>>>>>> main
   ```bash
   uv sync
   uv run -- uvicorn backend.app.main:app --reload
   ```
<<<<<<< HEAD
3. Visit `http://localhost:8000/`, allow camera/mic access, and monitor the server logs. After each conversation (8 s of silence), Whisper prints a red transcript snippet.

The stack now uses FastAPI + aiortc for WebRTC ingress, optional RNNoise denoising, and Whisper for speech-to-text on conversation boundaries.
=======
4. Visit `http://localhost:8000/`, allow camera/mic access, and monitor the server logs. You’ll see WebRTC-VAD segments, 10 s conversation boundaries, local-speaker reuse without vector DB hits, and periodic vector-store stats.

The stack uses FastAPI + aiortc for WebRTC ingest, pyannote.audio for speaker embeddings, Whisper for end-of-conversation snippets, and stubs for the future MongoDB Atlas vector store.
>>>>>>> main
