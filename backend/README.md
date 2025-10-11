# Backend Service

## Overview
FastAPI application that accepts WebRTC offers, ingests live audio, and runs a diarization pipeline (denoising, VAD segmentation, speaker embedding, Whisper transcription, and MongoDB Atlas vector-store stubs).

## Prerequisites
- Python 3.10 or newer
- `ffmpeg` available on `PATH` (required by Whisper)
- Hugging Face account with access to `pyannote/embedding`

## Installation
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install PyTorch + Torchaudio first (CPU wheels shown below)
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

## Environment Configuration
- Copy `.env.example` (if present) to `.env` and adjust ports or logging.
- Export a Hugging Face token so Pyannote can download gated weights:
  ```bash
  export PYANNOTE_AUTH_TOKEN=<your-token>
  ```
- Optionally override the Whisper size (default `large-v3`) with `WHISPER_MODEL`; smaller models reduce startup time and memory.

## Running the Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Visit `http://localhost:8000` for the dummy frontend harness.
- Establish a WebRTC session; audio tracks will flow through the pipeline and log speaker matches + transcription summaries.
- Shutdown gracefully to flush buffered audio and final metrics.

## Optional Features & Notes
- RNNoise denoiser auto-enables when the `rnnoise` wheel loads; otherwise audio passes through.
- Without torchaudio or Whisper installed, the pipeline still runs but skips advanced diarization/transcription.
- Vector-store operations run against the in-memory stub; integrate a real Atlas collection by swapping `MongoDBVectorStore` with a production client.
