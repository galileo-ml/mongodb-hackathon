# Backend Overview

FastAPI service that accepts WebRTC offers at `/offer`, keeps peer connections open, and streams inbound audio tracks through an async pipeline. Each audio chunk is denoised (RNNoise when available), grouped by WebRTC VAD into speech segments, embedded with the pyannote speaker model, and checked against an in-memory MongoDB Atlas vector-store stub. The pipeline maintains per-session conversation state to reuse local speakers, promote them to global identities when similarity crosses configured thresholds, and flushes tail audio on disconnect.

When conversations go idle or a session closes, buffered segments are optionally transcribed with Whisper; the backend prints timestamped summaries and logs vector-store metrics on a timer. Optional dependencies (RNNoise, torchaudio, Whisper, pyannote) are guarded so the app still runs with stubs, but richer diarization features appear when those packages and tokens are present.

Problems:
- Too many speakers detected (even if only monologuing)
- s2t is not accurate (conversations logged do not accurately reflect what transpired)

Essentially, it's an accuracy issue.