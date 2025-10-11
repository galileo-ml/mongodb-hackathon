# Multimodal Speaker Diarization System Plan

## 1. Vision & Goals
- Ingest live webcam (default) or recorded video, identify who is speaking, and link voices to faces in near real-time.
- Persist per-person memories (conversation summaries, metadata, engagement history) to support future personalization.
- Provide an API to retrieve the current context for any recognized individual via face or voice matching.
- Design modular components so we can iterate on models, storage, and interaction layers independently.

## 2. Guiding Principles
- **Modality fusion first-class**: treat voice and face embeddings as peers with shared identity resolution.
- **Incremental refinement**: start with robust diarization and face recognition; defer higher-order reasoning until baselines are stable.
- **Explainable identities**: log evidence (voiceprints, face crops, timestamps) for each identity link to aid debugging.
- **Extensible memory**: store structured context summaries so downstream applications can enrich or query easily.

## 3. High-Level Architecture
1. **Capture Layer**: webcam capture service (USB/UVC) produces synchronized audio waveform + video frames.
2. **Preprocessing Pipelines**
   - Audio: voice activity detection (VAD), denoising, turn segmentation.
   - Video: face detection, tracking, and frame sampling aligned to audio turns.
3. **Feature Extraction**
   - Speaker embeddings via pretrained diarization/verification model (e.g., pyannote, Resemblyzer).
   - Face embeddings via modern face recognizer (e.g., ArcFace, FaceNet).
4. **Identity Resolution Service**
   - Maintains gallery of known embeddings per person.
   - Uses clustering + cross-modal matching heuristics to link new voice/face observations to existing identities or create new ones.
5. **Context & Memory Service (MongoDB backing, stubbed initially)**
   - Stores person profiles, interaction history, and aggregated summaries.
   - Exposes `get_person_context(person_id)` / `upsert_interaction(...)` interfaces.
6. **Application Layer**
   - Real-time dashboard, API endpoints, or downstream agents consume consolidated identity + context data.

## 4. Core Modules & Responsibilities

### 4.1 Capture & Synchronization
- Implement capture adapters for local USB/UVC webcam (default path) and optional RTSP/file ingestion.
- Handle dual-channel capture (video frames + microphone audio) from the webcam; ensure consistent device selection and fallback handling when hardware is missing.
- Add lightweight calibration routine (resolution, frame rate, exposure) to stabilize incoming frames before downstream processing.
- Guarantee synchronized timestamps between audio chunks and video frames (consider MediaPipe, OpenCV + ffmpeg).

### 4.2 Audio Pipeline & Diarization
- Voice Activity Detection to prune silence/noise.
- Segment audio into speaker turns (diarization model) and label with provisional speaker IDs.
- Extract speaker embeddings per turn; maintain rolling average per provisional speaker.
- Optional: keyword spotting or ASR hooks to enrich context (future).

### 4.3 Video Pipeline & Face Analytics
- Detect faces frame-by-frame, track across frames (e.g., SORT/DeepSORT/ByteTrack).
- Generate face embeddings at key frames (aligned with speaker turns where possible).
- Handle occlusions, low-light conditions, and multi-face scenes via confidence scoring.

### 4.4 Multimodal Association Layer
- Align speaker turns with face tracks using overlapping timestamps + proximity heuristics.
- For each turn, compute association score between speaker embedding and candidate face embeddings.
- Maintain joint identity graph: nodes = provisional voice/face clusters, edges = evidential matches.
- Promote to stable person ID when confidence exceeds threshold; log supporting evidence.

### 4.5 Identity & Memory Store (MongoDB Stub)
- Collections (initial draft):
  - `persons`: canonical profiles, embedding prototypes, metadata.
  - `interactions`: time-stamped records of turns, transcripts, evidence references.
  - `contexts`: aggregated summaries per person (latest view).
- Define stubbed repository layer returning in-memory mocks while Mongo integration is pending.
- Establish schema versioning + migration strategy.

### 4.6 Context Aggregation & Retrieval
- Aggregation jobs update person context after each interaction (e.g., summary, sentiment, topics).
- Provide accessors:
  ```python
  def get_person_context(person_id: str) -> PersonContext:
      """Return aggregated view: profile, recent interactions, summary."""

  def upsert_interaction(event: InteractionEvent) -> None:
      """Persist new evidence and trigger incremental aggregation."""
  ```
- Plan for caching and TTL to support low-latency queries.

## 5. Memory & Conversation Handling
- Aggregate per-person timeline with links to raw media snippets for audit.
- Support cross-interaction memory (e.g., recurring topics, known preferences).
- Build fallback path for unnamed/anonymous individuals (store session-limited IDs until resolved).
- Enable global speaker insights (e.g., top speakers, talk-time ratios) for analytics.

## 6. Processing Modes
- **Real-time**: streaming pipelines using async queues (e.g., asyncio, Kafka-like bus) with sliding-window inference; ensure frame/audio buffers read directly from webcam capture without dropping frames.
- **Batch**: offline processing for archived videos with job orchestration (Celery, prefect) reusing same modules.
- Instrument both for latency, throughput, and failure recovery.

## 7. Observability & Ops
- Structured logging with correlation IDs per person/interaction.
- Metrics: diarization accuracy, association confidence, latency per module, memory freshness.
- Alert on identity merge/split events, high-conflict matches, datastore write failures.

## 8. Security & Privacy Considerations
- Handle personally identifiable information (PII) with opt-in consent tracking.
- Provide configurable retention periods; support deletion requests.
- Encrypt embeddings and transcripts at rest; restrict access via RBAC.
- Audit trail for identity link changes and memory updates.

## 9. Evaluation Strategy
- Curate labeled multimodal datasets (internal recordings + public corpora) for benchmarking.
- Metrics: diarization error rate (DER), face ID accuracy, association precision/recall, memory correctness.
- Establish regression suite for new model versions and edge-case scenarios (crowded scenes, overlapping speech).

## 10. Roadmap (Draft)
1. Prototype webcam ingestion + audio diarization pipeline with mock storage.
2. Add face detection/tracking and align outputs; validate on small dataset.
3. Implement multimodal association heuristics; iterate until confident matches.
4. Introduce identity store abstraction + stubbed Mongo repository.
5. Build context aggregation layer with simple summarization (rule-based or LLM-assisted later).
6. Expose retrieval API (`get_person_context`) + minimal dashboard/CLI explorer.
7. Harden for real-time performance, add observability, finalize data governance.

## 11. Open Questions
- Desired latency targets for real-time deployments?
- Constraints on hardware acceleration (GPU availability, edge devices)?
- Expected webcam specs (resolution, frame rate, audio path) and whether multiple cameras must be supported?
- Preferred summarization techniques for context aggregation (rules vs. LLMs)?
- How do we reconcile conflicting identity evidence (manual review workflow vs. automatic)?
- Integration touchpoints for downstream applications (webhooks, message bus, direct queries)?

## 12. Component APIs (Parallel Workstreams)

### 12.1 Shared Data Contracts
- Define lightweight Pydantic/dataclass models in `common.schemas` to prevent cross-team coupling:
  ```python
  class AudioTurn(BaseModel):
      turn_id: str
      stream_id: str
      start_ts: datetime
      end_ts: datetime
      embedding: np.ndarray  # normalized speaker vector
      raw_uri: str  # optional pointer to audio snippet

  class FaceTrack(BaseModel):
      track_id: str
      stream_id: str
      frames: list[FrameRef]  # frame refs with bounding boxes
      embedding: np.ndarray  # representative face vector
      start_ts: datetime
      end_ts: datetime

  class IdentityMatch(BaseModel):
      person_id: str | None
      confidence: float
      evidence: list[EvidenceLink]

  class InteractionEvent(BaseModel):
      event_id: str
      person_id: str
      audio_turn_id: str | None
      face_track_id: str | None
      transcript: str | None
      observed_at: datetime

  class PersonContext(BaseModel):
      person_id: str
      profile: dict[str, Any]
      summary: str | None
      recent_events: list[InteractionEvent]
      last_updated: datetime
  ```

### 12.2 Speaker/Face Identification + Identity Store API
- Lives under `identity_service/`; wraps diarization, face recognition, and persistence stubs.
- Public interface (callable from capture worker or message bus consumer):
  ```python
  class IdentityServiceProtocol(Protocol):
      def ingest_audio_turn(self, turn: AudioTurn) -> IdentityMatch:
          """Upsert speaker embedding, return best identity match (new or existing)."""

      def ingest_face_track(self, track: FaceTrack) -> IdentityMatch:
          """Upsert face embedding, return best identity match (new or existing)."""

      def link_modalities(self, turn_id: str, track_id: str) -> IdentityMatch:
          """Fuse evidence across modalities, update confidence, and emit resolved person_id."""

      def get_identity_profile(self, person_id: str) -> PersonContext:
          """Fetch core profile details + embedding exemplars (stubbed Mongo)."""

      def register_manual_identity(self, person_id: str, metadata: dict[str, Any]) -> None:
          """Allow manual overrides/merges when automated matching is inconclusive."""
  ```
- Emits `InteractionEvent` messages (via queue or direct call) for downstream aggregation once identity is resolved.

### 12.3 Context Aggregation API
- Lives under `context_aggregation/`; consumes identity-resolved events.
- Responsibilities: maintain rolling summaries, sentiment/topics, and hand-offs to memory store.
- Interface:
  ```python
  class ContextAggregatorProtocol(Protocol):
      def record_interaction(self, event: InteractionEvent) -> None:
          """Persist event (stubbed Mongo), update in-memory aggregation state."""

      def rebuild_person_context(self, person_id: str) -> PersonContext:
          """Recompute full summary from source events (used for backfill / corrections)."""

      def flush_batch(self) -> None:
          """Commit buffered aggregates to the memory store at configurable cadence."""
  ```
- Publishes `PersonContext` snapshots to a cache (Redis/in-memory placeholder) accessible by retrieval API.

### 12.4 Context Retrieval API
- Lives under `context_retrieval/`; exposes query surface for UI/agents.
-  Should be deployable as REST/gRPC service or library wrapper.
- Interface:
  ```python
  class ContextRetrievalProtocol(Protocol):
      def get_person_context(self, person_id: str) -> PersonContext | None:
          """Return latest aggregated context; fall back to aggregator if cache miss."""

      def search_identities(self, query: str, limit: int = 10) -> list[PersonContext]:
          """Lookup by name metadata, tags, or semantic summary."""

      def get_recent_interactions(self, person_id: str, limit: int = 20) -> list[InteractionEvent]:
          """Surface chronological view for audit or UI detail panel."""
  ```
- Consumes read-only repositories (Mongo stub + cache) and never mutates state directly.

### 12.5 Integration Notes
- Components communicate through well-defined DTOs (`AudioTurn`, `FaceTrack`, `InteractionEvent`, `PersonContext`).
- Use message queue abstraction (e.g., `EventBus.publish(event: InteractionEvent)`) to decouple identity service and aggregator.
- Shared contracts versioned separately to minimize merge conflicts across parallel streams.

## 13. Browser-Oriented Frontend/Backend Flow

### 13.1 Frontend (Web)
- Browser obtains webcam stream via `navigator.mediaDevices.getUserMedia`; central video tile renders the raw feed while a `Canvas`/`OffscreenCanvas` layer draws overlays (bounding boxes, speaker labels, context snippets).
- Capture path options:
  - **Streaming**: pipe media tracks to backend through WebRTC (preferred) with VP8/H264 video + Opus audio; fall back to chunked WebSocket uploads using `MediaRecorder` if peer connection unsupported.
  - **Non-streaming**: periodically capture short clips or still frames + audio bursts, upload over HTTPS for batch processing.
- UI maintains WebSocket/DataChannel subscription for identity/context updates; applies overlays in near real-time without rerequesting the media stream.

### 13.2 Backend Ingress & Processing
- Deploy a signaling service to negotiate WebRTC sessions (can be part of backend API).
- Ingress service demuxes remote tracks into audio/video pipelines already defined (Section 4); timestamps from RTP packets feed directly into synchronization module.
- For chunked uploads, a media assembler service reconstructs timelines before handing off to diarization/face pipelines.
- Identity service emits overlay payloads (e.g., `OverlayUpdate` containing face box coordinates, speaker name, confidence, context summary) to a publish/subscribe channel keyed by session.

### 13.3 Overlay & Output Delivery
- Real-time mode: backend pushes `OverlayUpdate` messages via WebRTC DataChannel or WebSocket; frontend draws overlays atop the central video tile and updates side panels with context data (`PersonContext` snippets, conversation history links).
- Non-streaming mode: backend returns annotated frames/video segments and structured context JSON; frontend replaces/augments the video tile when results arrive.
- Provide REST endpoints to download recorded sessions with baked-in overlays, enabling asynchronous review without live streaming.

### 13.4 State & Session Coordination
- Introduce `SessionManager` service to track active frontend sessions, map them to identity store and aggregator pipelines, and clean up resources when sessions end.
- Use short-lived tokens for WebRTC/WebSocket authentication; ensure session metadata (browser, user consent) is logged alongside interaction events for audit.

## 14. Backend Platform & API Strategy
- **Control Plane API**: lightweight FastAPI service (Python) aligns with ML-heavy pipelines, exposes REST endpoints for session management, querying contexts, and fallback upload flows. FastAPI natively supports async WebSocket routes to forward overlay updates when WebRTC DataChannels are unavailable.
- **Media Ingress**: implement WebRTC termination using `aiortc` (Python) for tight coupling with diarization modules, or integrate an SFU (e.g., Janus, mediasoup) if multi-viewer scaling is needed; FastAPI service can act as signaling endpoint regardless of media backend choice.
- **Processing Workers**: run identity, aggregation, and retrieval components as separate async workers (Python) connected via message bus (Redis Streams/NATS/Kafka depending on scale). Workers share MongoDB stub and caches.
- **Batch & Admin API**: reuse FastAPI app to provide admin tasks (manual identity merges, exporting sessions) while keeping real-time processing in worker processes to avoid blocking HTTP threads.
- **Deployment Footprint**: containerize services; orchestrate via Kubernetes or Docker Compose for dev. Scale media ingress horizontally; offload GPU-bound inference to dedicated worker pool with gRPC endpoints if needed.
- **Why FastAPI**: consistent typing (Pydantic models match Section 12 contracts), async support for WebSockets, easy integration with OpenAPI docs for frontend team. Alternative stacks (Node/TypeScript or Go) could host signaling/SFU if organizational preference, but Python remains ideal for ML-centric path.
