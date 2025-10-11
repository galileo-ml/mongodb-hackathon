# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI WebRTC ingress service with entrypoint at `app/main.py`.
- `mock-backend/`: Standalone FastAPI simulator (`env.example`, `QUICKSTART.md`) for UI testing without live services.
- `frontend/`: Next.js 15 app; routes in `app/`, shared UI in `components/`, hooks in `hooks/`, workers in `workers/`, styling in `styles/`.
- `inference/`: SSE prototype trio (`main.py`, `mock_metadata_service.py`, `mock_consumer.py`) with one-click setup via `setup.sh`.
- `dummy_frontend/`: Legacy static WebRTC harness retained for regression checks and fallback demos.

## Build, Test, and Development Commands
- `uv sync` (root or service folder): installs Python dependencies tracked in `pyproject.toml`/`uv.lock`; fallback to `pip install -r requirements.txt` when uv is unavailable.
- `uv run uvicorn backend.app.main:app --reload --port 8000`: serves the production backend locally for WebRTC offer handling.
- `cd mock-backend && uv run uvicorn app.main:app --reload --port 8000`: spins up the mock backend used by the Next.js client during demos.
- `cd frontend && pnpm install` followed by `pnpm dev`: installs packages and launches the custom Next.js dev server at `http://localhost:3000` (npm equivalents also work).
- `cd frontend && pnpm lint`: runs the ESLint and TypeScript checks enforced by Next.js.
- `cd inference && ./setup.sh`: provisions a virtualenv; then run `python main.py` or `python mock_metadata_service.py` in separate terminals for end-to-end SSE flows.

## Coding Style & Naming Conventions
- Keep Python modules PEP 8 compliant with 4-space indents, explicit type hints, and `snake_case` filenames.
- Follow React/TypeScript norms: PascalCase for components, camelCase utilities, and colocate UI logic with Tailwind-powered styles.
- Reuse existing logging patterns and avoid introducing new global configuration without documenting impacts in `plan.md` or service README files.

## Testing Guidelines
- No automated test suites yet: verify health endpoints via `curl http://localhost:8000/health` and stream integrity with `curl -N` against SSE routes (`/stream/conversation`, `/stream/inference`, `/events`).
- Frontend changes must pass `pnpm lint` and a manual smoke test in Chromium-based browsers with camera access granted.
- New automated tests should mirror structure (`tests/` beside Python services, `__tests__/` for frontend) and document coverage expectations in accompanying PR descriptions.

## Commit & Pull Request Guidelines
- Recent history is inconsistent (`save`, `one shot inference`); prefer imperative, scoped messages like `feat(frontend): add call status banner` and reference issues in the body.
- Summaries must state affected services and include verification notes (e.g., `pnpm lint`, SSE curl transcript).
- Pull requests should describe context, link design docs if relevant, attach UI screenshots or terminal captures, and call out cross-service dependencies or config changes.

## Security & Configuration Tips
- Duplicate `mock-backend/env.example` when introducing secrets; never commit live credentials or TURN/TLS keys.
- Document any WebRTC TURN/STUN updates, expected ports, and CORS changes directly in the PR to keep frontend/backend in sync.
- Remove transient demo tokens immediately after recording sessions and scrub logs before sharing externally.
