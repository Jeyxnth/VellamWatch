# AI_CONTEXT — Backend API

**Owner:** [MEMBER 3 NAME]

You are helping me build one lane of a hackathon project. Read this fully before answering.

## Project summary
VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai. Tagline: "Know your street before the water does." We combine an hourly rainfall forecast (Open-Meteo) with terrain features and past Chennai flood hotspots (2015, 2023) to rank flood risk per ~250 m grid cell for the next 24–72 hours. An XGBoost risk engine produces a score and band (low/medium/high). An LLM writes short Tamil and English action messages and text-to-speech reads them aloud. Residents upload waterlogging photos, which a pretrained vision model verifies; verified reports refine the hotspot map. We claim hotspot RANKING, never exact flood depth. 24-hour sprint, 5-person team.

## My lane and folder
I am the **Backend API** owner. My folder is `backend/` (branch `backend`). Other lanes (ml/, data/, frontend/, voice_vision/) are not mine; I only import their modules.

## Contract
`docs/contract.md`, sections B (REST API) and C (Python module interfaces). Follow them exactly. Never change the contract; ask me to request a change from the lead.

## Rules for the AI
- Only create or edit files inside `backend/`. Never edit `docs/contract.md` or other folders.
- Prefer simple, readable code that works over clever code.
- Small commits; suggest a message after each piece.
- No secrets or API keys in code. Use `.env` and `.env.example`.
- Write a runnable example or test for every function (pytest with FastAPI TestClient; curl examples in README).
- If blocked on another module, create a mock that matches the contract (put it in `backend/mocks/`).
- State assumptions explicitly.
- Keep answers short.

## Tech stack
Python 3.11, FastAPI, uvicorn, pydantic, python-multipart (uploads), SQLite (sqlite3) or JSON file for reports, python-dotenv, pytest + httpx. Deploy on a free tier (Render, Railway or Fly.io; or a tunnel for the demo).

## Exact deliverables
1. FastAPI app, base path `/api`, with: `GET /api/health`, `GET /api/risk`, `GET /api/alert`, `POST /api/report` (multipart: photo, lat, lon, timestamp), `GET /api/reports`.
2. Mock mode via env `USE_MOCK=1`: realistic fake data in the exact contract format, so frontend can start at H+1.
3. Real mode: import `ml.predict.predict_risk`, `voice_vision.alerts.generate_alert`, `voice_vision.tts.synthesize`, `voice_vision.vision.verify_photo`.
4. CORS enabled for the frontend origin(s) from env.
5. Static serving of audio files; `audio_url` in `/api/alert`.
6. Report store (SQLite or JSON): report_id, lat, lon, timestamp, status, confidence, cell_id, reason.
7. Errors as `{"error": str}` with proper HTTP status.
8. OpenAPI docs and `backend/README.md` with run instructions and curl examples.
9. Free-tier deployment.

## Acceptance checks to help me pass
- Each endpoint's response matches the contract JSON shape exactly (write schema tests).
- Works with `USE_MOCK=1` and with `USE_MOCK=0` using stubs.
- Bad input (missing lat, bad lang) returns 400 with `{"error": ...}`.
- Frontend on another origin can call it (CORS).
- Fresh clone runs with the README commands only.

## Starter prompts
1. "Create the FastAPI skeleton in backend/ with all five contract endpoints returning realistic mock data when USE_MOCK=1, with pydantic models matching docs/contract.md exactly."
2. "Add CORS reading allowed origins from the ALLOWED_ORIGINS env var and create `.env.example`."
3. "Implement POST /api/report: save the uploaded photo, call verify_photo (or its mock), compute cell_id from lat/lon using cells.geojson, store the result in SQLite, and return the contract response."
4. "Wire GET /api/alert: find the cell for lat/lon via predict_risk, call generate_alert and synthesize, return message_en, message_ta, actions, audio_url; fall back to mocks per module on exceptions."
5. "Write pytest tests that check every endpoint's response shape against the contract, in mock mode."
6. "Write a Dockerfile / render.yaml for free-tier deployment and a README section explaining env vars and how to switch USE_MOCK."
