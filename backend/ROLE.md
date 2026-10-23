# ROLE — Backend API

**Owner:** [MEMBER 3 NAME]

## Mission
Serve the REST API from the contract so the frontend can build against it from H+1, then replace the mocks with the real ML model and the voice/vision modules. You are the integration hub: if your API matches the contract, the whole demo works.

## What you own
- The FastAPI app in `backend/` implementing every endpoint in contract section B.
- Mock mode (`USE_MOCK=1`) returning realistic fake data in the exact contract format.
- Wiring `ml.predict.predict_risk`, `voice_vision.alerts.generate_alert`, `voice_vision.tts.synthesize`, `voice_vision.vision.verify_photo`.
- CORS for the frontend; serving audio files; report storage (SQLite or JSON).
- OpenAPI docs, `backend/README.md`, `.env.example`, and free-tier deployment.

## Deliverables
- [ ] H+1: FastAPI skeleton with `/api/health` and all other endpoints returning mock data in exact contract format (`USE_MOCK=1`); CORS on; shared with frontend (local run instructions)
- [ ] H+2: `backend/README.md` with run instructions and example curl for each endpoint
- [ ] H+4: Report storage (SQLite or JSON) + `POST /api/report` and `GET /api/reports` working in mock mode
- [ ] H+6: Static audio serving route and `audio_url` in `/api/alert`
- [ ] H+8: Free-tier deployment live with mock mode (public URL given to frontend)
- [ ] H+10: First end-to-end run (checkpoint): frontend -> backend -> mock modules
- [ ] H+12: Wire `predict_risk` (stub, then real) into `/api/risk` behind a flag
- [ ] H+14: Real model wired in (checkpoint); wire `generate_alert`, `synthesize`
- [ ] H+16: Wire `verify_photo` into `POST /api/report`; reports update status
- [ ] H+18: Error handling: `{"error": str}` with proper status; input validation; caching of `/api/risk` for a short TTL
- [ ] H+20: Feature freeze; redeploy final version
- [ ] H+23: Demo rehearsal done; fallback to mock mode documented if a module fails

## Definition of done
- Every endpoint in contract B responds with exactly the documented JSON shape in both mock and real mode.
- Errors are `{"error": "..."}` with 400/404/500 as appropriate.
- `GET /docs` (OpenAPI) works; a fresh teammate can run the server from `backend/README.md` in under 5 minutes.
- Deployed URL works from a phone; `/api/risk` responds in under 3 seconds.
- Switching `USE_MOCK` 1/0 requires no code changes.

## Out of scope (do NOT touch)
- Model training or prediction logic (ml/); you only import `predict_risk`.
- Data downloading (data/).
- UI code (frontend/).
- Prompt writing, TTS and vision internals (voice_vision/).
- `docs/contract.md` — request changes from the lead.

## Git branch and folder
- Branch: `backend`
- Folder: `backend/`

## Cut-first list (drop in this order if time runs short)
1. Short-TTL caching.
2. SQLite (use a JSON file).
3. Free-tier deployment polish (run locally with a tunnel such as ngrok for the demo).
4. Rate limiting / auth (none needed).
5. `GET /api/reports` bbox filtering beyond a simple range check.
Never cut: mock mode for all endpoints and CORS.
