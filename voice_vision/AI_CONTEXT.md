# AI_CONTEXT — Voice, language and vision

**Owner:** [MEMBER 5 NAME]

You are helping me build one lane of a hackathon project. Read this fully before answering.

## Project summary
VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai. Tagline: "Know your street before the water does." We combine an hourly rainfall forecast (Open-Meteo) with terrain features and past Chennai flood hotspots (2015, 2023) to rank flood risk per ~250 m grid cell for the next 24–72 hours. An XGBoost risk engine produces a score and band (low/medium/high). An LLM turns the risk into a short action message in Tamil and English and text-to-speech reads it aloud. Residents upload a waterlogging photo; a pretrained vision model verifies it (location + time + visual check) and verified reports refine the hotspot map. We claim hotspot RANKING, never exact flood depth. 24-hour sprint, 5-person team.

## My lane and folder
I am the **Voice, language and vision** owner. My folder is `voice_vision/` (branch `voice-vision`), with prompts in `voice_vision/prompts/`. Other lanes (ml/, data/, backend/, frontend/) are not mine.

## Contract
`docs/contract.md`, section C (module interfaces), plus A (conventions) and B (shape of `cell` and `peak_window`). Follow it exactly. Never change the contract; ask me to request a change from the lead.
- `voice_vision/alerts.py`: `generate_alert(cell: dict) -> {"message_en","message_ta","actions"}`
- `voice_vision/tts.py`: `synthesize(text: str, lang: str) -> str` (URL or file path under a served audio folder)
- `voice_vision/vision.py`: `verify_photo(image_bytes: bytes, lat: float, lon: float, timestamp: str) -> {"status","confidence","reason"}`

## Rules for the AI
- Only create or edit files inside `voice_vision/`. Never edit `docs/contract.md`.
- Prefer simple, readable code that works over clever code.
- Small commits; suggest a message after each piece.
- No secrets or API keys in code. Use a `.env` file and `.env.example`.
- Write a runnable example or test for every function.
- If blocked on another module, create a mock that matches the contract.
- State assumptions explicitly.
- Keep answers short.

## Tech stack
Python 3.11, an LLM API for alert text (key via env), a TTS service supporting Tamil and English (e.g. a cloud TTS or gTTS as a quick option), PyTorch + a pretrained vision model (e.g. CLIP via Hugging Face transformers for zero-shot "flooded street" classification), Pillow, haversine/geopy-style distance, pytest.

## Exact deliverables
1. `alerts.py`: LLM-generated, short (about 40 words max), calm, action-first messages in English and Tamil with 2–3 concrete actions (avoid underpasses, move vehicle, charge phone...). Prompts stored in `voice_vision/prompts/`. Template fallback if the LLM fails. Wording says "risk", never flood depth.
2. `TAMIL_REVIEW.md`: native-speaker review and recorded fixes.
3. `tts.py`: Tamil and English speech, saved to the audio folder agreed with backend; returns the URL/path.
4. `vision.py`: pretrained model visual check + location proximity (photo vs claimed risk cell) + timestamp freshness; returns `status` (pending|verified|rejected), `confidence` 0–1, one-line `reason`.
5. A test set of 10 sample images with expected outcomes and a script that scores it.

## Acceptance checks to help me pass
- Signatures and return keys match contract C exactly; `status` values limited to pending|verified|rejected.
- Tamil text is valid Unicode, natural, and reviewed; messages are short and action-first.
- No secrets in code; works with a template fallback when no API key is set.
- Vision accounts for all three checks and gives a one-line reason; test-set accuracy reported honestly.
- Each function has a runnable example or test.

## Starter prompts
1. "Write the alert-generation prompt files in voice_vision/prompts/ for English and Tamil: calm, action-first, max 40 words, 2–3 concrete actions chosen from the cell's risk_band and drivers."
2. "Implement `generate_alert(cell)` with an LLM call using an env var for the key, JSON output {message_en, message_ta, actions}, validation, and a template fallback per risk band."
3. "Implement `synthesize(text, lang)` for 'ta' and 'en' that writes an mp3 into the audio folder and returns its URL path; add a test that checks the file exists."
4. "Implement `verify_photo` using a pretrained CLIP model for zero-shot 'waterlogged street' vs 'dry street', plus a distance check against the cell and a timestamp freshness check (<= 3 hours); combine into status/confidence/reason."
5. "Create `voice_vision/tests/test_set.md` and a script to run 10 sample images with expected outcomes and print accuracy and a confusion table."
6. "Draft 10 Tamil alert messages for low/medium/high risk with different drivers and a checklist I can send to a native speaker for review in TAMIL_REVIEW.md."
