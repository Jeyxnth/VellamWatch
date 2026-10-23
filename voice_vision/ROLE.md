# ROLE — Voice, language and vision

**Owner:** [MEMBER 5 NAME]

## Mission
Turn a cell's risk into a calm, action-first message in English and Tamil, read it aloud, and verify resident-submitted waterlogging photos. This is the human-facing half of the product and the part judges will remember.

## What you own
- Everything in `voice_vision/`.
- `alerts.py`: LLM-based alert generation (short, calm, action-first, 2–3 concrete actions such as avoid underpasses, move vehicle, charge phone) in English and Tamil.
- `tts.py`: text-to-speech for Tamil and English.
- `vision.py`: photo verification with a pretrained vision model plus location proximity and timestamp freshness checks, returning status, confidence and a one-line reason.
- Prompts in `voice_vision/prompts/`.
- `voice_vision/TAMIL_REVIEW.md`: native-speaker review and fixes.
- A test set of 10 sample images with expected outcomes.

## Deliverables
- [ ] H+2: Stubs for `generate_alert`, `synthesize`, `verify_photo` returning contract-shaped dummy data (so backend can import)
- [ ] H+4: First prompt in `voice_vision/prompts/` and working `generate_alert(cell)` with the LLM, English first
- [ ] H+6: Tamil output working; send 10 Tamil alerts to a native speaker for review
- [ ] H+7: `synthesize(text, lang)` for English and Tamil, writing audio into the served audio folder agreed with backend
- [ ] H+9: Collect the 10-image test set (5 genuine waterlogging, 5 not) with expected outcomes in a table
- [ ] H+10: First end-to-end run (checkpoint)
- [ ] H+12: `verify_photo` v1: pretrained vision model zero-shot classification ("flooded street" vs not) + distance check + timestamp freshness check
- [ ] H+14: Real implementations delivered to backend (checkpoint)
- [ ] H+16: Apply native-speaker Tamil fixes; record them in `TAMIL_REVIEW.md`
- [ ] H+18: Run the 10-image test set, report accuracy honestly; tune threshold
- [ ] H+20: Feature freeze; cache audio for the demo locations (offline fallback)
- [ ] H+23: Demo rehearsal done

## Definition of done
- `generate_alert`, `synthesize`, `verify_photo` have exactly the signatures and return keys in contract C.
- Alerts are under ~40 words, calm, action-first, with 2–3 concrete actions; Tamil has been reviewed by a native speaker and fixes logged in `TAMIL_REVIEW.md`.
- Audio plays for both languages; failures return a clear error, not a crash.
- Vision returns `status` (pending/verified/rejected), `confidence` 0–1 and a one-line reason; all three checks (visual, location, time) influence the result.
- 10-image test set runs with one command and the real accuracy is written down.

## Out of scope (do NOT touch)
- API routes (backend/), UI (frontend/), model training (ml/), data downloads (data/).
- `docs/contract.md` — request changes from the lead.
- Training your own vision model (use pretrained only).
- Claiming flood depth from photos.

## Git branch and folder
- Branch: `voice-vision`
- Folder: `voice_vision/`

## Cut-first list (drop in this order if time runs short)
1. Photo-based confidence tuning (use a fixed threshold).
2. Location accuracy checks beyond a simple distance threshold.
3. Second TTS voice/provider (one working provider is enough).
4. Fancy prompt versions (one good prompt per language).
5. Full 10-image test set (reduce to 6).
Never cut: English + Tamil text alerts, at least one working TTS, and a basic `verify_photo`.
