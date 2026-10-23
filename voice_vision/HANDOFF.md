# HANDOFF — Voice, language and vision

**Owner:** [MEMBER 5 NAME]

## What you need from others
| From | What | Contract | By |
|------|------|----------|----|
| Lead ([LEAD NAME]) | Frozen contract | C | H+2 |
| Lead (ml/) | Sample `cells` objects (with `drivers`, `peak_window`, `risk_band`) as realistic alert input | B, C | H+6 |
| Backend ([MEMBER 3 NAME]) | Where audio files are served from (folder + URL pattern) and how uploaded photos reach `verify_photo` (bytes) | B, C | H+4 |
| Data ([MEMBER 2 NAME]) | Nothing blocking; optionally a handful of hotspot lat/lon for building test cases | D | H+8 |
| Native Tamil speaker (team's contact) | Review of 10 Tamil alerts | — | H+6 sent, H+16 back |

## What others need from you
| To | What | By |
|----|------|----|
| Backend | Stub `alerts.py`, `tts.py`, `vision.py` with exact signatures/returns | H+2 |
| Backend | Real `generate_alert` and `synthesize` | H+14 |
| Backend | Real `verify_photo` | H+16 |
| Frontend ([MEMBER 4 NAME]) | Sample Tamil strings for font testing | H+6 |
| Lead | Vision test-set results (accuracy) for slides | H+18 |
| Backend | Pre-generated audio for demo locations as a fallback | H+20 |

## Mocking plan if blocked
- No LLM key or quota: `generate_alert` falls back to templates in `voice_vision/prompts/` (a fixed EN/TA message per risk band with 3 actions).
- No TTS service: return `None`-equivalent path as agreed with backend so `audio_url` is null; prerecord audio for the demo cells as a fallback.
- Vision model too slow/large: use a smaller pretrained CLIP-style model; as a last resort do the location + timestamp checks only and return `pending` with a reason.
- Upstream cell data missing: hard-code three example `cells` dicts in tests under `voice_vision/`.

## Integration checkpoints that involve you
- **H+2** contract frozen — signatures locked.
- **H+10** first end-to-end run — stubs must work through the backend.
- **H+14** real model wired in — your alerts and TTS are live in the API.
- **H+20** feature freeze — audio cached, test results written.
- **H+23** demo rehearsal done — know the demo alert, audio and photo cases.
