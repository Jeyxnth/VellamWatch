# HANDOFF — Frontend

**Owner:** [MEMBER 4 NAME]

## What you need from others
| From | What | Contract | By |
|------|------|----------|----|
| Lead ([LEAD NAME]) | Frozen contract | B | H+2 |
| Backend ([MEMBER 3 NAME]) | Local mock API running (`USE_MOCK=1`) + README with run steps | B | H+1 / H+2 |
| Backend | Public deployed API URL and allowed CORS origin confirmation | B | H+8 |
| Backend | Real-mode API (real risk, alerts, audio, verification) | B | H+14 |
| Data ([MEMBER 2 NAME]) | Confirmed bounding box and map centre | A | H+2 |
| Voice/vision ([MEMBER 5 NAME]) | Example Tamil strings, so you can test font rendering | B | H+6 |

## What others need from you
| To | What | By |
|----|------|----|
| Backend | Your dev origin(s) and deployed origin for CORS | H+2 / H+19 |
| Backend | Bug reports where responses differ from the contract | ongoing |
| Lead | Screenshot or screen recording of the working UI for slides | H+20 |
| Everyone | Deployed frontend URL | H+19 |
| Whole team | Demo flow script (which location to tap, which photo to upload) | H+22 |

## Mocking plan if blocked
- Keep `frontend/src/mocks/` with JSON fixtures that follow contract B exactly (risk cells around Chennai, an alert in EN and TA, report responses in all three statuses).
- A `VITE_USE_MOCK=1` switch makes the API client return fixtures without any network call.
- If audio is missing, hide/disable the Listen button when `audio_url` is `null`.
- If geolocation is denied, fall back to the default Chennai centre.

## Integration checkpoints that involve you
- **H+2** contract frozen — you confirm the shapes are workable for UI.
- **H+10** first end-to-end run — UI talking to backend (mock or deployed).
- **H+14** real model wired in — verify the map and alerts look right with real scores.
- **H+20** feature freeze — deployed final build.
- **H+23** demo rehearsal done — phone and projector tested.
