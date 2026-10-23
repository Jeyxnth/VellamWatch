# ROLE — Frontend

**Owner:** [MEMBER 4 NAME]

## Mission
Build the mobile-first React + Leaflet web app that makes VellamWatch understandable in ten seconds: a Chennai map coloured by flood risk, a plain alert card in English/Tamil with a Listen button, and a photo-report flow. It must be readable at demo distance and work from day one against mocks.

## What you own
- Everything in `frontend/`.
- Map of Chennai coloured by risk band (green = low, amber = medium, red = high) with a legend.
- Location search and "use my location".
- Cell tap -> alert card with English/Tamil toggle and Listen button playing `audio_url`.
- Photo report screen (camera/file input) showing verification status.
- Verified report markers on the map.
- API base URL read from an env variable.

## Deliverables
- [ ] H+1: React app scaffold (Vite) running with Leaflet map centred on Chennai; API client reading `VITE_API_BASE_URL`; local JSON mocks matching contract B
- [ ] H+4: Risk layer drawn from `GET /api/risk` (cells coloured by band) + legend
- [ ] H+6: "Use my location" and location search (e.g. Nominatim) recentring the map
- [ ] H+8: Alert card on cell tap from `GET /api/alert`: band badge, peak window, actions, EN/TA toggle
- [ ] H+10: First end-to-end run (checkpoint) against backend mock/deployed URL
- [ ] H+11: Listen button playing `audio_url` (handle `null` gracefully)
- [ ] H+13: Photo report screen: camera/file input, geolocation + timestamp attached, `POST /api/report`, shows pending/verified/rejected with reason
- [ ] H+15: Verified-report markers from `GET /api/reports`
- [ ] H+17: Mobile polish, large readable fonts, loading and error states, Tamil font check
- [ ] H+19: Deploy to free-tier static hosting (Netlify/Vercel/GitHub Pages)
- [ ] H+20: Feature freeze
- [ ] H+22: Demo script run-through on a phone and on the projector
- [ ] H+23: Demo rehearsal done

## Definition of done
- Opening the deployed URL on a phone shows the coloured map within 3 seconds and a legend.
- Tapping any cell shows the alert card; the EN/TA toggle switches text; Listen plays audio or shows a friendly "audio unavailable".
- A photo report can be submitted and shows its status; verified reports show as markers.
- No hard-coded API URL; works against mocks and the real backend by changing one env variable.
- Never displays flood depth — only "risk" and band.

## Out of scope (do NOT touch)
- Backend routes (backend/), model code (ml/), data processing (data/), alert text/TTS/vision (voice_vision/).
- `docs/contract.md` — request changes from the lead.
- Heavy UI frameworks or custom design systems.

## Git branch and folder
- Branch: `frontend`
- Folder: `frontend/`

## Cut-first list (drop in this order if time runs short)
1. Location search box (keep "use my location" and map pan).
2. Verified-report markers (keep the upload flow).
3. Dark mode / animations / PWA install.
4. Time-horizon slider (fixed 24 h).
5. Tamil toggle polish (keep the toggle; skip transitions).
Never cut: map with bands, legend, alert card, Listen button.
