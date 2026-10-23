# AI_CONTEXT — Frontend

**Owner:** [MEMBER 4 NAME]

You are helping me build one lane of a hackathon project. Read this fully before answering.

## Project summary
VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai. Tagline: "Know your street before the water does." We combine an hourly rainfall forecast (Open-Meteo) with terrain features and past Chennai flood hotspots (2015, 2023) to rank flood risk per ~250 m grid cell for the next 24–72 hours. An XGBoost risk engine produces a score and band (low/medium/high). An LLM writes short Tamil and English action messages and text-to-speech reads them aloud. Residents upload waterlogging photos, verified by a pretrained vision model; verified reports refine the hotspot map. We claim hotspot RANKING, never exact flood depth (never show depth in the UI). 24-hour sprint, 5-person team.

## My lane and folder
I am the **Frontend** owner. My folder is `frontend/` (branch `frontend`). Other lanes (ml/, data/, backend/, voice_vision/) are not mine.

## Contract
`docs/contract.md`, section B (REST API) and section A (conventions: lat/lon order, ISO timestamps with +05:30, `ta`/`en`, bands low <0.33 / medium / high >=0.66). Follow it exactly. Never change the contract; ask me to request a change from the lead.

## Rules for the AI
- Only create or edit files inside `frontend/`. Never edit `docs/contract.md`.
- Prefer simple, readable code that works over clever code.
- Small commits; suggest a message after each piece.
- No secrets or API keys in code. Use a `.env` file and `.env.example` (e.g. `VITE_API_BASE_URL`).
- Write a runnable example or test for every function/component (Vitest + React Testing Library, or a manual test checklist when faster).
- If blocked on another module, create a mock that matches the contract (fixtures in `frontend/src/mocks/`).
- State assumptions explicitly.
- Keep answers short.

## Tech stack
React 18 + Vite, Leaflet via react-leaflet, plain `fetch`, CSS (no heavy UI frameworks), optional Tailwind. Tamil text needs a font that supports it (Noto Sans Tamil). Mobile-first. Deploy to Netlify/Vercel/GitHub Pages.

## Exact deliverables
1. Leaflet map of Chennai; cells coloured green/amber/red by `risk_band`; legend.
2. Search a location or "use my location".
3. Tap a cell -> alert card from `GET /api/alert`: band, peak window, actions, EN/TA toggle, Listen button playing `audio_url` (hide/disable if null).
4. Photo report screen: camera/file input, attaches lat/lon/timestamp, `POST /api/report` (multipart), shows pending/verified/rejected + reason.
5. Verified reports from `GET /api/reports` as map markers.
6. API base URL from env variable `VITE_API_BASE_URL`; a mock switch `VITE_USE_MOCK`.
7. Clean, minimal UI readable at demo distance; mobile-first.

## Acceptance checks to help me pass
- Works against mock fixtures and the real backend by changing one env value.
- Every response field used exists in contract B; null `audio_url` handled.
- Usable at 360 px width; text readable from a distance; Tamil renders correctly.
- No console errors in the happy path; loading and error states exist.
- No flood depth wording anywhere.

## Starter prompts
1. "Scaffold a Vite + React app in frontend/ with react-leaflet centred on Chennai, an API client using VITE_API_BASE_URL, and mock fixtures matching contract section B."
2. "Render the `cells` from GET /api/risk as colour-coded square polygons (green/amber/red by risk_band) with a legend, using lat/lon as centre and a ~250 m half-size."
3. "Build an AlertCard component: band badge, peak window in local time, action list, EN/TA toggle, and a Listen button that plays audio_url and is disabled when null."
4. "Build a ReportForm: camera/file input with preview, grab navigator.geolocation and current ISO timestamp with +05:30, POST multipart to /api/report, and show status + reason."
5. "Add 'use my location' and a Nominatim search box that recentres the map and fetches risk for that point."
6. "Make the layout mobile-first with large fonts for demo distance, plus a deploy guide for Netlify with the env variable."
