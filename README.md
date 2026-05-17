# 🗓️ Perfect Saturday Planner

An AI **agent** that plans a great Saturday for you. Tell it your city, budget,
free time, mood, interests and constraints — it researches **real places** and
streams back a personalised, costed itinerary, explaining *why* each stop fits
you and showing its reasoning step by step.

> **Live app:** _<add your Streamlit Cloud URL here after deploying>_

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/ui-streamlit-ff4b4b)
![Gemini](https://img.shields.io/badge/llm-gemini-4285f4)

---

## What makes it an *agent*, not a prompt

The plan is produced by **9 discrete tools** wired into a control loop — not one
giant prompt. The orchestrator decides what runs when, asks for clarification
when needed, and **replans** when a draft fails validation.

```
parse_preferences ─→ geocode_city ─→ get_weather
   (+ guardrails)         │
              ┌───────────┴────────────┐
        get_activity_options    get_food_options      ← real OpenStreetMap data
              └───────────┬────────────┘
                  generate_final_plan  (LLM)
                          │
                  verify_grounding  ──→  estimate_cost  ──→  validate_plan
                          │                                       │
                          └──────────── replan ◄──────────────────┘
                                          │           (on budget/time violation)
                                    fallback plan  (if still no fit)
```

| # | Tool | What it does | Type |
|---|------|--------------|------|
| 1 | `parse_preferences` | Form **or** free text → structured profile; input guardrails; flags vague input | LLM |
| 2 | `geocode_city` | City → coordinates + search box | Real API (Nominatim) |
| 3 | `get_weather` | Forecast for the upcoming Saturday | Real API (Open-Meteo) |
| 4 | `get_activity_options` | Real things-to-do near the city | Real API (Overpass) |
| 5 | `get_food_options` | Real places to eat, ranked by diet constraints | Real API (Overpass) |
| 6 | `generate_final_plan` | Drafts the itinerary from real venues | LLM |
| 7 | `verify_grounding` | Confirms every venue traces back to real OSM data | Code |
| 8 | `estimate_cost` | Sums spend + computes travel from real coordinates | Code |
| 9 | `validate_plan` | Checks budget, time and constraints | Code |

## Everything is dynamic — nothing about places is hardcoded

The agent works for **any city** the user types. Real data comes from free,
keyless APIs:

- **OpenStreetMap Nominatim** — geocoding
- **OpenStreetMap Overpass** — real activity & food venues (with mirror-endpoint
  fallback for reliability). Interests are mapped to OSM tags *by the LLM*, so
  even unusual interests work — there is no fixed category list.
- **Open-Meteo** — the real forecast, used to bias indoor vs. outdoor stops.

The only thing OSM lacks is prices, so the LLM estimates *typical local spend*
per venue and the app is transparent about that in the cost breakdown.

## Requirements from the brief — and where each is met

| Requirement | Where |
|-------------|-------|
| Hosted web app, public URL | Streamlit Community Cloud |
| Accepts structured **or** free-text input | `app.py` — two input tabs |
| ≥ 3 tools, not one prompt | 9 tools in `agent/tools.py` |
| Realistic, specific plan | `generate_final_plan` over real OSM venues |
| Explains why each part fits | `why_it_fits` on every stop |
| Handles a failure case gracefully | City-not-found, no venues, weather/Overpass outage, LLM error |
| Shows an agent trace | Live streamed trace + a full trace expander |

### Bonus features included

- ✅ **Real data** instead of mocks (OSM + Open-Meteo)
- ✅ **Streaming "agent is thinking" trace** — steps appear live with timings
- ✅ **Fallback plan** when nothing fits the budget/time
- ✅ **Clarifying questions** (1–2) when the request is too vague
- ✅ **Trade-off explanations** — honest notes on compromises
- ✅ **Realistic suggestions** — a `validate → replan` loop enforces budget/time

---

## Guardrails & hallucination control

### Input guardrails
- **Length cap** — raw input over 4,000 characters is rejected: bounds cost and
  blunts large prompt-injection payloads.
- **Scope / abuse / injection check** — the parse step classifies the request.
  Abusive, unsafe, off-topic or *"ignore your instructions"*-style input is
  declined politely (`is_plannable = false`). User text is treated strictly as
  **data to extract from**, never as instructions, and only structured fields
  flow downstream — so an injection attempt has nothing to escalate into.

### Hallucination control
The dangerous failure for a planner is inventing places that don't exist.
Several layers prevent it:

1. **Closed-list selection** — the LLM may only choose venues from a fixed list
   of real OSM results, *by index*. It cannot name a venue freely.
2. **Authoritative identity** — a stop's venue name, coordinates and OSM link
   come from OpenStreetMap data, never from the LLM's text.
3. **`verify_grounding` step** — after drafting, the agent confirms every
   food/activity stop traces back to a real OSM place, computes a *grounding
   score*, and flags anything unverified — shown in the trace and the UI.
4. **Prompt hardening** — the planner is explicitly forbidden from inventing
   venue-specific facts; it may only use attributes present in the OSM tags.
5. **Verifiable provenance** — every real stop links to its OpenStreetMap page,
   so a reviewer can independently confirm it exists.
6. **Schema enforcement + clamping** — malformed or out-of-range model output
   (bad indices, negative costs) can't reach the user.

### Why not RAG?
RAG retrieves from a *static document corpus* (a vector store of pre-indexed
text). This agent does something stronger: it retrieves **live, authoritative
data** from OpenStreetMap and Open-Meteo through tools — real-time and
verifiable, where a pre-indexed corpus would be stale. RAG was **deliberately
skipped**: it would add complexity and *reduce* freshness. The grounding here is
tool-based, not document-based — which is the right choice for live place data.

---

## Run locally

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd perfect-saturday-planner

# 2. Create a virtualenv and install dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your Gemini API key (free: https://aistudio.google.com/apikey)
cp .env.example .env
#   then edit .env and set GEMINI_API_KEY=...

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`. You can also paste the key into the
sidebar instead of using `.env`.

Run the offline tests (no key/network needed):

```bash
python tests/test_offline.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → pick this repo, branch `main`, main file `app.py`.
4. **Advanced settings → Secrets** — add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Deploy. You get a public `*.streamlit.app` URL.

## Project layout

```
app.py                 Streamlit UI + live streaming trace
agent/
  orchestrator.py      the agent loop (control flow, replan, fallback)
  tools.py             the 9 tools
  llm.py               Gemini wrapper (structured output + retries)
  models.py            Pydantic models — the contract between tools
  prompts.py           all LLM prompt text
data/
  osm.py               Nominatim geocoding + Overpass venue search
  weather.py           Open-Meteo forecast
tests/test_offline.py  network-free unit tests
```

## Tech stack

Python 3.12 · Streamlit · Google Gemini (`google-genai`) · Pydantic · httpx ·
OpenStreetMap (Nominatim + Overpass) · Open-Meteo.

## How AI tools were used in the build

This project was built with **Claude Code** as a pair-programmer. I used it to
scaffold the agent/tool architecture, write the OpenStreetMap and Open-Meteo
clients, and iterate on the prompt design and the streaming-trace UI. Every
external API was tested live during the build, and I reviewed and directed all
architectural decisions (tool boundaries, the replan/fallback loop, dynamic
OSM-tag mapping) myself.
