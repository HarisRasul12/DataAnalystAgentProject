
<img width="1536" height="1024" alt="Logo design for OceanWatch analytics" src="https://github.com/user-attachments/assets/fbe5ee3f-eb05-4c17-87c6-41cc7ef48a50" />

# OceanWatch: BlueTide Intelligence
By: **Haris Rasul**

OceanWatch is a Streamlit + Google ADK + Vertex AI multi-agent system for U.S. coastal analytics using live NOAA data (no API key required).

It is built to satisfy **Project 2: Data Analyst Agent** requirements end-to-end:
- Collect real external data at runtime
- Perform EDA with tool calls and computation
- Produce a data-grounded hypothesis with evidence

Live deployment (Cloud Run): **https://oceanwatch-q7smatrnpa-uc.a.run.app**
Last verified live revision: **oceanwatch-00013-ndb** (April 3, 2026)

> **Runtime disclaimer:** After you click **Run Multi-Agent Analysis**, the pipeline can take **~3-5 minutes** to return results depending on NOAA response time and ADK/Vertex latency.  
> The **Monte Carlo handoff** can also take **~3-5 minutes** after activation before returning simulation outputs and interpretation.

---

## 0) Product Context
OceanWatch is a coastal decision-support assistant for analysts, students, and operations teams that need fast, evidence-backed understanding of changing ocean conditions.

It is designed for questions like:
- “What is happening at this station right now?”
- “Are there irregular patterns or elevated risk signals?”
- “What might happen over the next tide cycle or next few days?”

OceanWatch is not a replacement for official forecast centers; it is an analytics layer that explains current conditions and plausible risk scenarios from live NOAA observations.

---

## 1) What This App Does
- Pulls **NDBC buoy** data at runtime (wave, wind, temperature).
- Pulls **NOAA CO-OPS** runtime data (water levels + tide predictions).
- Runs EDA + advanced analytics:
  - summary stats, trend slopes, anomaly detection
  - regression signals + exceedance probabilities
  - risk scoring + scenario outputs
- Uses **Google ADK multi-agent orchestration** for hypothesis generation.
- Adds a post-analysis **Monte Carlo handoff**:
  - deterministic simulation stats/charts first
  - ADK interpretation and post-simulation hypothesis refinement next
- Shows an executive frontend with:
  - tabbed briefing workspace (Executive Brief, Geo Lens, Quant Lab, Monte Carlo Lab, Data Room)
  - source-health badges + expanded metric cards
  - station map + **Ocean Lens port-impact intelligence map**
  - regression/simulation visuals with hover interpretation callouts
  - downloadable data exports

---

## 1.1) What OceanWatch Can Do
- **Runtime data collection** from two NOAA systems (NDBC + CO-OPS) without API keys.
- **Station-level diagnostics** on waves, wind, water temperature, water level, and tide context.
- **EDA outputs** with trends, anomaly density, distribution summaries, and regression signals.
- **Risk framing** with severity scoring and exceedance probabilities.
- **Port-impact analytics** using selected-station conditions to estimate relative U.S. port exposure scores and bands.
- **Multi-agent hypothesis generation** using Google ADK orchestration.
- **Monte Carlo follow-up** for stochastic wave-path scenarios and post-simulation hypothesis refinement.
- **User-facing explainability** via plain-English summaries, evidence bullets, caveats, and visual callouts.

---

## 1.2) How It Works (End-to-End)
1. User selects a coastal station, horizon, units, and asks a question.
2. Coordinator triggers runtime NOAA collection tools.
3. Specialist agents run EDA in parallel (buoy, tide/water level, risk/pattern analysis).
4. Hypothesis agents synthesize findings into a single, evidence-backed hypothesis.
5. Optional Monte Carlo handoff runs advanced simulation and then a post-simulation hypothesis pass.
6. UI renders an executive tabbed dashboard with cards, charts, map context, port intelligence, and downloadable artifacts.

---

## 1.5) Executive Dashboard Layout

OceanWatch is intentionally designed like an analytics command center rather than a raw notebook:
- **Executive Brief:** Final hypothesis, evidence, caveats, station map, and metadata.
- **Geo Lens:** Ocean Lens port-risk propagation map + top exposed ports + impact table.
- **Quant Lab:** Severity/risk diagnostics, regression/exceedance visuals, scenario matrix, and figure gallery.
- **Monte Carlo Lab:** Agent handoff controls plus path fan-chart, histogram, and exceedance curves.
- **Data Room:** CSV downloads and dataset previews for reproducibility and grading.

---

## 1.3) EDA Question Examples You Can Ask

### Current-state and trend questions
- “What are the strongest changing signals at this station in the last 48 hours?”
- “Is wave height trending up or down, and how quickly?”
- “How are wind and water-level trends interacting right now?”

### Pattern and anomaly questions
- “What irregular patterns are present in wave and water-level behavior?”
- “Are current observations outside typical distribution bands?”
- “Which variable is contributing most to anomaly pressure?”

### Risk and likelihood questions
- “What is the near-term chance of exceeding 2m or 3m wave thresholds?”
- “Is this station currently in a low, moderate, or high coastal stress regime?”
- “What do the regression and probability signals imply for next-tide operations?”

### Comparison and follow-up questions
- “How would this station’s risk profile differ with stronger wind forcing?”
- “What changed after Monte Carlo simulation versus the preliminary hypothesis?”
- “What are the top follow-up checks to reduce uncertainty?”

---

## 1.4) Data Collection + Retrieval Grounding (RAG-Style) In Depth

### A) Where the data comes from (external runtime sources)
OceanWatch pulls data live from NOAA on every run. It does not rely on static bundled CSVs for core analysis.

1. **NOAA NDBC buoy realtime feed** (text table)
- URL pattern: `https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt`
- Provides wave/wind/ocean-atmosphere observations like:
  - wave height (`WVHT`)
  - wind speed (`WSPD`)
  - gust (`GST`)
  - water temperature (`WTMP`)
  - pressure (`PRES`)

2. **NOAA CO-OPS DataGetter API** (JSON)
- Base URL: `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`
- Product calls:
  - `product=water_level` (observed water levels)
  - `product=predictions` (tide highs/lows)
- Typical runtime query fields include:
  - `station`, `begin_date`, `end_date`
  - `datum=MLLW`
  - `time_zone=gmt`
  - `units=metric` (water levels) / `units=english` (tide predictions)
  - `format=json`

### B) How OceanWatch pulls and transforms it
1. User picks a station and horizon in Streamlit.
2. Station IDs (NDBC + CO-OPS) are resolved from curated station catalog.
3. Service prefetches NOAA sources in parallel for speed.
4. Raw payloads are parsed into typed dataframes:
- NDBC text -> timestamped numeric frame
- CO-OPS JSON -> observed level frame + prediction frame
5. Parsed rows feed EDA tools and ADK agent workflows.

### C) Reliability behavior (runtime robustness)
- HTTP timeout + retry policy (`httpx` + `tenacity`)
- Per-source health tracking (`ok`, `empty`, `error`)
- Graceful degraded mode when one feed fails
- Explicit confidence/warning messaging in UI

### D) “RAG” in OceanWatch (what it is and isn’t)
OceanWatch uses **retrieval-grounded context packaging** for agent reasoning:
- It retrieves live NOAA observations and computed metrics first.
- It then injects this structured runtime context into ADK session state for hypothesis agents.
- Monte Carlo handoff uses retrieved simulation snapshots + recent wave context for post-simulation interpretation.

This is **RAG-style grounding over live tabular/API data**, not a document-vector-store RAG over PDFs.  
In other words: OceanWatch’s agents are grounded by retrieved operational ocean data, not by static docs.

### E) Exact code locations for data/retrieval flow
- Source collection client:
  - `src/oceanwatch/noaa_clients.py:NOAAClient.fetch_ndbc_observations`
  - `src/oceanwatch/noaa_clients.py:NOAAClient.fetch_coops_water_level`
  - `src/oceanwatch/noaa_clients.py:NOAAClient.fetch_coops_tide_predictions`
- Parsing:
  - `src/oceanwatch/noaa_clients.py:parse_ndbc_realtime_text`
  - `src/oceanwatch/noaa_clients.py:parse_coops_water_level_json`
  - `src/oceanwatch/noaa_clients.py:parse_coops_predictions_json`
- Runtime orchestration and prefetch:
  - `src/oceanwatch/service.py:OceanWatchService.run_analysis`
  - `src/oceanwatch/service.py:OceanWatchService._prefetch_sources`
- Tool-call interface used by agents:
  - `src/oceanwatch/service.py:OceanWatchToolset`
- Retrieval-grounded Monte Carlo handoff:
  - `src/oceanwatch/service.py:OceanWatchService.run_wave_monte_carlo_agent`

---

## 2) Assignment Three Steps Mapping

### Step 1: Collect
Real-world data retrieval at runtime from external sources:
- NDBC realtime feed
- NOAA CO-OPS DataGetter endpoints

Implemented in:
- `src/oceanwatch/noaa_clients.py`
- `src/oceanwatch/service.py`
  - `get_ndbc_observations`
  - `get_coops_water_level`
  - `get_coops_tide_predictions`

### Step 2: Explore / Analyze (EDA)
At least one tool call computes on collected data (in practice: several):
- statistical aggregation
- filtering/grouping
- anomaly detection
- regression and probability estimates
- scenario computation + simulation

Implemented in:
- `src/oceanwatch/metrics.py`
  - `compute_ocean_metrics`
  - `compute_wave_monte_carlo`
- `src/oceanwatch/service.py`
  - `compute_ocean_metrics`
  - `compute_risk_analytics`
  - `simulate_coastal_scenarios`
  - `build_visuals`

### Step 3: Hypothesize
Hypothesis is derived from data and presented with evidence/caveats:
- ADK synthesis outputs structured `InsightReport`
- Monte Carlo follow-up refines hypothesis narrative

Implemented in:
- `src/oceanwatch/agents.py`
  - `HypothesisBuilderAgent`
  - `HypothesisCriticRefinerAgent`
  - `WaveHypothesisAgent`
  - `PostMonteCarloHypothesisAgent`
- `src/oceanwatch/schemas.py`
  - `InsightReport`
  - `WaveHypothesisInterpretation`
  - `PostMonteCarloHypothesis`

---

## 3) Core Requirements Checklist (Required)

### Frontend
- **How OceanWatch does it:** Users pick a station + horizon, ask an ocean analytics question, and get an executive tabbed workspace with map context, health badges, metric cards, port-intelligence analytics, chart panels, and hypothesis outputs.
- **Where in code (file + function/class):**
  - `streamlit_app.py:main`
  - `streamlit_app.py:render_station_map`
  - `streamlit_app.py:render_ocean_lens`
  - `streamlit_app.py:render_metric_cards`
  - `streamlit_app.py:render_advanced_panels`
  - `streamlit_app.py:render_wave_monte_carlo_panel`
  - `streamlit_app.py:dataframe_download_button`

### Agent Framework
- **How OceanWatch does it:** ADK runs specialist ocean agents (buoy, tide, risk, pattern) in parallel, then synthesis agents produce a hypothesis. A second ADK chain handles post-Monte-Carlo interpretation.
- **Where in code (file + function/class):**
  - `src/oceanwatch/agents.py:create_oceanwatch_root_agent`
  - `src/oceanwatch/agents.py:create_wave_monte_carlo_agent`
  - `src/oceanwatch/service.py:OceanWatchService._run_adk_pipeline`
  - `src/oceanwatch/service.py:OceanWatchService.run_wave_monte_carlo_agent`
  - `src/oceanwatch/service.py:OceanWatchService._run_post_mc_hypothesis_recovery`

### Tool Calling
- **How OceanWatch does it:** Tool calls fetch live NOAA rows and compute ocean EDA features (mean/median/min/max/latest, slope/hour, z-score anomalies, regression/exceedance, severity score, scenario outputs, chart payloads).
- **Where in code (file + function/class):**
  - `src/oceanwatch/service.py:OceanWatchToolset`
    - `get_ndbc_observations`
    - `get_coops_water_level`
    - `get_coops_tide_predictions`
    - `compute_ocean_metrics`
    - `compute_risk_analytics`
    - `simulate_coastal_scenarios`
    - `build_visuals`

### Non-Trivial Runtime Dataset
- **How OceanWatch does it:** Every run retrieves live station-specific ocean observations from external NOAA systems; data changes by time and station and is large enough to require runtime retrieval + parsing.
- **Where in code (file + function/class):**
  - `src/oceanwatch/noaa_clients.py:NOAAClient.fetch_ndbc_observations`
  - `src/oceanwatch/noaa_clients.py:NOAAClient.fetch_coops_water_level`
  - `src/oceanwatch/noaa_clients.py:NOAAClient.fetch_coops_tide_predictions`
  - `src/oceanwatch/noaa_clients.py:parse_ndbc_realtime_text`
  - `src/oceanwatch/noaa_clients.py:parse_coops_water_level_json`
  - `src/oceanwatch/noaa_clients.py:parse_coops_predictions_json`

### Multi-Agent Pattern
- **How OceanWatch does it:** Uses fan-out + synthesis in the ocean domain: separate agents inspect buoy dynamics, tide/water-level context, risk likelihoods, and cross-signal patterns before final hypothesis synthesis.
- **Where in code (file + function/class):**
  - `src/oceanwatch/agents.py:create_oceanwatch_root_agent`
    - `BuoyAgent`
    - `TideAgent`
    - `RiskAgent`
    - `PatternHypothesisAgent`
    - `HypothesisBuilderAgent`
    - `HypothesisCriticRefinerAgent`
  - `src/oceanwatch/agents.py:create_wave_monte_carlo_agent`
    - `WaveSimulationAgent`
    - `WaveHypothesisAgent`
    - `PostMonteCarloHypothesisAgent`

### Deployed
- **How OceanWatch does it:** The same app (NOAA runtime pulls + ADK/Vertex pipeline) runs locally and on Cloud Run using containerized uv-managed dependencies.
- **Where in code/config (file names):**
  - `Dockerfile`
  - `cloudbuild.yaml`
  - `cloudrun.yaml`
  - `pyproject.toml` (scripts + dependency lock intent)

### README Mapping
- **How OceanWatch does it:** This README gives rubric-to-implementation evidence for Collect → EDA → Hypothesize plus required/elective concepts in ocean analytics terms.
- **Where in this file:** Sections `2`, `3`, and `4`.

---

## 4) Grab-Bag Electives Implemented

### Chosen Electives For Grading (at least two)

1) **Code execution (chosen)**
- **How OceanWatch does it in-domain:** Executes numeric ocean analytics over runtime NOAA tables: trend slopes, anomaly counts, regression projections, exceedance probabilities, severity scoring, and advanced Monte Carlo wave-path simulation (OU + stochastic vol + jump/regime dynamics).
- **Where in code (file + function/class):**
  - `src/oceanwatch/metrics.py:compute_ocean_metrics`
  - `src/oceanwatch/metrics.py:compute_wave_monte_carlo`
  - `src/oceanwatch/metrics.py:_build_regression_signal`
  - `src/oceanwatch/metrics.py:_severity_profile`
  - `src/oceanwatch/metrics.py:_irregular_pattern_records`

2) **Structured output (chosen)**
- **Status:** Fully implemented with high confidence.
- **How OceanWatch does it in-domain:** OceanWatch uses schema-constrained ADK outputs (Pydantic + ADK `output_schema`) so ocean hypotheses and Monte Carlo interpretations are emitted as reliable structured objects, then validated/parsing-checked before UI rendering.
- **What is structured:** thesis/hypothesis text, narrative paragraphs, evidence bullets, anomalies, limitations, follow-ups, Monte Carlo interpretation fields, and post-Monte-Carlo hypothesis deltas.
- **Where in code (file + function/class):**
  - `src/oceanwatch/schemas.py:InsightReport`
  - `src/oceanwatch/schemas.py:WaveHypothesisInterpretation`
  - `src/oceanwatch/schemas.py:PostMonteCarloHypothesis`
  - `src/oceanwatch/agents.py:create_oceanwatch_root_agent` (`output_schema=InsightReport`)
  - `src/oceanwatch/agents.py:create_wave_monte_carlo_agent` (`output_schema=WaveSimulationSnapshot`, `output_schema=WaveHypothesisInterpretation`, `output_schema=PostMonteCarloHypothesis`)
  - `src/oceanwatch/service.py:OceanWatchService._insight_from_state`
  - `src/oceanwatch/service.py:OceanWatchService._wave_hypothesis_interpretation_from_state`
  - `src/oceanwatch/service.py:OceanWatchService._post_monte_carlo_hypothesis_from_state`

### Additional Electives Implemented (bonus evidence)
3) **Data visualization (bonus)**
- **How OceanWatch does it in-domain:** It renders coastal analytics as runtime interactive visuals, including:
- buoy wave/wind/temp dynamics
- observed water level + predicted tide context
- distribution band charts (P25/P50/P75) for ocean variables
- regression slope-vs-exceedance risk plots
- projected-vs-threshold comparison charts
- scenario severity-vs-likelihood risk matrix
- Monte Carlo wave-path fan charts, final-state histogram, and exceedance probability curves
- Ocean Lens geospatial port-risk propagation and exposure ranking
- **Where in code (file + function/class):**
  - `src/oceanwatch/visuals.py:build_visuals`
  - `src/oceanwatch/visuals.py:_buoy_conditions_figure`
  - `src/oceanwatch/visuals.py:_water_level_figure`
  - `src/oceanwatch/visuals.py:_distribution_figure`
  - `src/oceanwatch/visuals.py:_scenario_figure`
  - `streamlit_app.py:render_ocean_lens`
  - `streamlit_app.py:render_advanced_panels`
  - `streamlit_app.py:render_wave_monte_carlo_panel`

4) **Parallel execution (bonus)**
- **How OceanWatch does it in-domain:** It parallelizes specialist ocean analysis so buoy dynamics, tide/water-level context, risk modeling, and pattern interpretation run concurrently before synthesis. This reduces latency and improves signal coverage prior to hypothesis generation.
- **Where in code (file + function/class):**
  - `src/oceanwatch/agents.py:create_oceanwatch_root_agent`
  - `src/oceanwatch/agents.py` `ParallelAgent(name="ParallelOceanCollection", sub_agents=[buoy_agent, tide_agent, risk_agent, pattern_agent])`
  - `src/oceanwatch/agents.py` specialist sub-agents: `BuoyAgent`, `TideAgent`, `RiskAgent`, `PatternHypothesisAgent`

---

## 5) Multi-Agent Architecture

### Main analysis pipeline
- `CoordinatorAgent` (sequential orchestrator)
  - `ParallelOceanCollection`
    - `BuoyAgent`
    - `TideAgent`
    - `RiskAgent`
    - `PatternHypothesisAgent`
  - `HypothesisBuilderAgent`
  - `HypothesisCriticRefinerAgent`

### Monte Carlo follow-up pipeline
- `WaveMonteCarloCoordinator`
  - `WaveSimulationAgent`
  - `WaveHypothesisAgent`
  - `PostMonteCarloHypothesisAgent`

If ADK interpretation fails in Monte Carlo, service attempts ADK recovery before marking blocked.

---

## 6) Local Run (uv) — Exact Commands

```bash
cd "/Users/harisrasul/Desktop/google-cloud-sdk/Projects/Project /IEORE4576-Project2/DataAnalystAgentProject"

uv sync --refresh

gcloud config set project ieor-4576-agents-haris
gcloud config set ai/region us-central1
gcloud config set run/region us-central1

gcloud auth application-default login
gcloud auth application-default set-quota-project ieor-4576-agents-haris

export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=ieor-4576-agents-haris
export GOOGLE_CLOUD_LOCATION=us-central1
export VERTEX_MODEL=gemini-2.0-flash-lite
export OCEANWATCH_VERTEX_MODEL_CANDIDATES="gemini-2.0-flash,gemini-2.5-flash-lite"
export OCEANWATCH_ADK_ENABLED=true
export OCEANWATCH_REQUIRE_ADK_SUCCESS=true
export OCEANWATCH_ADK_TIMEOUT_SECONDS=40

uv run serve
```

Open: `http://127.0.0.1:8000`

---

## 7) Useful Commands

### Run tests
```bash
uv run pytest tests -v
```

### Fast smoke (deterministic path)
```bash
uv run smoke
```

---

## 8) Environment Variables
- `VERTEX_MODEL` (default: `gemini-2.0-flash-lite`)
- `OCEANWATCH_VERTEX_MODEL_CANDIDATES`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION` (default: `us-central1`)
- `GOOGLE_GENAI_USE_VERTEXAI=true`
- `OCEANWATCH_ADK_ENABLED` (default: `true`)
- `OCEANWATCH_REQUIRE_ADK_SUCCESS` (default: `true`)
- `OCEANWATCH_ALLOW_TRANSIENT_FALLBACK_WHEN_STRICT` (default: `true`)
- `OCEANWATCH_ADK_TIMEOUT_SECONDS` (default: `40`)
- `OCEANWATCH_DEFAULT_HOURS` (default: `168`)

---

## 9) Cloud Run Deployment

```bash
cd "/Users/harisrasul/Desktop/google-cloud-sdk/Projects/Project /IEORE4576-Project2/DataAnalystAgentProject"

PROJECT_ID="$(gcloud config get-value project)"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user"

gcloud artifacts repositories create oceanwatch-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="OceanWatch images" || true

gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_SERVICE=oceanwatch,_REGION=us-central1,_REPO=oceanwatch-repo,_IMAGE_TAG=latest

gcloud run services describe oceanwatch \
  --region us-central1 \
  --format='value(status.url)'
```

Current deployed URL:
- `https://oceanwatch-q7smatrnpa-uc.a.run.app`
- Latest verified ready revision: `oceanwatch-00013-ndb` (April 3, 2026)

If Cloud Build succeeds but deploy fails on env-var parsing, run direct deploy:

```bash
gcloud run deploy oceanwatch \
  --image=us-central1-docker.pkg.dev/ieor-4576-agents-haris/oceanwatch-repo/oceanwatch:latest \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --timeout=900 \
  --cpu=2 \
  --memory=2Gi \
  --set-env-vars="^:^VERTEX_MODEL=gemini-2.0-flash-lite:OCEANWATCH_VERTEX_MODEL_CANDIDATES=gemini-2.0-flash,gemini-2.5-flash-lite:OCEANWATCH_REQUIRE_ADK_SUCCESS=true:OCEANWATCH_ALLOW_TRANSIENT_FALLBACK_WHEN_STRICT=true:OCEANWATCH_APP_NAME=oceanwatch:OCEANWATCH_DEFAULT_HOURS=168:OCEANWATCH_TIMEOUT_SECONDS=20:OCEANWATCH_RETRY_ATTEMPTS=3:OCEANWATCH_ADK_TIMEOUT_SECONDS=45:GOOGLE_CLOUD_LOCATION=us-central1:GOOGLE_GENAI_USE_VERTEXAI=true"
```

---

## 10) Troubleshooting (Most Common)

### “ADK blocked” or missing hypothesis
Usually one of:
- Vertex auth/ADC not set
- project/region mismatch
- transient network/rate-limit issue
- model unavailable in region

Quick checks:
```bash
gcloud auth list
gcloud auth application-default print-access-token
gcloud config get-value project
echo "$GOOGLE_CLOUD_PROJECT $GOOGLE_CLOUD_LOCATION $VERTEX_MODEL"
```

### Ocean Lens map looks flat or overly similar
Ocean Lens computes relative port exposure from selected-station severity, exceedance likelihoods, anomaly density, and geospatial distance weighting.  
Try changing station/horizon or rerunning analysis to refresh risk gradients.

---

## 11) Submission Note
OceanWatch is designed to be rubric-verifiable: all required concepts and elective concepts are explicitly mapped above to concrete files/functions for grading.
