# F1 Strategy Engine: AI Team General Manager

Real-time F1 race strategy through multi-agent deliberation. Deterministic
models (tire degradation regression, undercut math, safety car probability,
measured pit loss) compute every number; LLM agents on Groq debate those
numbers, disagree in the open, and converge on one recommendation with a
confidence level and a full transcript.

Extends the F1-Race-Prediction repo: reuses its 3.6 GB FastF1 cache
(2022-2024) as training data. The old Streamlit lap time predictor remains
untouched in the parent folder.

## Architecture

```
FastF1 cache (2022-2024, 32 races)
        |
data/   stint extraction, SC/VSC events, measured pit loss,
        live championship tracker, driver form
        |
models/ tire degradation (GBR), safety car (logistic),
        undercut calculator (pure math), regulation engine (rules)
        |
agents/ LangGraph:  data -> strategist -> risk -> skeptic --(disagree? one
        revision loop)--> synthesis        LLM: Groq llama-3.3-70b-versatile
        |
api/    FastAPI                ui/  React dashboard
```

Personas: strategist = race strategist, risk = team principal,
skeptic = data scientist, synthesis = chief strategist.

## Hard design rules

1. No Anthropic/Claude APIs anywhere. Runtime LLM calls go to Groq only
   (model set in .env, verified available on Groq's production tier).
2. Every agent is grounded in a deterministic model output computed before
   the LLM call. The LLM explains numbers; it never generates lap times,
   degradation rates or probabilities. With no GROQ_API_KEY the system
   still runs and returns the full numeric analysis with a labelled
   fallback narrative.
3. Metrics are honest. Validation is out-of-race (GroupKFold by race or
   leave-one-out). Baselines are reported next to model scores. Weak
   results are flagged, not hidden (see Model report).

## Setup

```bash
cd strategy-engine
pip install -r requirements.txt
cp .env.example .env        # add your free Groq key

python -m data.build_datasets   # ~5 min first run, resumable
python -m models.train          # trains + writes honest metrics

uvicorn api.server:app --port 8000
cd ui && npm install && npm run dev   # http://localhost:5173
```

## Model report (real numbers from this training run)

Tire degradation, GradientBoostingRegressor. 27,566 clean dry green flag
laps from 31 races. Target: lap time delta vs the driver's median clean
race pace. GroupKFold(5) grouped by race: MAE 0.61 s (+-0.06) against a
predict-the-mean baseline of 0.82 s, R2 0.39. Out-of-race error, so no
same-race leakage. Traffic remains unmodelled noise.

Compound rates (fuel separated, model partial derivative): SOFT 0.040
s/lap, MEDIUM 0.015 s/lap, HARD 0.023 s/lap. Raw stint slopes are also
reported in artifacts and are lower because fuel burn masks degradation;
that confound is documented rather than silently absorbed.

Safety car, logistic regression on smoothed circuit SC rate + street
flag. 32 races, 40.6% SC rate. Leave-one-out Brier 0.264 vs 0.241 for
always predicting the base rate: the model does NOT yet beat the trivial
baseline and is flagged low_confidence everywhere it is consumed,
including in the UI. It is best read as a smoothed circuit tendency.
More cached seasons would tighten it.

Pit loss: median of (in lap + out lap - 2 x driver median green lap) over
788 green flag stops, 24 circuits, sample sizes in the table. Circuits
under 15 stops carry a low confidence flag.

Undercut calculator and regulation engine are deterministic and have no
trained parameters. Assumptions (linear degradation over horizon, no
traffic model) are attached to every output.

## FastF1 endpoints used

| Data | Source | Notes |
|---|---|---|
| Laps, stints, compounds, tire age | livetiming.formula1.com static archive via session.load(laps=True) | cached to disk forever |
| Weather per session | same, weather=True | TrackTemp, Rainfall, Humidity |
| Track status (SC/VSC/red) | same, session.track_status | codes 1,2,4,5,6,7 |
| Race control messages | same, messages=True | |
| Schedules | github f1schedule + ergast fallback | |
| Results, standings (live tracker) | jolpica Ergast mirror via fastf1.ergast | 4 req/s, 500 req/h; tracker makes 2 calls per run |

fastf1.Cache.enable_cache() persists every response, so no endpoint is
hit twice for the same session. The championship tracker no-ops when no
new race has happened and serves the last cached pull (marked stale) when
offline.

## Honesty caveats, stated once and plainly

The SC model sample (32 races) is small; its output is a tendency.
Wet degradation is out of scope; the deg model trains on dry laps only.
Driver form falls back to the newest cached season when live results are
unreachable and labels itself accordingly. The regulation engine covers
the strategy-relevant subset of the FIA Sporting Regulations (two
compound rule, pit window bounds, late stop warning), not the full
document. One cached 2022 race has incomplete data and is excluded, and
that exclusion is recorded in data/store/dataset_manifest.json.

## Repo layout

```
config/    settings.py, circuits.yaml
data/      fastf1_client, stints, track_status, pit_loss,
           championship (live tracker), driver_form, race_state,
           build_datasets, store/ (generated CSV/JSON)
models/    tire_degradation, safety_car, undercut, regulations,
           train, artifacts/ (joblib + metrics.json)
agents/    state, llm (Groq), graph, nodes/{data,strategist,risk,
           skeptic,synthesis}
api/       server.py (FastAPI)
ui/        React + Vite dashboard
```
