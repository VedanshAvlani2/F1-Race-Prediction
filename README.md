# 🏎️ F1 Race Strategist: AI Team General Manager

A real-time F1 race strategy system built on multi-agent deliberation. Deterministic models (tire degradation regression, undercut math, safety car probability, measured pit loss) compute every number. LLM agents on Groq (llama-3.3-70b-versatile) debate those numbers, disagree in the open, and converge on one recommendation with a confidence level and a full transcript.

This is the second generation of this repo. The original lap time predictor it grew out of is documented at the bottom.

## ✨ Features

- **Live Strategy tab**: pick any cached race, lap, and driver. The agent panel (strategist, risk, skeptic, synthesis) proposes a pit window, stress tests it, audits it against raw undercut math, and shows every disagreement on record.
- **Pre-race planner**: pick a circuit, forecast track temperature, and a starting grid (predicted from rolling driver form or drag-and-drop your own). Enumerates every legal 1 and 2 stop plan, prices each with the degradation model plus measured pit loss, runs 800 Monte Carlo safety car simulations, and writes a pit-wall brief with Plan A, fallback Plan B, and switch triggers.
- **Race replay**: real telemetry replay of any race from 2022 to today. True circuit shapes drawn from position data, team-colored driver dots where spacing equals the actual time gap, a real pit lane path, live flags (yellow, SC, VSC, red), and live weather (track temp, air, humidity, wind, rain).
- **Championship tab**: live driver and constructor standings with a disk-cached tracker that no-ops when no new race has happened.
- **Models tab**: honest metrics for every model, including validation methodology and low-confidence flags.

## 🧠 Architecture

```
FastF1 cache (2022-2026, 100+ races)
        |
data/    stint extraction, SC/VSC events, measured pit loss,
         live championship tracker, driver form, race replay builder
        |
models/  tire degradation (GBR), safety car (logistic),
         undercut calculator (pure math), regulation engine,
         pre-race strategy planner (Monte Carlo)
        |
agents/  LangGraph: data -> strategist -> risk -> skeptic
         (disagree? one revision loop) -> synthesis
         LLM: Groq llama-3.3-70b-versatile
        |
api/     FastAPI          ui/  React (Vite) dashboard
```

Design rules: every number comes from a deterministic model or real measured data. The LLM explains and reasons over those numbers; it never generates lap times, degradation rates, or probabilities. Validation is out-of-race (GroupKFold by race, leave-one-out), baselines are reported next to model scores, and weak results are flagged rather than hidden.

## 🚀 How to Run (New Project)

Prerequisites: Python 3.10+, Node 18+, and a free Groq API key from https://console.groq.com/keys

```bash
cd "F1 Race Strategist"
pip install -r requirements.txt
```

Create a `.env` file in the `F1 Race Strategist` folder:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Build the datasets and train the models (first run downloads race data via FastF1, is resumable, and respects the 500 calls/hour API limit; rerun the fetch command after an hour if it stops):

```bash
python -m data.fetch_seasons 2022 2023 2024 2025 2026
python -m data.build_datasets
python -m models.train
```

Start the backend (terminal 1):

```bash
uvicorn api.server:app --port 8000
```

Start the dashboard (terminal 2):

```bash
cd ui
npm install
npm run dev
```

Open http://localhost:5173 and use the tabs: Strategy, Pre-race, Replay, Championship, Models.

## 📊 Model Report (honest numbers)

- **Tire degradation**: GradientBoostingRegressor on 85,000+ clean dry green-flag laps across 99 races. GroupKFold(5) by race: MAE 0.575s against a 0.797s predict-the-mean baseline, R2 0.43. Out-of-race error, no same-race leakage.
- **Compound rates (fuel separated)**: SOFT 0.037, HARD 0.027, MEDIUM 0.022 s/lap.
- **Safety car**: logistic regression on smoothed circuit SC rate plus street flag, 101 races, leave-one-out validated. It does not yet beat the predict-the-base-rate baseline (Brier 0.257 vs 0.247) because safety cars are largely random at 1 to 4 races per circuit. The system flags it low confidence everywhere it is consumed. That is a real finding, not a bug.
- **Pit loss**: median of (in lap + out lap minus 2x driver median green lap) over 2,400+ real green-flag stops, 26 circuits, sample sizes reported.
- **Undercut calculator and regulation engine**: deterministic, no trained parameters, assumptions attached to every output.

## ⚠️ Stated Limitations

- No traffic or track position model; plan times are clean-air optima.
- Wet degradation is out of scope; the model trains on dry laps only.
- The regulation engine covers the strategy-relevant subset of the FIA Sporting Regulations (two compound rule, pit window bounds, late stop warning), not the full document.
- The predicted grid is a rolling form ranking, not a qualifying simulation, and says so in the UI.

---
---

# 📜 Legacy: Predicting Formula 1 Race Times with Machine Learning

The original project this repo started as. It predicts F1 race lap times using driver sector performance, qualifying sessions, team standings, and weather conditions via a Gradient Boosting Regressor, served through a Streamlit dashboard.

**Key inputs**: driver sector time averages, user-input qualifying times, normalized constructor points, and rain conditions adjusted with driver-specific wet-weather factors.

**Reported metrics**: MAE 0.289 seconds on an 80/20 same-race split. Top features: Sector3Time, Sector2Time, TeamPerformanceDelta.

## How to Run (Legacy Code)

Option 1, script mode:

```bash
pip install fastf1 pandas numpy matplotlib scikit-learn
python "F1 Race Prediction.py"
```

You will be prompted to choose the race round (2024 season), enter driver qualifying times, and select a rain condition.

Option 2, Streamlit dashboard:

```bash
pip install fastf1 pandas numpy matplotlib scikit-learn streamlit
cd f1-predictive-dashboard
streamlit run app.py
```

Select any completed 2024 race, enter qualifying times, choose weather, and view predicted results, model accuracy, and feature importance live.

Note: on first run FastF1 builds a local cache (`f1_cache/`). Both legacy modes train the model per run on single-race laps; the new project above replaces this with pre-trained, cross-race validated models.
