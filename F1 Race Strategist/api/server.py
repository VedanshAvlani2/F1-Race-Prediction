"""FastAPI backend serving the React dashboard.

Run:  uvicorn api.server:app --reload --port 8000
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import ARTIFACTS_DIR, STORE_DIR
from agents.graph import run_deliberation
from data.championship import get_history, update_standings
from data.driver_form import update_driver_form
from data.fastf1_client import list_cached_races, load_race
from data.race_state import build_race_state

app = FastAPI(title="F1 Strategy Engine")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)

_session_cache: dict = {}


def _get_session(year: int, event: str):
    key = (year, event)
    if key not in _session_cache:
        if len(_session_cache) > 3:  # keep memory bounded
            _session_cache.pop(next(iter(_session_cache)))
        _session_cache[key] = load_race(year, event)
    return _session_cache[key]


class DeliberateRequest(BaseModel):
    year: int
    event: str
    lap: int
    driver: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions")
def sessions():
    races = list_cached_races()
    return races.to_dict(orient="records")


@app.get("/api/session-info")
def session_info(year: int, event: str):
    try:
        session = _get_session(year, event)
    except Exception as exc:
        raise HTTPException(404, f"Session not available: {exc}")
    laps = session.laps
    drivers = sorted(laps["Driver"].dropna().unique().tolist())
    return {"year": year, "event": event,
            "total_laps": int(laps["LapNumber"].max()),
            "drivers": drivers}


@app.get("/api/race-state")
def race_state(year: int, event: str, lap: int, driver: str):
    try:
        session = _get_session(year, event)
    except Exception as exc:
        raise HTTPException(404, f"Session not available: {exc}")
    return build_race_state(session, year, event, lap, driver)


@app.post("/api/deliberate")
def deliberate(req: DeliberateRequest):
    try:
        return run_deliberation(req.year, req.event, req.lap, req.driver)
    except Exception as exc:
        raise HTTPException(500, f"Deliberation failed: {exc}")


@app.get("/api/standings")
def standings():
    return update_standings()


@app.get("/api/standings/history")
def standings_history():
    return get_history()


@app.get("/api/driver-form")
def driver_form():
    return update_driver_form()


class PreRaceRequest(BaseModel):
    event: str
    track_temp: float
    total_laps: int
    focus_driver: str = ""
    grid: list = []


@app.get("/api/pre-race/grid")
def pre_race_grid():
    """The current lineup: exactly the drivers who started the most recent
    cached race, ordered by rolling form where available. Form entries for
    drivers no longer racing (substitutes, previous seasons) are dropped,
    never merged in."""
    from models.strategy_planner import predicted_grid
    # Refresh rolling form from the live current season first; falls back
    # to the last good pull on its own when offline.
    try:
        update_driver_form()
    except Exception:
        pass
    form = predicted_grid()
    form_map = {g["code"]: g for g in form.get("grid", [])}
    try:
        races = list_cached_races()
        last = races.sort_values(["year", "date"]).iloc[-1]
        session = _get_session(int(last["year"]), last["event"])
        entries = []
        for _, r in session.results.iterrows():
            code = r["Abbreviation"]
            color = str(r.get("TeamColor") or "888888").lstrip("#")
            f = form_map.get(code, {})
            entries.append({
                "code": code,
                "driver": r.get("FullName", code),
                "team": r.get("TeamName", ""),
                "color": f"#{color}" if len(color) == 6 else "#888888",
                "avg_finish": f.get("avg_finish"),
                "_fallback": float(r["Position"]) if pd.notna(r.get("Position")) else 99.0,
            })
        entries.sort(key=lambda e: e["avg_finish"] if e["avg_finish"]
                     is not None else 50.0 + e["_fallback"])
        for i, e in enumerate(entries):
            e["pos"] = i + 1
            e.pop("_fallback", None)
        return {
            "source": form.get("source"),
            "season": form.get("season"),
            "lineup_from": f"{last['event']} {int(last['year'])}",
            "note": f"Lineup: the {len(entries)} drivers from "
                    f"{last['event']} {int(last['year'])}, ordered by "
                    "rolling average finish (last 5 races). A form ranking, "
                    "not a qualifying simulation.",
            "grid": entries,
        }
    except Exception:
        # Offline fallback: form list alone, unmerged.
        return form


@app.get("/api/pre-race/circuits")
def pre_race_circuits():
    races = list_cached_races()
    latest = races.sort_values(["event", "year"]).groupby("event").tail(1)
    out = []
    for _, r in latest.iterrows():
        out.append({"event": r["event"], "year": int(r["year"])})
    return sorted(out, key=lambda x: x["event"])


@app.post("/api/pre-race/plan")
def pre_race_plan(req: PreRaceRequest):
    from agents.pre_race import run_pre_race
    try:
        return run_pre_race(req.event, req.track_temp, req.total_laps,
                            req.focus_driver, req.grid)
    except Exception as exc:
        raise HTTPException(500, f"Pre-race planning failed: {exc}")


@app.get("/api/replay")
def replay(year: int, event: str, force: bool = False):
    from data.replay import build_replay
    try:
        return build_replay(year, event, force=force)
    except Exception as exc:
        raise HTTPException(500, f"Replay build failed: {exc}")


@app.get("/api/metrics")
def metrics():
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    if not metrics_file.exists():
        raise HTTPException(404, "Models not trained yet. "
                                 "Run: python -m models.train")
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    pit_file = STORE_DIR / "circuit_pit_loss.csv"
    if pit_file.exists():
        data["pit_loss"]["rows"] = pd.read_csv(pit_file).to_dict(orient="records")
    return data
