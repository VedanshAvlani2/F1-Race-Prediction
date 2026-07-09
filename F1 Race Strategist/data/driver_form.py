"""Driver form tracker.

Rolling metrics over each driver's recent races, computed from official
race results. Live path pulls current season results via the Ergast
mirror; offline fallback computes the same metrics from the newest cached
season so the pipeline never fabricates numbers. The source is always
reported alongside the metrics.
"""

import json
from datetime import datetime, timezone

import pandas as pd

from config.settings import CURRENT_SEASON, STORE_DIR, TRAIN_SEASONS

FORM_FILE = STORE_DIR / "driver_form.json"
WINDOW = 5


def _form_from_results(results: pd.DataFrame, source: str, season: int) -> dict:
    """results columns required: driver, code, round, grid, finish, points, dnf."""
    out = {}
    for code, d in results.groupby("code"):
        d = d.sort_values("round").tail(WINDOW)
        n = len(d)
        if n == 0:
            continue
        gained = (d["grid"] - d["finish"]).mean()
        out[str(code)] = {
            "driver": d["driver"].iloc[-1],
            "races_counted": int(n),
            "avg_finish": round(float(d["finish"].mean()), 2),
            "avg_positions_gained": round(float(gained), 2),
            "points_last_n": float(d["points"].sum()),
            "dnf_rate": round(float(d["dnf"].mean()), 2),
        }
    return {
        "source": source,
        "season": season,
        "window": WINDOW,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "drivers": out,
    }


def _fetch_live_results(season: int) -> pd.DataFrame:
    from fastf1.ergast import Ergast

    ergast = Ergast()
    resp = ergast.get_race_results(season=season)
    rows = []
    for desc_row, content in zip(resp.description.itertuples(), resp.content):
        rnd = int(desc_row.round)
        for _, r in content.iterrows():
            status = str(r.get("status", ""))
            finished = status.startswith("Finished") or "Lap" in status
            rows.append({
                "driver": f"{r['givenName']} {r['familyName']}",
                "code": r.get("driverCode") or str(r.get("driverId", ""))[:3].upper(),
                "round": rnd,
                "grid": float(r["grid"]),
                "finish": float(r["position"]),
                "points": float(r["points"]),
                "dnf": 0.0 if finished else 1.0,
            })
    return pd.DataFrame(rows)


def _results_from_cache() -> tuple:
    """Offline fallback: race results from the newest fully cached season."""
    from data.fastf1_client import list_cached_races, load_race

    races = list_cached_races()
    if races.empty:
        return pd.DataFrame(), 0
    season = int(races["year"].max())
    season_races = races[races["year"] == season].sort_values("date")
    rows = []
    for i, (_, race) in enumerate(season_races.iterrows(), start=1):
        try:
            session = load_race(season, race["event"], weather=False, messages=False)
        except Exception:
            continue
        res = session.results
        if res is None or res.empty:
            continue
        for _, r in res.iterrows():
            status = str(r.get("Status", ""))
            finished = status.startswith("Finished") or "Lap" in status
            if pd.isna(r.get("Position")):
                continue
            rows.append({
                "driver": r.get("FullName", r.get("Abbreviation", "")),
                "code": r.get("Abbreviation", ""),
                "round": i,
                "grid": float(r.get("GridPosition", 0)),
                "finish": float(r["Position"]),
                "points": float(r.get("Points", 0)),
                "dnf": 0.0 if finished else 1.0,
            })
    return pd.DataFrame(rows), season


def update_driver_form(season: int = CURRENT_SEASON) -> dict:
    try:
        results = _fetch_live_results(season)
        if results.empty:
            raise RuntimeError("empty results")
        form = _form_from_results(results, source="ergast_live", season=season)
    except Exception:
        cached = None
        if FORM_FILE.exists():
            try:
                cached = json.loads(FORM_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached = None
        if cached and cached.get("source") == "ergast_live":
            cached["note"] = "Live fetch failed. Serving previous live pull."
            return cached
        results, cache_season = _results_from_cache()
        if results.empty:
            return {"source": "unavailable", "drivers": {},
                    "note": "No live connection and no cached results."}
        form = _form_from_results(
            results, source="cached_season_fallback", season=cache_season)
        form["note"] = ("Computed from newest cached season because live "
                        "results were unreachable. Not current form.")
    FORM_FILE.write_text(json.dumps(form, indent=2), encoding="utf-8")
    return form
