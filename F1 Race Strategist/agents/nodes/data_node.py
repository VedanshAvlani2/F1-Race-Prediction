"""Data node: assembles the shared race state from measured data only."""

import json

import pandas as pd

from config.settings import STORE_DIR
from data.fastf1_client import load_race
from data.race_state import build_race_state


def _pit_loss_for(event: str) -> dict:
    table_file = STORE_DIR / "circuit_pit_loss.csv"
    if not table_file.exists():
        return {"pit_loss_s": None, "note": "pit loss table missing; "
                "run data.build_datasets"}
    table = pd.read_csv(table_file)
    row = table[table["event"] == event]
    if row.empty:
        med = float(table["pit_loss_s"].median())
        return {"pit_loss_s": round(med, 2), "n_stops": 0,
                "note": f"no measured stops for {event}; using median of "
                        f"{len(table)} circuits", "low_confidence": True}
    r = row.iloc[0]
    return {"pit_loss_s": float(r["pit_loss_s"]),
            "pit_loss_std": float(r["pit_loss_std"]),
            "n_stops": int(r["n_stops"]),
            "low_confidence": bool(r["low_confidence"])}


def _form_for(driver: str) -> dict:
    form_file = STORE_DIR / "driver_form.json"
    if not form_file.exists():
        return {"note": "driver form not built; run data.driver_form"}
    data = json.loads(form_file.read_text(encoding="utf-8"))
    entry = data.get("drivers", {}).get(driver)
    return {"source": data.get("source"), "season": data.get("season"),
            "metrics": entry,
            "note": data.get("note", "")}


def _championship_summary() -> dict:
    st_file = STORE_DIR / "standings.json"
    if not st_file.exists():
        return {"tracker_status": "never_run"}
    data = json.loads(st_file.read_text(encoding="utf-8"))
    return {"tracker_status": data.get("tracker_status"),
            "season": data.get("season"), "round": data.get("round"),
            "stale": data.get("stale", False),
            "top3": data.get("drivers", [])[:3]}


def data_node(state: dict) -> dict:
    session = load_race(state["year"], state["event"])
    race_state = build_race_state(
        session, state["year"], state["event"], state["lap"],
        state["focus_driver"])
    transcript = list(state.get("transcript", []))
    transcript.append({
        "node": "data",
        "summary": f"Race state assembled: lap {race_state['lap']}/"
                   f"{race_state['total_laps']}, track {race_state['track_status']}, "
                   f"{len(race_state['order'])} cars classified.",
    })
    return {
        "race_state": race_state,
        "pit_loss": _pit_loss_for(state["event"]),
        "driver_form": _form_for(state["focus_driver"]),
        "championship": _championship_summary(),
        "revision_count": 0,
        "transcript": transcript,
    }
