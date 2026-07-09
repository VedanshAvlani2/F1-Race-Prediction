"""Strategist agent (persona: race strategist).

Deterministic core: scans every candidate pit lap in the next 12 laps.
For each candidate, total projected cost over a fixed horizon =
  degradation losses accrued while staying out on the current tire
  + pit loss
  + degradation on the fresh tire afterwards.
The proposed window is every candidate within 0.5 s of the best. The
Groq call then explains the choice; it cannot change the numbers.
"""

import json

from agents.llm import narrate
from models.tire_degradation import DegradationModel

SCAN_LAPS = 12
HORIZON = 20


def _next_compound(compounds_used: list, current: str) -> str:
    # Prefer a compound that satisfies the two compound rule and is not
    # the softest option late in a stint war. Simple deterministic pick:
    # switch to HARD if unused, else MEDIUM, else SOFT.
    for c in ("HARD", "MEDIUM", "SOFT"):
        if c != current:
            return c
    return "MEDIUM"


def _scan_pit_laps(deg: DegradationModel, focus: dict, race_state: dict,
                   pit_loss_s: float) -> dict:
    compound = focus["compound"]
    age = focus["tyre_age"] or 0.0
    lap_now = race_state["lap"]
    temp = race_state["weather"].get("track_temp", 30.0)
    total = race_state["total_laps"]
    new_comp = _next_compound(race_state["compounds_used"], compound)

    horizon_end = min(lap_now + HORIZON, total)
    candidates = {}
    for pit_lap in range(lap_now + 1,
                         min(lap_now + 1 + SCAN_LAPS, total - 1) + 1):
        cost = 0.0
        # Stay out on current rubber until pit_lap.
        for l in range(lap_now + 1, pit_lap + 1):
            a = age + (l - lap_now)
            cost += max(deg.predict_delta(a, compound, temp, l), 0.0)
        cost += pit_loss_s
        # Fresh tire from pit_lap+1 to horizon end.
        for l in range(pit_lap + 1, horizon_end + 1):
            a = l - pit_lap
            cost += max(deg.predict_delta(a, new_comp, temp, l), 0.0)
        candidates[pit_lap] = round(cost, 3)

    if not candidates:
        return {"error": "no legal pit laps remain in scan range"}

    best_lap = min(candidates, key=candidates.get)
    best_cost = candidates[best_lap]
    window = [l for l, c in candidates.items() if c - best_cost <= 0.5]
    return {
        "candidate_costs_s": candidates,
        "best_pit_lap": best_lap,
        "window_earliest": min(window),
        "window_latest": max(window),
        "new_compound": new_comp,
        "current_compound": compound,
        "current_tyre_age": age,
        "deg_rate_now_s_per_lap": round(
            deg.deg_rate(age, compound, temp, lap_now), 4),
        "pit_loss_s": pit_loss_s,
        "horizon_end_lap": horizon_end,
    }


def strategist_node(state: dict) -> dict:
    race_state = state["race_state"]
    focus = race_state.get("focus")
    transcript = list(state.get("transcript", []))

    if focus is None or focus.get("tyre_age") is None:
        out = {"role": "strategist", "numbers": {},
               "narrative": f"No lap data for {race_state['focus_driver']} "
                            f"on lap {race_state['lap']}; cannot propose.",
               "proposal": {}}
        transcript.append({"node": "strategist", "summary": out["narrative"]})
        return {"strategist": out, "transcript": transcript}

    pit_loss_s = state["pit_loss"].get("pit_loss_s") or 22.0
    deg = DegradationModel()
    numbers = _scan_pit_laps(deg, focus, race_state, pit_loss_s)

    revision = state.get("revision_count", 0) > 0
    skeptic_note = ""
    if revision and state.get("skeptic", {}).get("disagreement_reason"):
        skeptic_note = ("\nThe skeptic rejected the previous proposal: "
                        + state["skeptic"]["disagreement_reason"]
                        + "\nAddress that objection explicitly.")

    form = state.get("driver_form", {}).get("metrics")
    role_prompt = (
        f"As the race strategist for {race_state['focus_driver']} "
        f"(P{focus['position']}, lap {race_state['lap']}/{race_state['total_laps']}), "
        "explain the proposed pit window and compound choice." + skeptic_note)
    data_block = json.dumps({
        "pit_window": {k: numbers.get(k) for k in
                       ("window_earliest", "window_latest", "best_pit_lap",
                        "new_compound", "deg_rate_now_s_per_lap", "pit_loss_s")},
        "gap_ahead_s": focus.get("gap_ahead_s"),
        "gap_behind_s": focus.get("gap_behind_s"),
        "driver_form": form,
    }, indent=1)

    out = {
        "role": "strategist",
        "numbers": numbers,
        "narrative": narrate(role_prompt, data_block),
        "proposal": {
            "pit_lap": numbers.get("best_pit_lap"),
            "window": [numbers.get("window_earliest"),
                       numbers.get("window_latest")],
            "compound": numbers.get("new_compound"),
        },
    }
    transcript.append({
        "node": "strategist" + (" (revised)" if revision else ""),
        "summary": f"Proposes pit lap {out['proposal'].get('pit_lap')} "
                   f"(window {out['proposal'].get('window')}) onto "
                   f"{out['proposal'].get('compound')}.",
        "narrative": out["narrative"],
        "numbers": {k: numbers.get(k) for k in
                    ("best_pit_lap", "window_earliest", "window_latest",
                     "new_compound", "deg_rate_now_s_per_lap", "pit_loss_s")},
    })
    return {"strategist": out, "transcript": transcript}
