"""Risk agent (persona: team principal).

Deterministic core: safety car probability model output for the remaining
laps plus a measured weather trend. Produces a numeric shift suggestion
for the strategist's window; Groq explains the big picture including
championship context. The SC model's low_confidence flag is always
surfaced, never hidden.
"""

import json

from agents.llm import narrate
from models.safety_car import SafetyCarModel


def risk_node(state: dict) -> dict:
    race_state = state["race_state"]
    transcript = list(state.get("transcript", []))

    sc_model = SafetyCarModel()
    sc = sc_model.probability_next_laps(
        race_state["event"], race_state["laps_remaining"],
        race_state["total_laps"])

    weather = race_state.get("weather", {})
    rain_trend = weather.get("rain_last_20_samples", 0.0)
    raining = weather.get("rainfall", False)

    # Deterministic shift logic, thresholds stated in the open:
    #   p(SC in remaining laps) > 0.35 or active rain trend: keep window
    #   open later (a cheap stop may appear under SC).
    #   SC currently deployed: pit now advice.
    shift = "none"
    if race_state["track_status"] in ("safety car", "virtual safety car"):
        shift = "pit_now_sc_active"
    elif sc["p_sc_remaining_laps"] > 0.35 or raining:
        shift = "extend_window_later"
    elif rain_trend > 0.3:
        shift = "extend_window_later"

    numbers = {
        "sc_probability": sc,
        "weather_now": weather,
        "track_status": race_state["track_status"],
        "suggested_shift": shift,
        "shift_thresholds": {"p_sc_remaining": 0.35, "rain_trend": 0.3},
        "model_low_confidence": True,
    }

    strategist = state.get("strategist", {})
    role_prompt = (
        "As the team principal weighing risk, assess how the safety car "
        "probability and weather should shift the strategist's proposed "
        f"window {strategist.get('proposal', {}).get('window')}. The SC "
        "model is flagged low confidence (32 race sample); say so.")
    data_block = json.dumps({
        "sc": sc, "weather": weather,
        "suggested_shift": shift,
        "championship": state.get("championship", {}),
    }, indent=1)

    out = {
        "role": "risk",
        "numbers": numbers,
        "narrative": narrate(role_prompt, data_block),
        "proposal": {"shift": shift},
    }
    transcript.append({
        "node": "risk",
        "summary": f"P(SC in remaining {sc['laps_remaining']} laps) = "
                   f"{sc['p_sc_remaining_laps']} (low confidence). "
                   f"Suggested shift: {shift}.",
        "narrative": out["narrative"],
        "numbers": {"p_sc_remaining_laps": sc["p_sc_remaining_laps"],
                    "p_sc_race": sc["p_sc_race"],
                    "suggested_shift": shift},
    })
    return {"risk": out, "transcript": transcript}
