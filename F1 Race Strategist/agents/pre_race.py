"""Pre-race strategy briefing: a second small LangGraph over the plan
enumerator. Same personas as the live graph, same rule: deterministic
numbers first, Groq narrates after.

plan -> strategist -> risk -> skeptic -> synthesis

The skeptic disagrees when Plan A's Monte Carlo margin over Plan B is
under 1 second (a toss up presented as a clear winner is dishonest) or
when a stint exceeds 38 laps (outside the degradation model's dense
training range). Disagreement lowers the brief's confidence.
"""

import json

import pandas as pd
from langgraph.graph import END, StateGraph
from typing import TypedDict

from agents.llm import narrate
from config.settings import STORE_DIR
from models.safety_car import SafetyCarModel
from models.strategy_planner import enumerate_plans

MAX_MODELED_STINT = 38


class PreRaceState(TypedDict, total=False):
    event: str
    track_temp: float
    total_laps: int
    focus_driver: str
    grid: list
    grid_source: str
    plan_table: dict
    strategist: dict
    risk: dict
    skeptic: dict
    synthesis: dict
    transcript: list


def _pit_loss_for(event: str) -> dict:
    table_file = STORE_DIR / "circuit_pit_loss.csv"
    if table_file.exists():
        table = pd.read_csv(table_file)
        row = table[table["event"] == event]
        if not row.empty:
            return {"pit_loss_s": float(row.iloc[0]["pit_loss_s"]),
                    "n_stops": int(row.iloc[0]["n_stops"]),
                    "low_confidence": bool(row.iloc[0]["low_confidence"])}
        med = float(table["pit_loss_s"].median())
        return {"pit_loss_s": round(med, 2), "n_stops": 0,
                "low_confidence": True,
                "note": "no measured stops here, using all circuit median"}
    return {"pit_loss_s": 22.0, "low_confidence": True,
            "note": "pit loss table missing"}


def plan_node(state: dict) -> dict:
    sc = SafetyCarModel().race_probability(state["event"])
    pit = _pit_loss_for(state["event"])
    table = enumerate_plans(
        state["event"], state["track_temp"], state["total_laps"],
        pit["pit_loss_s"], sc["p_sc_race"])
    table["pit_loss_detail"] = pit
    table["sc_detail"] = sc
    transcript = [{
        "node": "data",
        "summary": f"{len(table['plans'])} legal plans enumerated for "
                   f"{state['event']} at {state['track_temp']}C, pit loss "
                   f"{pit['pit_loss_s']}s, P(SC)={sc['p_sc_race']}.",
    }]
    return {"plan_table": table, "transcript": transcript}


def _fmt_plan(p: dict) -> str:
    return (f"{p['stops']}-stop {'-'.join(c[0] for c in p['sequence'])} "
            f"(pit {p['pit_laps']}), MC {p['mc_mean_time_delta_s']}s, "
            f"+{p['gap_to_best_s']}s to best")


def strategist_node(state: dict) -> dict:
    table = state["plan_table"]
    a, b = table["plans"][0], table["plans"][1]
    grid_pos = next((g["pos"] for g in state.get("grid", [])
                     if g["code"] == state.get("focus_driver")), None)
    numbers = {"plan_a": a, "plan_b": b,
               "focus_driver": state.get("focus_driver"),
               "grid_position": grid_pos}
    role_prompt = (
        f"As race strategist briefing for {state['event']}, present Plan A "
        f"and the fallback Plan B for {state.get('focus_driver') or 'the team'}"
        + (f" starting P{grid_pos}" if grid_pos else "")
        + ". Explain why Plan A's compound sequence and windows win on the "
          "numbers. Mention the traffic caveat if starting in the midfield.")
    out = {"role": "strategist", "numbers": numbers,
           "narrative": narrate(role_prompt, json.dumps(numbers, indent=1)),
           "proposal": {"plan_a": a, "plan_b": b}}
    transcript = list(state["transcript"])
    transcript.append({"node": "strategist",
                       "summary": f"Plan A: {_fmt_plan(a)}. Plan B: {_fmt_plan(b)}.",
                       "narrative": out["narrative"]})
    return {"strategist": out, "transcript": transcript}


def risk_node(state: dict) -> dict:
    table = state["plan_table"]
    a = table["plans"][0]
    sc = table["sc_detail"]
    most_robust = max(table["plans"], key=lambda p: p["p_sc_helps"])
    numbers = {"p_sc_race": sc["p_sc_race"],
               "plan_a_p_sc_helps": a["p_sc_helps"],
               "most_sc_robust_plan": _fmt_plan(most_robust),
               "sc_model_low_confidence": True}
    role_prompt = (
        "As team principal, assess how safety car likelihood should shape "
        "the pre-race plan and what trigger conditions should switch plans "
        "mid race. The SC model is low confidence; say so.")
    out = {"role": "risk", "numbers": numbers,
           "narrative": narrate(role_prompt, json.dumps(numbers, indent=1))}
    transcript = list(state["transcript"])
    transcript.append({
        "node": "risk",
        "summary": f"P(SC)={sc['p_sc_race']} (low confidence). Plan A "
                   f"benefits from SC in {a['p_sc_helps']:.0%} of sims.",
        "narrative": out["narrative"]})
    return {"risk": out, "transcript": transcript}


def skeptic_node(state: dict) -> dict:
    table = state["plan_table"]
    a, b = table["plans"][0], table["plans"][1]
    margin = round(b["mc_mean_time_delta_s"] - a["mc_mean_time_delta_s"], 2)

    issues = []
    if margin < 1.0:
        issues.append(f"Plan A beats Plan B by only {margin}s, inside model "
                      "noise (deg model CV MAE is larger per stint). This "
                      "is a toss up, not a clear winner.")
    stint_edges = [0] + a["pit_laps"] + [table["total_laps"]]
    longest = max(e - s for s, e in zip(stint_edges, stint_edges[1:]))
    if longest > MAX_MODELED_STINT:
        issues.append(f"Plan A includes a {longest} lap stint, beyond the "
                      f"degradation model's dense training range "
                      f"({MAX_MODELED_STINT} laps). Extrapolation risk.")
    if table["pit_loss_detail"].get("low_confidence"):
        issues.append("Pit loss for this circuit rests on few measured "
                      "stops.")

    numbers = {"margin_a_over_b_s": margin, "longest_stint": longest,
               "issues": issues, "assumptions": table["assumptions"]}
    role_prompt = (
        "As the data scientist auditing this brief, "
        + ("state the objections plainly." if issues else
           "confirm the numbers hold and restate the standing assumptions."))
    out = {"role": "skeptic", "numbers": numbers,
           "narrative": narrate(role_prompt, json.dumps(numbers, indent=1)),
           "disagrees": bool(issues),
           "disagreement_reason": " ".join(issues)}
    transcript = list(state["transcript"])
    transcript.append({
        "node": "skeptic",
        "summary": ("DISAGREES: " + " ".join(issues)) if issues
                   else "Numbers hold within stated assumptions.",
        "narrative": out["narrative"]})
    return {"skeptic": out, "transcript": transcript}


def synthesis_node(state: dict) -> dict:
    table = state["plan_table"]
    a, b = table["plans"][0], table["plans"][1]
    skeptic = state.get("skeptic", {})
    conf = 2
    reasons = []
    if skeptic.get("disagrees"):
        conf -= 1
        reasons.append(skeptic.get("disagreement_reason", ""))
    if table["sc_detail"].get("circuit_history_races", 0) < 2:
        conf -= 1
        reasons.append("thin circuit history for the SC tendency")
    conf = max(conf, 0)
    levels = ["low", "medium", "high"]

    brief = {
        "event": table["event"],
        "plan_a": a,
        "plan_b": b,
        "switch_triggers": [
            f"SC before lap {a['pit_laps'][0] - WINDOW_HINT}: box "
            "immediately, cheap stop outweighs tire timing",
            f"Plan A window missed (past lap {a['pit_laps'][-1] + 4}): "
            "convert to Plan B",
            "Rain: all slick plans void, revert to live strategy engine",
        ],
        "confidence": levels[conf],
        "confidence_reasons": reasons or ["agents aligned, "
                                          "models in validated range"],
        "skeptic_dissent": skeptic.get("disagreement_reason", ""),
    }
    role_prompt = (
        "As chief strategist, write the pre-race brief: Plan A, Plan B, "
        "switch triggers, confidence and why. Keep it pit wall terse.")
    out = {"role": "synthesis", "numbers": brief,
           "narrative": narrate(role_prompt, json.dumps(brief, indent=1))}
    transcript = list(state["transcript"])
    transcript.append({
        "node": "synthesis",
        "summary": f"BRIEF: Plan A {_fmt_plan(a)}; fallback {_fmt_plan(b)}; "
                   f"confidence {brief['confidence']}.",
        "narrative": out["narrative"]})
    return {"synthesis": out, "transcript": transcript}


WINDOW_HINT = 3


def build_pre_race_graph():
    g = StateGraph(PreRaceState)
    g.add_node("plan", plan_node)
    g.add_node("strategist", strategist_node)
    g.add_node("risk", risk_node)
    g.add_node("skeptic", skeptic_node)
    g.add_node("synthesis", synthesis_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "strategist")
    g.add_edge("strategist", "risk")
    g.add_edge("risk", "skeptic")
    g.add_edge("skeptic", "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()


def run_pre_race(event: str, track_temp: float, total_laps: int,
                 focus_driver: str = "", grid: list = None) -> dict:
    graph = build_pre_race_graph()
    result = graph.invoke({
        "event": event, "track_temp": track_temp, "total_laps": total_laps,
        "focus_driver": focus_driver, "grid": grid or [],
    })
    return {
        "inputs": {"event": event, "track_temp": track_temp,
                   "total_laps": total_laps, "focus_driver": focus_driver},
        "plan_table": result["plan_table"],
        "transcript": result["transcript"],
        "brief": result.get("synthesis", {}).get("numbers"),
        "final_narrative": result.get("synthesis", {}).get("narrative"),
    }
