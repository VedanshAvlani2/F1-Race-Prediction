"""Skeptic agent (persona: data scientist).

Cross-checks the strategist's proposal against the raw undercut math and
the regulation engine. Disagreement is a first class output: if the
numbers do not support the proposed lap, disagrees=True and the graph
routes back to the strategist for one revision before synthesis.
"""

import json
from dataclasses import asdict

from agents.llm import narrate
from models.regulations import check_proposal
from models.tire_degradation import DegradationModel
from models.undercut import UndercutInput, compute


def skeptic_node(state: dict) -> dict:
    race_state = state["race_state"]
    strategist = state.get("strategist", {})
    proposal = strategist.get("proposal", {})
    focus = race_state.get("focus")
    transcript = list(state.get("transcript", []))

    if not proposal or focus is None:
        out = {"role": "skeptic", "numbers": {}, "disagrees": False,
               "narrative": "No proposal to audit."}
        transcript.append({"node": "skeptic", "summary": out["narrative"]})
        return {"skeptic": out, "transcript": transcript}

    pit_loss_s = state["pit_loss"].get("pit_loss_s") or 22.0
    temp = race_state["weather"].get("track_temp", 30.0)
    deg = DegradationModel()

    # Rival = car directly ahead; if leading, audit against car behind.
    order = race_state["order"]
    my_pos = focus["position"]
    rival = next((e for e in order if e["position"] == my_pos - 1), None) \
        or next((e for e in order if e["position"] == my_pos + 1), None)

    undercut = None
    if rival and rival.get("tyre_age") is not None:
        inp = UndercutInput(
            gap_ahead_s=focus.get("gap_ahead_s") or 0.0,
            gap_behind_s=focus.get("gap_behind_s") or 0.0,
            pit_loss_s=pit_loss_s,
            my_tyre_age=focus["tyre_age"],
            my_deg_rate=deg.deg_rate(focus["tyre_age"], focus["compound"],
                                     temp, race_state["lap"]),
            rival_tyre_age=rival["tyre_age"],
            rival_deg_rate=deg.deg_rate(rival["tyre_age"], rival["compound"],
                                        temp, race_state["lap"]),
            horizon_laps=min(15, race_state["laps_remaining"]),
        )
        undercut = compute(inp)

    rules = check_proposal(
        proposed_pit_lap=proposal.get("pit_lap"),
        current_lap=race_state["lap"],
        total_laps=race_state["total_laps"],
        compounds_used=race_state["compounds_used"],
        race_is_wet=race_state["weather"].get("rainfall", False),
        sc_active=race_state["track_status"] in ("safety car",
                                                 "virtual safety car"),
    )
    violations = [asdict(r) for r in rules]

    disagrees = False
    reasons = []
    if any(r["level"] == "violation" for r in violations):
        disagrees = True
        reasons.append("regulation violation: " + "; ".join(
            r["message"] for r in violations if r["level"] == "violation"))
    if undercut is not None and undercut.break_even_lap is None:
        disagrees = True
        reasons.append(
            f"undercut math does not support pitting: pit loss "
            f"{pit_loss_s:.1f}s is never recovered within the horizon "
            f"({undercut.net_gain_if_rival_stays_out[-1]:+.1f}s at end)")
    # revision_count is 1 after the strategist's first pass and 2 after
    # the revision pass, so "already revised" means count > 1.
    already_revised = state.get("revision_count", 0) > 1

    numbers = {
        "undercut": None if undercut is None else {
            "verdict": undercut.verdict,
            "break_even_lap": undercut.break_even_lap,
            "one_lap_undercut_delta_s": undercut.undercut_delta_one_lap,
            "net_gain_curve": undercut.net_gain_if_rival_stays_out,
            "assumptions": undercut.assumptions,
            "rival": None if rival is None else rival["driver"],
        },
        "regulation_checks": violations,
        "audited_pit_lap": proposal.get("pit_lap"),
    }

    role_prompt = (
        "As the data scientist auditing the strategist, state whether the "
        "raw undercut arithmetic and regulation checks support the proposed "
        f"pit lap {proposal.get('pit_lap')}. "
        + ("You disagree; articulate exactly why using the numbers."
           if disagrees else "You broadly agree; note residual caveats "
           "(no traffic model, linear degradation assumption)."))
    data_block = json.dumps(numbers, indent=1, default=str)

    out = {
        "role": "skeptic",
        "numbers": numbers,
        "narrative": narrate(role_prompt, data_block),
        "disagrees": disagrees and not already_revised,
        "disagreement_reason": "; ".join(reasons),
        "unresolved_disagreement": disagrees and already_revised,
    }
    transcript.append({
        "node": "skeptic",
        "summary": (("DISAGREES: " + "; ".join(reasons)) if disagrees
                    else "Numbers support the proposal within stated "
                         "assumptions."),
        "narrative": out["narrative"],
        "numbers": {"break_even_lap": numbers["undercut"]["break_even_lap"]
                    if numbers["undercut"] else None,
                    "violations": len([v for v in violations
                                       if v["level"] == "violation"])},
    })
    return {"skeptic": out, "transcript": transcript,
            "revision_count": state.get("revision_count", 0)}
