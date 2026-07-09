"""Synthesis node: reconciles the three agents into one recommendation
with a confidence level. Shows its work: every agent position and how it
was weighed is in the output, including unresolved disagreement.

Confidence is deterministic:
  start at high
  -1 step if the skeptic still disagrees after revision
  -1 step if the SC model shifted the window (low confidence model)
  -1 step if pit loss for this circuit is low confidence
"""

import json

from agents.llm import narrate

LEVELS = ["low", "medium", "high"]


def synthesis_node(state: dict) -> dict:
    strategist = state.get("strategist", {})
    risk = state.get("risk", {})
    skeptic = state.get("skeptic", {})
    race_state = state["race_state"]
    transcript = list(state.get("transcript", []))

    proposal = dict(strategist.get("proposal", {}))
    shift = risk.get("proposal", {}).get("shift", "none")
    window = proposal.get("window") or [None, None]

    adjustments = []
    if shift == "extend_window_later" and window[1] is not None:
        window = [window[0], min(window[1] + 3,
                                 race_state["total_laps"] - 1)]
        adjustments.append("window extended 3 laps later per risk agent "
                           "(SC/weather tendency)")
    if shift == "pit_now_sc_active":
        window = [race_state["lap"] + 1, race_state["lap"] + 1]
        proposal["pit_lap"] = race_state["lap"] + 1
        adjustments.append("SC/VSC active: pit at once, stop is cheap")

    conf = 2
    reasons = []
    if skeptic.get("unresolved_disagreement"):
        conf -= 1
        reasons.append("skeptic disagreement unresolved after revision")
    if shift != "none":
        conf -= 1
        reasons.append("recommendation leans on the low confidence SC model")
    if state.get("pit_loss", {}).get("low_confidence"):
        conf -= 1
        reasons.append("pit loss estimate for this circuit has few samples")
    conf = max(conf, 0)

    final = {
        "driver": race_state["focus_driver"],
        "pit_lap": proposal.get("pit_lap"),
        "window": window,
        "compound": proposal.get("compound"),
        "confidence": LEVELS[conf],
        "confidence_reasons": reasons or ["all agents aligned, models in "
                                          "validated range"],
        "adjustments": adjustments,
        "agent_positions": {
            "strategist": strategist.get("proposal"),
            "risk": risk.get("proposal"),
            "skeptic": {
                "disagrees": skeptic.get("disagrees", False)
                             or skeptic.get("unresolved_disagreement", False),
                "reason": skeptic.get("disagreement_reason", ""),
            },
        },
    }

    role_prompt = (
        "As the chief strategist, write the final call. State each agent's "
        "position, how conflicts were reconciled, and the confidence level "
        "with its reasons. Be direct; a pit wall has seconds.")
    data_block = json.dumps(final, indent=1)

    out = {
        "role": "synthesis",
        "numbers": final,
        "narrative": narrate(role_prompt, data_block),
    }
    transcript.append({
        "node": "synthesis",
        "summary": f"FINAL: pit lap {final['pit_lap']} "
                   f"(window {final['window']}) onto {final['compound']}, "
                   f"confidence {final['confidence']}.",
        "narrative": out["narrative"],
        "numbers": final,
    })
    return {"synthesis": out, "transcript": transcript}
