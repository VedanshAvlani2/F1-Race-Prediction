"""LangGraph wiring.

data -> strategist -> risk -> skeptic -> (revise once?) -> synthesis

The skeptic can send the flow back to the strategist exactly once when
the raw math contradicts the proposal. Both passes stay in the
transcript, so disagreement is visible in the final output instead of
being papered over. This routing is the reason LangGraph was chosen over
CrewAI: disagreement handling is an explicit, inspectable edge.
"""

from langgraph.graph import END, StateGraph

from agents.nodes.data_node import data_node
from agents.nodes.risk import risk_node
from agents.nodes.skeptic import skeptic_node
from agents.nodes.strategist import strategist_node
from agents.nodes.synthesis import synthesis_node
from agents.state import StrategyState


def _strategist_with_revision(state: dict) -> dict:
    update = strategist_node(state)
    update["revision_count"] = state.get("revision_count", 0) + 1
    return update


def _route_after_skeptic(state: dict) -> str:
    skeptic = state.get("skeptic", {})
    if skeptic.get("disagrees") and state.get("revision_count", 0) < 2:
        return "revise"
    return "synthesise"


def build_graph():
    g = StateGraph(StrategyState)
    g.add_node("data", data_node)
    g.add_node("strategist", _strategist_with_revision)
    g.add_node("risk", risk_node)
    g.add_node("skeptic", skeptic_node)
    g.add_node("synthesis", synthesis_node)

    g.set_entry_point("data")
    g.add_edge("data", "strategist")
    g.add_edge("strategist", "risk")
    g.add_edge("risk", "skeptic")
    g.add_conditional_edges("skeptic", _route_after_skeptic,
                            {"revise": "strategist",
                             "synthesise": "synthesis"})
    g.add_edge("synthesis", END)
    return g.compile()


def run_deliberation(year: int, event: str, lap: int,
                     focus_driver: str) -> dict:
    graph = build_graph()
    result = graph.invoke({
        "year": year, "event": event, "lap": lap,
        "focus_driver": focus_driver,
    })
    return {
        "inputs": {"year": year, "event": event, "lap": lap,
                   "driver": focus_driver},
        "race_state": result.get("race_state"),
        "transcript": result.get("transcript", []),
        "recommendation": result.get("synthesis", {}).get("numbers"),
        "final_narrative": result.get("synthesis", {}).get("narrative"),
    }
