"""Shared LangGraph state. One object flows through every node; each
agent appends its numeric findings and narrative so disagreement stays
visible end to end."""

from typing import Any, TypedDict


class AgentOutput(TypedDict, total=False):
    role: str
    numbers: dict          # deterministic model outputs, never LLM generated
    narrative: str         # Groq prose over those numbers
    proposal: dict         # e.g. {"pit_lap_earliest": 18, "pit_lap_latest": 22}
    disagrees: bool
    disagreement_reason: str


class StrategyState(TypedDict, total=False):
    # Inputs
    year: int
    event: str
    lap: int
    focus_driver: str

    # Data node output
    race_state: dict
    pit_loss: dict          # circuit pit loss row
    driver_form: dict       # focus driver's form metrics + source
    championship: dict      # tracker output summary

    # Agent outputs
    strategist: AgentOutput
    risk: AgentOutput
    skeptic: AgentOutput
    synthesis: AgentOutput

    # Control
    revision_count: int
    transcript: list        # ordered log of every agent turn
    error: str
