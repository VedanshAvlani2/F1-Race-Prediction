"""Groq LLM access for agent narratives.

Hard rule enforced by construction: the LLM receives numbers computed by
the deterministic models and writes prose about them. It is never asked
to produce a lap time, degradation rate or probability. If the API key is
missing or the call fails, agents fall back to a clearly labelled
template narrative built from the same numbers, so the system degrades
gracefully instead of inventing content.
"""

from config.settings import GROQ_API_KEY, GROQ_MODEL

SYSTEM = (
    "You are part of an F1 race strategy engineering team. You are given "
    "numeric outputs from validated models. Explain and reason about those "
    "numbers in 3-5 sentences of plain engineering language. Never invent "
    "numbers that were not provided. Refer only to the data given."
)


def narrate(role_prompt: str, data_block: str) -> str:
    if not GROQ_API_KEY:
        return ("[LLM narrative unavailable: GROQ_API_KEY not set. "
                "Numeric analysis above is complete and model-derived.]")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=GROQ_MODEL, temperature=0.3, max_tokens=400)
        msg = llm.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"{role_prompt}\n\nMODEL OUTPUTS:\n{data_block}"),
        ])
        return msg.content.strip()
    except Exception as exc:
        return (f"[LLM narrative unavailable: {type(exc).__name__}. "
                "Numeric analysis above is complete and model-derived.]")
