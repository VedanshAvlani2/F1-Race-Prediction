"""Regulation rule engine.

Encodes the strategy-relevant subset of the FIA Sporting Regulations as
deterministic checks. This is NOT a full regulation ingestion system;
only rules that change pit strategy are covered, and that scope is stated
in the README.

Rules encoded:
  R1 two compound rule: in a dry race each driver must use at least two
     different slick compounds (Sporting Regs Art. 30.5m). A zero stop
     strategy is therefore illegal unless the race is declared wet.
  R2 proposed pit lap must lie inside the remaining race window.
  R3 a stop within roughly the last 2 laps almost always loses net time
     unless a safety car neutralises it; flagged as a warning, not a veto.
"""

from dataclasses import dataclass


@dataclass
class RuleCheck:
    rule: str
    level: str   # "violation" or "warning"
    message: str


def check_proposal(proposed_pit_lap: int | None, current_lap: int,
                   total_laps: int, compounds_used: list,
                   race_is_wet: bool, sc_active: bool) -> list:
    checks = []
    distinct_slicks = {c for c in compounds_used if c in ("SOFT", "MEDIUM", "HARD")}

    if proposed_pit_lap is None:
        if not race_is_wet and len(distinct_slicks) < 2:
            checks.append(RuleCheck(
                rule="R1 two compound rule",
                level="violation",
                message="No further stop proposed but only "
                        f"{len(distinct_slicks)} slick compound(s) used in a "
                        "dry race. At least one more stop is mandatory.",
            ))
        return checks

    if not (current_lap < proposed_pit_lap <= total_laps):
        checks.append(RuleCheck(
            rule="R2 pit window bounds",
            level="violation",
            message=f"Proposed pit lap {proposed_pit_lap} is outside the "
                    f"remaining window (laps {current_lap + 1}-{total_laps}).",
        ))

    if proposed_pit_lap > total_laps - 2 and not sc_active:
        checks.append(RuleCheck(
            rule="R3 late stop",
            level="warning",
            message=f"Pit on lap {proposed_pit_lap} of {total_laps} leaves "
                    "under 2 laps to recover pit loss without a safety car.",
        ))
    return checks
