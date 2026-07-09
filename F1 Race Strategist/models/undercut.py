"""Undercut / overcut calculator. Pure deterministic math, no learning.

Model assumptions, stated explicitly:
  * Degradation rates are locally linear over the projection horizon.
  * A fresh tire's advantage over an old one equals the degradation delta
    accumulated by the old tire (age * deg rate) plus any compound offset.
  * Pit loss is the circuit median measured from real in/out laps.
  * Traffic after the stop is not modelled; the skeptic agent flags this.

All numbers consumed by the agents come from this module or the fitted
models. The LLM never invents any of them.
"""

from dataclasses import dataclass, field


@dataclass
class UndercutInput:
    gap_ahead_s: float          # to the car we are attacking (positive)
    gap_behind_s: float         # to the car defending against us
    pit_loss_s: float           # circuit specific, measured
    my_tyre_age: float
    my_deg_rate: float          # s/lap, from degradation model
    rival_tyre_age: float
    rival_deg_rate: float       # s/lap
    fresh_compound_offset_s: float = 0.0  # pace delta of new compound vs current, negative = faster
    horizon_laps: int = 15


@dataclass
class UndercutResult:
    net_gain_if_rival_stays_out: list = field(default_factory=list)
    break_even_lap: int | None = None
    undercut_delta_one_lap: float = 0.0
    verdict: str = ""
    assumptions: list = field(default_factory=list)


def compute(inp: UndercutInput) -> UndercutResult:
    """Two scenarios:

    1. Rival pits one lap after us (classic undercut attempt): our gain is
       one lap on fresh rubber against his worn tire, minus nothing (both
       pay pit loss).
    2. Rival stays out: we pay pit loss now and claw it back lap by lap
       through the tire delta. break_even_lap is when cumulative gain
       exceeds the pit loss.
    """
    res = UndercutResult()

    # Scenario 1: both pit, we go first. Advantage = rival's extra lap of
    # degradation on old tires vs our out-lap on fresh.
    old_tire_pace_penalty = inp.rival_tyre_age * inp.rival_deg_rate
    fresh_advantage = old_tire_pace_penalty - inp.fresh_compound_offset_s
    res.undercut_delta_one_lap = round(fresh_advantage, 3)

    # Scenario 2: rival stays out.
    cumulative = -inp.pit_loss_s
    for i in range(1, inp.horizon_laps + 1):
        my_pace_delta = i * inp.my_deg_rate + inp.fresh_compound_offset_s
        rival_pace_delta = (inp.rival_tyre_age + i) * inp.rival_deg_rate
        cumulative += rival_pace_delta - my_pace_delta
        res.net_gain_if_rival_stays_out.append(round(cumulative, 3))
        if res.break_even_lap is None and cumulative > 0:
            res.break_even_lap = i

    gain_end = res.net_gain_if_rival_stays_out[-1]
    if res.break_even_lap is not None and res.break_even_lap <= inp.horizon_laps // 2:
        res.verdict = (f"undercut favourable: pit loss recovered in "
                       f"{res.break_even_lap} laps, projected "
                       f"{gain_end:+.1f}s after {inp.horizon_laps} laps")
    elif res.break_even_lap is not None:
        res.verdict = (f"marginal: break even only on lap "
                       f"{res.break_even_lap} of {inp.horizon_laps}")
    else:
        res.verdict = (f"undercut unfavourable: pit loss never recovered "
                       f"inside {inp.horizon_laps} laps "
                       f"({gain_end:+.1f}s at horizon)")

    res.assumptions = [
        f"pit loss {inp.pit_loss_s:.1f}s (circuit median from real stops)",
        f"my deg {inp.my_deg_rate:.3f} s/lap at age {inp.my_tyre_age:.0f}",
        f"rival deg {inp.rival_deg_rate:.3f} s/lap at age {inp.rival_tyre_age:.0f}",
        "linear degradation over horizon, no traffic model",
    ]
    return res
