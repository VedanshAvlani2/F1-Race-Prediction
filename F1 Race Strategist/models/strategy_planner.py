"""Pre-race strategy planner.

Enumerates every sensible legal race plan (1 and 2 stop, all compound
sequences satisfying the two compound rule), computes total race time
deterministically from the fitted degradation model plus measured circuit
pit loss, then stress tests each plan with a Monte Carlo safety car
simulation driven by the SC model's per lap hazard.

Simplifications, stated openly and repeated in the output:
  * No traffic or track position model. Times are clean air optima.
  * A safety car saves roughly 45 percent of pit loss if it appears
    while a planned stop window is open (field bunched, pits at reduced
    relative cost). Other SC effects cancel across plans.
  * Track temperature is the user's forecast input.
"""

import itertools
import json

import numpy as np

from config.settings import STORE_DIR
from models.tire_degradation import DegradationModel

SLICKS = ("SOFT", "MEDIUM", "HARD")
MIN_STINT = 8
SC_PIT_SAVING = 0.45
WINDOW = 3  # laps around a planned stop where an SC makes it cheap


def _stint_cost(deg_table: dict, compound: str, start_lap: int, length: int) -> float:
    col = deg_table[compound]
    return float(sum(col[min(age, len(col) - 1)]
                     for age in range(1, length + 1)))


def _build_deg_table(deg: DegradationModel, temp: float, total_laps: int) -> dict:
    """Per compound: delta seconds by tire age, fuel effect averaged out by
    evaluating at mid race lap so every plan is compared on equal fuel."""
    mid = total_laps / 2
    table = {}
    for comp in SLICKS:
        # Raw model deltas, no clamping: negative values (fresh tire faster
        # than the driver's median pace) are real and must be kept so
        # compound and age differences separate the plans.
        table[comp] = [0.0] + [
            deg.predict_delta(age, comp, temp, mid) for age in range(1, 46)
        ]
    return table


def enumerate_plans(event: str, track_temp: float, total_laps: int,
                    pit_loss_s: float, p_sc_race: float,
                    n_mc: int = 800, seed: int = 7) -> dict:
    deg = DegradationModel()
    table = _build_deg_table(deg, track_temp, total_laps)
    rng = np.random.default_rng(seed)

    p_lap = 1.0 - (1.0 - min(max(p_sc_race, 0.01), 0.99)) ** (1.0 / total_laps)
    sc_laps = np.where(
        rng.random(n_mc) < 1.0 - (1.0 - p_lap) ** total_laps,
        rng.integers(1, total_laps + 1, n_mc), 0)

    plans = []

    # One stop: ordered pairs of distinct compounds. Ties within 0.05s are
    # broken toward balanced stints; when the deg surface is flat the
    # model genuinely cannot separate windows, and a mid race stop is the
    # operationally sane default.
    for c1, c2 in itertools.permutations(SLICKS, 2):
        best = None
        for pit in range(MIN_STINT, total_laps - MIN_STINT + 1):
            cost = (_stint_cost(table, c1, 1, pit)
                    + _stint_cost(table, c2, pit + 1, total_laps - pit)
                    + pit_loss_s)
            imbalance = abs(pit - (total_laps - pit))
            if best is None or cost < best[0] - 0.05 or (
                    abs(cost - best[0]) <= 0.05 and imbalance < best[2]):
                best = (cost, pit, imbalance)
        plans.append({"stops": 1, "sequence": [c1, c2],
                      "pit_laps": [best[1]], "clean_time_delta_s": best[0]})

    # Two stops: triples using at least two distinct compounds.
    for seq in itertools.product(SLICKS, repeat=3):
        if len(set(seq)) < 2:
            continue
        best = None
        third = total_laps / 3
        for p1 in range(MIN_STINT, total_laps - 2 * MIN_STINT + 1, 2):
            for p2 in range(p1 + MIN_STINT, total_laps - MIN_STINT + 1, 2):
                cost = (_stint_cost(table, seq[0], 1, p1)
                        + _stint_cost(table, seq[1], p1 + 1, p2 - p1)
                        + _stint_cost(table, seq[2], p2 + 1, total_laps - p2)
                        + 2 * pit_loss_s)
                imbalance = (abs(p1 - third) + abs((p2 - p1) - third)
                             + abs((total_laps - p2) - third))
                if best is None or cost < best[0] - 0.05 or (
                        abs(cost - best[0]) <= 0.05 and imbalance < best[3]):
                    best = (cost, p1, p2, imbalance)
        plans.append({"stops": 2, "sequence": list(seq),
                      "pit_laps": [best[1], best[2]],
                      "clean_time_delta_s": best[0]})

    # Deduplicate identical compound sequences, keep the cheapest.
    seen = {}
    for p in plans:
        key = tuple(p["sequence"])
        if key not in seen or p["clean_time_delta_s"] < seen[key]["clean_time_delta_s"]:
            seen[key] = p
    plans = list(seen.values())

    # Monte Carlo SC benefit per plan.
    for p in plans:
        savings = np.zeros(n_mc)
        for i, sc in enumerate(sc_laps):
            if sc == 0:
                continue
            if any(abs(int(sc) - pit) <= WINDOW for pit in p["pit_laps"]):
                savings[i] = SC_PIT_SAVING * pit_loss_s
        p["mc_mean_time_delta_s"] = round(p["clean_time_delta_s"]
                                          - float(savings.mean()), 2)
        p["p_sc_helps"] = round(float((savings > 0).mean()), 3)
        p["clean_time_delta_s"] = round(p["clean_time_delta_s"], 2)

    plans.sort(key=lambda p: p["mc_mean_time_delta_s"])
    base = plans[0]["mc_mean_time_delta_s"]
    for i, p in enumerate(plans):
        p["rank"] = i + 1
        p["gap_to_best_s"] = round(p["mc_mean_time_delta_s"] - base, 2)

    return {
        "event": event,
        "track_temp": track_temp,
        "total_laps": total_laps,
        "pit_loss_s": pit_loss_s,
        "p_sc_race": p_sc_race,
        "n_monte_carlo": n_mc,
        "plans": plans[:8],
        "assumptions": [
            "clean air optimum, no traffic or track position model",
            f"SC saves {int(SC_PIT_SAVING * 100)}% of pit loss when it "
            f"appears within {WINDOW} laps of a planned stop",
            "degradation from fitted model at the forecast track temp, "
            "fuel effect equalised at mid race",
            "SC probability is the low confidence circuit tendency",
        ],
    }


def predicted_grid() -> dict:
    """Predicted starting order from rolling driver form (average finish
    over the last 5 races). Deterministic, source flagged, and honest:
    this is a form ranking, not a qualifying simulation."""
    form_file = STORE_DIR / "driver_form.json"
    if not form_file.exists():
        return {"source": "unavailable", "grid": [],
                "note": "Run the driver form tracker first (it fills "
                        "automatically when the API starts with network)."}
    data = json.loads(form_file.read_text(encoding="utf-8"))
    drivers = data.get("drivers", {})
    order = sorted(drivers.items(),
                   key=lambda kv: kv[1].get("avg_finish", 99))
    return {
        "source": data.get("source"),
        "season": data.get("season"),
        "note": "Ranked by rolling average finish over each driver's last "
                f"{data.get('window', 5)} races. A form ranking, not a "
                "qualifying simulation.",
        "grid": [{"pos": i + 1, "code": code,
                  "driver": v.get("driver", code),
                  "avg_finish": v.get("avg_finish")}
                 for i, (code, v) in enumerate(order)],
    }
