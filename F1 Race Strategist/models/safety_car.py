"""Safety car probability model.

Two stage design, both stages fully deterministic:

1. Race level: logistic regression on circuit features predicting whether
   at least one full safety car appears during a race. Features are the
   circuit's smoothed historical SC rate (empirical Bayes shrinkage toward
   the global rate, leave-one-race-out to avoid leakage) and a street
   circuit flag.

2. In race: the race level probability converts to a per lap hazard
   p_lap = 1 - (1 - P)^(1/total_laps), so the probability of a safety car
   in the next k laps is 1 - (1 - p_lap)^k.

HONESTY FLAG: with roughly 32 cached races this is a small sample for a
rare event model. Metrics report Brier score against a
predict-the-base-rate baseline. Treat outputs as calibrated tendencies,
not precise probabilities. The model is marked low_confidence=True in its
artifact and every agent that consumes it sees that flag.
"""

import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import LeaveOneOut

from config.settings import ARTIFACTS_DIR, ENGINE_ROOT

MODEL_FILE = ARTIFACTS_DIR / "safety_car.joblib"
CIRCUIT_FILE = ARTIFACTS_DIR / "circuit_sc_rates.json"
SHRINKAGE = 3.0  # pseudo-races of global-rate prior per circuit


def _street_set() -> set:
    cfg = yaml.safe_load((ENGINE_ROOT / "config" / "circuits.yaml").read_text(encoding="utf-8"))
    return set(cfg.get("street_circuits", []))


def _build_features(race_status: pd.DataFrame) -> pd.DataFrame:
    df = race_status.copy()
    df["any_sc"] = df["any_sc"].astype(bool)
    streets = _street_set()
    df["street"] = df["event"].isin(streets).astype(int)
    global_rate = df["any_sc"].mean()

    # Leave-one-out smoothed circuit rate: for each race, the rate uses
    # every other race at that circuit plus a global-rate prior.
    rates = []
    for idx, row in df.iterrows():
        others = df[(df["event"] == row["event"]) & (df.index != idx)]
        k = others["any_sc"].sum()
        n = len(others)
        rates.append((k + SHRINKAGE * global_rate) / (n + SHRINKAGE))
    df["circuit_rate_loo"] = rates
    df["global_rate"] = global_rate
    return df


def train(race_status: pd.DataFrame) -> dict:
    df = _build_features(race_status)
    x = df[["circuit_rate_loo", "street"]].values
    y = df["any_sc"].astype(int).values
    n = len(df)

    # Leave-one-out CV: the honest option for n this small.
    loo_probs = np.zeros(n)
    for train_idx, test_idx in LeaveOneOut().split(x):
        m = LogisticRegression()
        m.fit(x[train_idx], y[train_idx])
        loo_probs[test_idx] = m.predict_proba(x[test_idx])[:, 1]

    base_rate = y.mean()
    brier_model = brier_score_loss(y, loo_probs)
    brier_base = brier_score_loss(y, np.full(n, base_rate))

    model = LogisticRegression()
    model.fit(x, y)
    joblib.dump(model, MODEL_FILE)

    # Deployment-time circuit rates use ALL races (no LOO needed there).
    circuit_rates = {}
    global_rate = float(base_rate)
    for event, g in df.groupby("event"):
        k, cn = int(g["any_sc"].sum()), len(g)
        circuit_rates[event] = {
            "smoothed_rate": round((k + SHRINKAGE * global_rate) / (cn + SHRINKAGE), 4),
            "raw_sc_races": k,
            "n_races": cn,
            "street": bool(g["street"].iloc[0]),
        }
    CIRCUIT_FILE.write_text(json.dumps({
        "global_rate": round(global_rate, 4),
        "circuits": circuit_rates,
    }, indent=2), encoding="utf-8")

    return {
        "model": "LogisticRegression",
        "target": "at least one full safety car during the race",
        "features": ["circuit_rate_loo (shrunk, leakage-free)", "street"],
        "training_rows": int(n),
        "positive_rate": round(float(base_rate), 3),
        "validation": "leave-one-out cross validation",
        "brier_model": round(float(brier_model), 4),
        "brier_baseline_predict_base_rate": round(float(brier_base), 4),
        "log_loss": round(float(log_loss(y, np.clip(loo_probs, 1e-6, 1 - 1e-6))), 4),
        "low_confidence": True,
        "beats_baseline": bool(brier_model < brier_base),
        "caveats": [
            f"Only {n} races. Safety cars are rare events; per circuit "
            "sample sizes are 1-3 races. Treat as tendency, not truth.",
            "VSC-only interruptions are not counted as full SC here.",
            "If brier_model exceeds the baseline, the circuit features do "
            "not yet beat always-predicting-the-base-rate on this sample. "
            "The output is then best read as a smoothed base rate.",
        ],
    }


class SafetyCarModel:
    """Runtime wrapper used by the risk agent."""

    def __init__(self):
        self.model = joblib.load(MODEL_FILE)
        data = json.loads(CIRCUIT_FILE.read_text(encoding="utf-8"))
        self.circuits = data["circuits"]
        self.global_rate = data["global_rate"]
        self.low_confidence = True

    def race_probability(self, event: str) -> dict:
        info = self.circuits.get(event)
        if info is None:
            rate, street, n = self.global_rate, False, 0
        else:
            rate, street, n = info["smoothed_rate"], info["street"], info["n_races"]
        p = float(self.model.predict_proba([[rate, int(street)]])[0, 1])
        return {"event": event, "p_sc_race": round(p, 3),
                "circuit_history_races": n, "street": street}

    def probability_next_laps(self, event: str, laps_remaining: int,
                              total_laps: int) -> dict:
        race = self.race_probability(event)
        p_race = race["p_sc_race"]
        total = max(total_laps, 1)
        p_lap = 1.0 - (1.0 - p_race) ** (1.0 / total)
        k = max(min(laps_remaining, total), 0)
        p_window = 1.0 - (1.0 - p_lap) ** k
        return {
            **race,
            "laps_remaining": laps_remaining,
            "p_sc_per_lap": round(p_lap, 4),
            "p_sc_remaining_laps": round(p_window, 3),
            "low_confidence": True,
        }
