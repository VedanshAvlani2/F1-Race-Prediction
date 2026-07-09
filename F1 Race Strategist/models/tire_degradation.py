"""Tire degradation model.

Predicts lap time delta (seconds relative to that driver's median clean
race pace) as a function of tire age, compound, track temperature and lap
number (fuel load proxy). Trained on real green flag, dry, accurate laps
from the cached seasons.

Validation is GroupKFold grouped by race, so the reported error is
out-of-race generalisation, not a same-race split. A baseline MAE
(always predict the training mean) is reported next to the model MAE so
the improvement is visible and honest.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from config.settings import ARTIFACTS_DIR

SLICKS = ("SOFT", "MEDIUM", "HARD")
MODEL_FILE = ARTIFACTS_DIR / "tire_degradation.joblib"
MAX_AGE = 40


def build_training_frame(stint_laps: pd.DataFrame) -> pd.DataFrame:
    df = stint_laps.copy()
    df = df[
        df["green_lap"]
        & df["is_accurate"].astype(bool)
        & ~df["wet"]
        & ~df["is_box_lap"]
        & df["compound"].isin(SLICKS)
        & df["track_temp"].notna()
        & (df["tyre_life"] <= MAX_AGE)
        & (df["tyre_life"] >= 1)
    ].copy()

    # Delta vs the driver's own median clean lap in that race removes car
    # and circuit pace, leaving tire age, fuel and temperature effects.
    med = df.groupby(["year", "event", "driver"])["lap_time_s"].transform("median")
    df["delta_s"] = df["lap_time_s"] - med
    # Trim extreme outliers (traffic, small mistakes) at +-5 s.
    df = df[df["delta_s"].abs() <= 5.0]
    df["race_id"] = df["year"].astype(str) + " " + df["event"]
    return df


def _features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({
        "tyre_life": df["tyre_life"].astype(float),
        "track_temp": df["track_temp"].astype(float),
        "lap": df["lap"].astype(float),
        "compound_soft": (df["compound"] == "SOFT").astype(int),
        "compound_medium": (df["compound"] == "MEDIUM").astype(int),
    })
    return x


def train(stint_laps: pd.DataFrame) -> dict:
    df = build_training_frame(stint_laps)
    x = _features(df)
    y = df["delta_s"].values
    groups = df["race_id"].values

    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=7)

    gkf = GroupKFold(n_splits=5)
    maes, r2s = [], []
    for train_idx, test_idx in gkf.split(x, y, groups):
        m = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=7)
        m.fit(x.iloc[train_idx], y[train_idx])
        pred = m.predict(x.iloc[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        r2s.append(r2_score(y[test_idx], pred))

    baseline_mae = float(np.mean(np.abs(y - y.mean())))
    model.fit(x, y)
    joblib.dump(model, MODEL_FILE)

    metrics = {
        "model": "GradientBoostingRegressor",
        "target": "lap time delta vs driver median clean race pace (s)",
        "features": list(x.columns),
        "training_rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "validation": "GroupKFold(5) grouped by race",
        "cv_mae_s": round(float(np.mean(maes)), 3),
        "cv_mae_std": round(float(np.std(maes)), 3),
        "cv_r2": round(float(np.mean(r2s)), 3),
        "baseline_mae_s": round(baseline_mae, 3),
        "caveats": [
            "Dry green flag laps only; wet degradation is out of scope.",
            "Delta baseline absorbs car pace but traffic effects remain "
            "as noise the model cannot see.",
        ],
    }
    return metrics


def compound_deg_rates(stint_laps: pd.DataFrame) -> dict:
    """Per compound degradation rates from two views:

    raw_stint_slope_s_per_lap: median linear slope of raw lap time vs tire
      age inside each stint of 8+ clean laps. IMPORTANT: this mixes fuel
      burn (which makes the car faster each lap) with tire degradation, so
      it understates true degradation. Kept as a transparent diagnostic.

    model_deg_s_per_lap: mean partial derivative of the fitted model with
      respect to tyre_life, with lap number (fuel) held constant, averaged
      over that compound's real training rows. This is the fuel-separated
      estimate the undercut calculator uses as fallback."""
    df = build_training_frame(stint_laps)

    slopes = {c: [] for c in SLICKS}
    for _, stint in df.groupby(["year", "event", "driver", "stint"]):
        if len(stint) < 8:
            continue
        comp = stint["compound"].iloc[0]
        coeffs = np.polyfit(stint["tyre_life"], stint["lap_time_s"], 1)
        slope = float(coeffs[0])
        if -0.5 < slope < 1.0:
            slopes[comp].append(slope)

    model = joblib.load(MODEL_FILE)
    out = {}
    for comp in SLICKS:
        rows = df[df["compound"] == comp]
        sample = rows.sample(min(len(rows), 2000), random_state=7)
        x1 = _features(sample)
        x2 = x1.copy()
        x2["tyre_life"] = x2["tyre_life"] + 3
        partial = float(np.mean((model.predict(x2) - model.predict(x1)) / 3.0))
        raw = slopes[comp]
        out[comp] = {
            "deg_s_per_lap": round(max(partial, 0.0), 4),
            "raw_stint_slope_s_per_lap": round(float(np.median(raw)), 4) if raw else None,
            "raw_slope_note": "fuel confounded, understates degradation",
            "n_stints": len(raw),
            "n_laps": int(len(rows)),
        }
    return out


class DegradationModel:
    """Runtime wrapper used by the agents."""

    def __init__(self):
        self.model = joblib.load(MODEL_FILE)
        rates_file = ARTIFACTS_DIR / "compound_deg_rates.json"
        self.fallback_rates = json.loads(rates_file.read_text(encoding="utf-8")) \
            if rates_file.exists() else {}

    def predict_delta(self, tyre_life: float, compound: str,
                      track_temp: float, lap: float) -> float:
        x = pd.DataFrame([{
            "tyre_life": tyre_life,
            "track_temp": track_temp,
            "lap": lap,
            "compound_soft": 1 if compound == "SOFT" else 0,
            "compound_medium": 1 if compound == "MEDIUM" else 0,
        }])
        return float(self.model.predict(x)[0])

    def deg_rate(self, tyre_life: float, compound: str,
                 track_temp: float, lap: float) -> float:
        """Marginal seconds lost per additional lap of tire age, holding
        fuel constant. Finite difference over the fitted surface, clamped
        to the physically plausible range using the linear fallback."""
        d1 = self.predict_delta(tyre_life, compound, track_temp, lap)
        d2 = self.predict_delta(tyre_life + 3, compound, track_temp, lap)
        rate = (d2 - d1) / 3.0
        fb = self.fallback_rates.get(compound, {}).get("deg_s_per_lap")
        if fb is not None and not (0.0 <= rate <= 0.6):
            return max(float(fb), 0.0)
        return max(rate, 0.0)
