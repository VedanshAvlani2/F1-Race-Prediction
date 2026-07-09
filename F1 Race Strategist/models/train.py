"""Trains every model and writes honest metrics to artifacts/metrics.json.

Run:  python -m models.train
Requires data/store/ tables built by data.build_datasets first.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ARTIFACTS_DIR, STORE_DIR
from models import safety_car, tire_degradation


def main():
    laps_file = STORE_DIR / "stint_laps.csv"
    status_file = STORE_DIR / "race_status.csv"
    if not laps_file.exists() or not status_file.exists():
        print("Datasets missing. Run: python -m data.build_datasets")
        sys.exit(1)

    stint_laps = pd.read_csv(laps_file)
    race_status = pd.read_csv(status_file)

    print(f"Training tire degradation model on {len(stint_laps)} raw lap rows...")
    deg_metrics = tire_degradation.train(stint_laps)
    print(json.dumps(deg_metrics, indent=2))

    print("Fitting per compound linear degradation rates...")
    rates = tire_degradation.compound_deg_rates(stint_laps)
    (ARTIFACTS_DIR / "compound_deg_rates.json").write_text(
        json.dumps(rates, indent=2), encoding="utf-8")
    print(json.dumps(rates, indent=2))

    print(f"Training safety car model on {len(race_status)} races...")
    sc_metrics = safety_car.train(race_status)
    print(json.dumps(sc_metrics, indent=2))

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "tire_degradation": deg_metrics,
        "compound_deg_rates": rates,
        "safety_car": sc_metrics,
        "pit_loss": {
            "method": "median of (in_lap + out_lap - 2 * driver median green "
                      "lap) over all green flag stops per circuit",
            "table": str(STORE_DIR / "circuit_pit_loss.csv"),
        },
        "undercut_calculator": "deterministic, no training (models/undercut.py)",
        "regulation_engine": "deterministic rule checks (models/regulations.py)",
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {ARTIFACTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
