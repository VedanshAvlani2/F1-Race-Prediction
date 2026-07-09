"""Dataset builder. Iterates every cached race and writes the processed
tables the models train on:

  store/stint_laps.csv        one row per lap with tire and weather context
  store/race_status.csv       one row per race with SC/VSC summary
  store/pit_stops.csv         one row per green flag pit stop
  store/circuit_pit_loss.csv  median pit loss per circuit with sample size
  store/dataset_manifest.json provenance: races processed, rows, failures

Run:  python -m data.build_datasets
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import STORE_DIR
from data.fastf1_client import list_cached_races, load_race
from data.pit_loss import circuit_pit_loss_table, extract_pit_stops
from data.stints import extract_stint_laps
from data.track_status import extract_race_status


def _race_key(year: int, event: str) -> str:
    return f"{year}_{event.replace(' ', '_')}"


def process_races(max_seconds: float = 1e9) -> bool:
    """Process cached races one at a time, writing per race outputs into
    store/raw/ so the build is resumable. Returns True when every race
    has been processed."""
    races = list_cached_races()
    if races.empty:
        print("No cached races found. Check F1_CACHE_DIR.")
        sys.exit(1)

    raw_dir = STORE_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = True

    for i, (_, race) in enumerate(races.iterrows(), start=1):
        year, event = int(race["year"]), race["event"]
        key = _race_key(year, event)
        status_file = raw_dir / f"{key}_status.json"
        if status_file.exists():
            continue
        if time.time() - t0 > max_seconds:
            done = False
            break
        try:
            session = load_race(year, event)
            laps = extract_stint_laps(session, year, event)
            status = extract_race_status(session, year, event)
            stops = extract_pit_stops(session, year, event)
            laps.to_csv(raw_dir / f"{key}_laps.csv", index=False)
            stops.to_csv(raw_dir / f"{key}_stops.csv", index=False)
            status_file.write_text(json.dumps(status), encoding="utf-8")
            print(f"[{i}/{len(races)}] {year} {event}: "
                  f"{len(laps)} laps, {len(stops)} stops, "
                  f"SC={status['sc_deploys']} VSC={status['vsc_deploys']}")
        except Exception as exc:
            status_file.write_text(
                json.dumps({"year": year, "event": event, "error": str(exc)}),
                encoding="utf-8")
            print(f"[{i}/{len(races)}] {year} {event}: FAILED ({exc})")
    return done


def merge() -> None:
    races = list_cached_races()
    raw_dir = STORE_DIR / "raw"
    all_laps, all_status, all_stops, failures = [], [], [], []
    t0 = time.time()

    for _, race in races.iterrows():
        year, event = int(race["year"]), race["event"]
        key = _race_key(year, event)
        status_file = raw_dir / f"{key}_status.json"
        if not status_file.exists():
            failures.append({"year": year, "event": event,
                             "error": "not processed"})
            continue
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if "error" in status:
            failures.append(status)
            continue
        all_status.append(status)
        laps_file = raw_dir / f"{key}_laps.csv"
        stops_file = raw_dir / f"{key}_stops.csv"
        if laps_file.exists() and laps_file.stat().st_size > 10:
            all_laps.append(pd.read_csv(laps_file))
        if stops_file.exists() and stops_file.stat().st_size > 10:
            all_stops.append(pd.read_csv(stops_file))

    laps_df = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    status_df = pd.DataFrame(all_status)
    stops_df = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame()
    pit_table = circuit_pit_loss_table(stops_df)

    laps_df.to_csv(STORE_DIR / "stint_laps.csv", index=False)
    status_df.to_csv(STORE_DIR / "race_status.csv", index=False)
    stops_df.to_csv(STORE_DIR / "pit_stops.csv", index=False)
    pit_table.to_csv(STORE_DIR / "circuit_pit_loss.csv", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "races_found": int(len(races)),
        "races_processed": int(len(races) - len(failures)),
        "failures": failures,
        "stint_lap_rows": int(len(laps_df)),
        "pit_stop_rows": int(len(stops_df)),
        "circuits_with_pit_loss": int(len(pit_table)),
        "merge_seconds": round(time.time() - t0, 1),
    }
    (STORE_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
    done = process_races(max_seconds=budget)
    if done:
        merge()
    else:
        print("TIME_BUDGET_REACHED (rerun to resume)")


if __name__ == "__main__":
    main()
