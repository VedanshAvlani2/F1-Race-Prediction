"""Downloads any missing race sessions for the given seasons into the
shared FastF1 cache. Each race is fetched once, then lives on disk
forever. Needs internet; respects FastF1's built-in rate limiting.

Run:  python -m data.fetch_seasons 2022 2023
Then: python -m data.build_datasets
      python -m models.train
"""

import sys
from pathlib import Path

import pandas as pd
from fastf1.exceptions import RateLimitExceededError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fastf1_client import get_schedule, list_cached_races, load_race

RATE_LIMIT_MSG = (
    "\nFastF1 rate limit reached (500 API calls/hour, shared across all "
    "endpoints). Everything fetched so far is cached permanently.\n"
    "Wait about an hour, rerun the same command, and it resumes where it "
    "stopped. Already cached races are skipped instantly."
)


def main():
    seasons = [int(a) for a in sys.argv[1:]] or [2022, 2023]
    cached = list_cached_races()
    have = set(zip(cached["year"], cached["event"])) if not cached.empty else set()

    for year in seasons:
        try:
            schedule = get_schedule(year)
        except RateLimitExceededError:
            print(RATE_LIMIT_MSG)
            sys.exit(0)
        # Only completed championship rounds (round 0 is testing).
        races = schedule[schedule["RoundNumber"] > 0]
        for _, r in races.iterrows():
            event = r["EventName"]
            if pd.Timestamp(r["EventDate"]) > pd.Timestamp.now():
                continue
            if (year, event) in have:
                print(f"skip   {year} {event} (cached)")
                continue
            try:
                print(f"fetch  {year} {event} ...", flush=True)
                load_race(year, event)
                print(f"done   {year} {event}")
            except RateLimitExceededError:
                print(RATE_LIMIT_MSG)
                sys.exit(0)
            except Exception as exc:
                print(f"FAILED {year} {event}: {exc}")


if __name__ == "__main__":
    main()
