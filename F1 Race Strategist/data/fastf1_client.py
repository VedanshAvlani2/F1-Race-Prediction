"""Thin wrapper around FastF1.

FastF1 endpoints used by this project:
  fastf1.get_event_schedule(year)      season schedule (github f1schedule + ergast fallback)
  fastf1.get_session(...).load()       laps, weather_data, track_status, race control
                                       messages (livetiming.formula1.com static archive)
  fastf1.ergast.Ergast                 race results and championship standings
                                       (jolpica-f1 mirror of the retired Ergast API)

Rate limits and caching:
  jolpica enforces 4 req/s and 500 req/h burst limits. livetiming.formula1.com
  is an unauthenticated static archive but large (tens of MB per session).
  fastf1.Cache.enable_cache() persists every HTTP response to disk, so any
  session is downloaded at most once. This project reuses the parent repo's
  existing 3.6 GB cache and works fully offline for cached seasons.
"""

import logging
import warnings
from pathlib import Path

import fastf1
import pandas as pd

from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)

_initialised = False


def setup_cache() -> None:
    """Enable the shared FastF1 disk cache and silence noisy retry logs."""
    global _initialised
    if _initialised:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    for name in ("fastf1", "fastf1.req", "fastf1.core", "fastf1.api", "fastf1._api"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="fastf1")
    _initialised = True


def list_cached_races() -> pd.DataFrame:
    """Scan the cache directory for race sessions available offline.

    Returns a DataFrame with columns: year, event, date.
    """
    setup_cache()
    rows = []
    for year_dir in sorted(CACHE_DIR.glob("[0-9][0-9][0-9][0-9]")):
        year = int(year_dir.name)
        for event_dir in sorted(year_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            race_dirs = list(event_dir.glob("*_Race"))
            if not race_dirs:
                continue
            # Directory format: 2023-03-05_Bahrain_Grand_Prix
            parts = event_dir.name.split("_", 1)
            if len(parts) != 2:
                continue
            date, raw_name = parts
            rows.append(
                {"year": year, "event": raw_name.replace("_", " "), "date": date}
            )
    return pd.DataFrame(rows)


def load_race(year: int, event: str, laps: bool = True, weather: bool = True,
              messages: bool = True, telemetry: bool = False):
    """Load a race session. Telemetry is off by default because the strategy
    models work from lap level data; per sample telemetry loading is slow
    and only needed for the race state viewer."""
    setup_cache()
    session = fastf1.get_session(year, event, "R")
    session.load(laps=laps, telemetry=telemetry, weather=weather, messages=messages)
    return session


def get_schedule(year: int) -> pd.DataFrame:
    setup_cache()
    schedule = fastf1.get_event_schedule(year)
    return schedule[["RoundNumber", "EventName", "EventDate", "EventFormat"]]
