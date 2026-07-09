"""Safety car and VSC event extraction from FastF1 track status data.

Track status codes (F1 live timing):
  1 all clear, 2 yellow, 4 safety car deployed, 5 red flag,
  6 virtual safety car deployed, 7 virtual safety car ending.
"""

import pandas as pd

from config.settings import SC_CODE, VSC_CODES, RED_FLAG_CODE


def _count_deploys(status_series: pd.Series, code: str) -> int:
    """Count transitions into a given status code."""
    active = status_series == code
    return int((active & ~active.shift(1, fill_value=False)).sum())


def extract_race_status(session, year: int, event: str) -> dict:
    """Race level summary of interruptions plus lap counts under SC/VSC."""
    ts = session.track_status
    laps = session.laps
    total_laps = int(laps["LapNumber"].max()) if laps is not None and len(laps) else 0

    row = {
        "year": year,
        "event": event,
        "total_laps": total_laps,
        "sc_deploys": 0,
        "vsc_deploys": 0,
        "red_flags": 0,
        "any_sc": False,
        "any_vsc": False,
        "first_sc_lap": None,
        "laps_under_sc": 0,
        "laps_under_vsc": 0,
    }

    if ts is not None and len(ts) > 0:
        status = ts["Status"].astype(str)
        row["sc_deploys"] = _count_deploys(status, SC_CODE)
        row["vsc_deploys"] = _count_deploys(status, VSC_CODES[0])
        row["red_flags"] = _count_deploys(status, RED_FLAG_CODE)
        row["any_sc"] = row["sc_deploys"] > 0
        row["any_vsc"] = row["vsc_deploys"] > 0

    if laps is not None and len(laps) > 0:
        lap_status = laps[["LapNumber", "TrackStatus"]].dropna()
        lap_status["TrackStatus"] = lap_status["TrackStatus"].astype(str)
        sc_laps = lap_status[lap_status["TrackStatus"].str.contains(SC_CODE)]
        vsc_laps = lap_status[
            lap_status["TrackStatus"].str.contains(VSC_CODES[0])
        ]
        row["laps_under_sc"] = int(sc_laps["LapNumber"].nunique())
        row["laps_under_vsc"] = int(vsc_laps["LapNumber"].nunique())
        if not sc_laps.empty:
            row["first_sc_lap"] = int(sc_laps["LapNumber"].min())

    return row


def status_at_lap(session, lap_number: int) -> str:
    """Human readable track status for a given lap, for the race state view."""
    laps = session.laps
    if laps is None or len(laps) == 0:
        return "unknown"
    on_lap = laps[laps["LapNumber"] == lap_number]["TrackStatus"].dropna().astype(str)
    joined = "".join(on_lap.unique())
    if SC_CODE in joined:
        return "safety car"
    if VSC_CODES[0] in joined or VSC_CODES[1] in joined:
        return "virtual safety car"
    if RED_FLAG_CODE in joined:
        return "red flag"
    if "2" in joined:
        return "yellow"
    return "green"
