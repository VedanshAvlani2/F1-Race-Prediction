"""Tire stint and lap level extraction.

Produces the training table for the degradation model: one row per lap with
compound, tire age, track temperature and a set of quality flags so the
model trainer can filter to representative racing laps.
"""

import numpy as np
import pandas as pd


def extract_stint_laps(session, year: int, event: str) -> pd.DataFrame:
    """One row per completed lap with tire and weather context.

    Quality flags:
      is_accurate    FastF1 lap integrity flag
      green_lap      track status was all clear for the whole lap
      is_box_lap     in-lap or out-lap (pit time set)
      wet            rainfall reported during the lap
    """
    laps = session.laps
    if laps is None or len(laps) == 0:
        return pd.DataFrame()

    df = laps[[
        "Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound",
        "TyreLife", "FreshTyre", "TrackStatus", "IsAccurate",
        "PitInTime", "PitOutTime", "Time", "Position",
    ]].copy()

    df = df.dropna(subset=["LapTime", "Compound", "TyreLife"])
    df["lap_time_s"] = df["LapTime"].dt.total_seconds()
    df["is_box_lap"] = df["PitInTime"].notna() | df["PitOutTime"].notna()
    df["green_lap"] = df["TrackStatus"] == "1"

    weather = session.weather_data
    if weather is not None and len(weather) > 0:
        w = weather[["Time", "TrackTemp", "AirTemp", "Rainfall", "Humidity"]].copy()
        w = w.sort_values("Time")
        d = df.sort_values("Time")
        merged = pd.merge_asof(d, w, on="Time", direction="nearest")
        df = merged
    else:
        df["TrackTemp"] = np.nan
        df["AirTemp"] = np.nan
        df["Rainfall"] = False
        df["Humidity"] = np.nan

    df["wet"] = df["Rainfall"].fillna(False).astype(bool)
    df["year"] = year
    df["event"] = event

    out = df[[
        "year", "event", "Driver", "Team", "LapNumber", "Stint", "Compound",
        "TyreLife", "FreshTyre", "lap_time_s", "Position", "TrackTemp",
        "AirTemp", "Humidity", "wet", "green_lap", "is_box_lap", "IsAccurate",
    ]].rename(columns={
        "Driver": "driver", "Team": "team", "LapNumber": "lap",
        "Stint": "stint", "Compound": "compound", "TyreLife": "tyre_life",
        "FreshTyre": "fresh", "Position": "position",
        "TrackTemp": "track_temp", "AirTemp": "air_temp",
        "Humidity": "humidity", "IsAccurate": "is_accurate",
    })
    return out


def stint_summary(stint_laps: pd.DataFrame) -> pd.DataFrame:
    """Per stint aggregate: compound, length, mean pace. Used by the UI and
    for sanity checks, not model training."""
    if stint_laps.empty:
        return pd.DataFrame()
    g = stint_laps.groupby(["year", "event", "driver", "stint"])
    out = g.agg(
        compound=("compound", "first"),
        laps=("lap", "count"),
        first_lap=("lap", "min"),
        last_lap=("lap", "max"),
        mean_lap_s=("lap_time_s", "mean"),
        mean_track_temp=("track_temp", "mean"),
    ).reset_index()
    return out
