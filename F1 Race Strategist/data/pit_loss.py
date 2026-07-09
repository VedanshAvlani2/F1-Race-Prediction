"""Circuit pit loss computed from real in-lap and out-lap data.

Method: for every pit stop under green flag conditions,
  pit_loss = (in_lap + out_lap) - 2 * driver_median_green_lap
The driver median baseline removes car pace, so what remains is the time
cost of driving through the pit lane plus the stationary stop. Stops under
SC or VSC are excluded because the field is slowed and the loss is not
representative. Per circuit value is the median across all valid stops in
all cached races, reported with the sample count.
"""

import numpy as np
import pandas as pd


def extract_pit_stops(session, year: int, event: str) -> pd.DataFrame:
    laps = session.laps
    if laps is None or len(laps) == 0:
        return pd.DataFrame()

    df = laps[[
        "Driver", "LapNumber", "LapTime", "PitInTime", "PitOutTime",
        "TrackStatus",
    ]].copy()
    df["lap_time_s"] = df["LapTime"].dt.total_seconds()
    df["TrackStatus"] = df["TrackStatus"].astype(str)

    rows = []
    for driver, d in df.groupby("Driver"):
        d = d.sort_values("LapNumber").reset_index(drop=True)
        green = d[
            (d["TrackStatus"] == "1")
            & d["PitInTime"].isna()
            & d["PitOutTime"].isna()
            & d["lap_time_s"].notna()
        ]["lap_time_s"]
        if len(green) < 5:
            continue
        baseline = green.median()

        for i in range(len(d) - 1):
            in_lap = d.iloc[i]
            out_lap = d.iloc[i + 1]
            if pd.isna(in_lap["PitInTime"]) or pd.isna(out_lap["PitOutTime"]):
                continue
            if pd.isna(in_lap["lap_time_s"]) or pd.isna(out_lap["lap_time_s"]):
                continue
            # Require green flag on both laps so SC pit windows do not
            # contaminate the estimate.
            if "4" in in_lap["TrackStatus"] or "4" in out_lap["TrackStatus"]:
                continue
            if "6" in in_lap["TrackStatus"] or "6" in out_lap["TrackStatus"]:
                continue
            loss = in_lap["lap_time_s"] + out_lap["lap_time_s"] - 2 * baseline
            if 5 < loss < 60:  # sanity bounds, discard damaged car outliers
                rows.append({
                    "year": year,
                    "event": event,
                    "driver": driver,
                    "in_lap": int(in_lap["LapNumber"]),
                    "pit_loss_s": round(float(loss), 3),
                })
    return pd.DataFrame(rows)


def circuit_pit_loss_table(all_stops: pd.DataFrame) -> pd.DataFrame:
    if all_stops.empty:
        return pd.DataFrame()
    g = all_stops.groupby("event")
    out = g.agg(
        pit_loss_s=("pit_loss_s", "median"),
        pit_loss_std=("pit_loss_s", lambda s: float(np.std(s))),
        n_stops=("pit_loss_s", "count"),
        n_races=("year", "nunique"),
    ).reset_index()
    out["pit_loss_s"] = out["pit_loss_s"].round(2)
    out["pit_loss_std"] = out["pit_loss_std"].round(2)
    out["low_confidence"] = out["n_stops"] < 15
    return out.sort_values("event")
