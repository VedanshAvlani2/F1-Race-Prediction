"""Race state assembly: reconstructs the live picture at any lap of a
cached session. Every field is measured data from FastF1; nothing is
estimated here."""

import pandas as pd

from data.track_status import status_at_lap


def build_race_state(session, year: int, event: str, lap: int,
                     focus_driver: str) -> dict:
    laps = session.laps
    total_laps = int(laps["LapNumber"].max())
    lap = max(1, min(lap, total_laps))

    at_lap = laps[laps["LapNumber"] == lap].copy()
    at_lap = at_lap.dropna(subset=["Position", "Time"])
    at_lap = at_lap.sort_values("Position")

    order = []
    times = at_lap.set_index("Driver")["Time"]
    for _, r in at_lap.iterrows():
        drv = r["Driver"]
        pos = int(r["Position"])
        entry = {
            "driver": drv,
            "position": pos,
            "compound": r["Compound"] if pd.notna(r["Compound"]) else "UNKNOWN",
            "tyre_age": float(r["TyreLife"]) if pd.notna(r["TyreLife"]) else None,
            "lap_time_s": round(r["LapTime"].total_seconds(), 3)
                          if pd.notna(r["LapTime"]) else None,
        }
        order.append(entry)

    # Gaps from cumulative session time at lap completion.
    for i, entry in enumerate(order):
        if i == 0:
            entry["gap_ahead_s"] = None
        else:
            t_self = times[entry["driver"]]
            t_ahead = times[order[i - 1]["driver"]]
            entry["gap_ahead_s"] = round((t_self - t_ahead).total_seconds(), 3)
    for i, entry in enumerate(order):
        entry["gap_behind_s"] = (order[i + 1]["gap_ahead_s"]
                                 if i + 1 < len(order) else None)

    weather_now = {}
    w = session.weather_data
    if w is not None and len(w) > 0 and len(at_lap) > 0:
        ref_time = at_lap["Time"].min()
        w_sorted = w.sort_values("Time")
        idx = (w_sorted["Time"] - ref_time).abs().idxmin()
        row = w_sorted.loc[idx]
        recent = w_sorted[w_sorted["Time"] <= ref_time].tail(20)
        weather_now = {
            "track_temp": float(row["TrackTemp"]),
            "air_temp": float(row["AirTemp"]),
            "humidity": float(row["Humidity"]),
            "rainfall": bool(row["Rainfall"]),
            "rain_last_20_samples": round(float(recent["Rainfall"].mean()), 3)
                                    if len(recent) else 0.0,
        }

    focus = next((e for e in order if e["driver"] == focus_driver), None)

    # Compounds already used by the focus driver up to this lap (for the
    # two compound rule check).
    used = laps[(laps["Driver"] == focus_driver) & (laps["LapNumber"] <= lap)]
    compounds_used = [c for c in used["Compound"].dropna().unique().tolist()]

    return {
        "year": year,
        "event": event,
        "lap": lap,
        "total_laps": total_laps,
        "laps_remaining": total_laps - lap,
        "track_status": status_at_lap(session, lap),
        "weather": weather_now,
        "focus_driver": focus_driver,
        "focus": focus,
        "compounds_used": compounds_used,
        "order": order,
    }
