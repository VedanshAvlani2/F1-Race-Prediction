"""Race replay builder.

Produces a compact replay JSON for a cached race from REAL data only:
  track outline   X/Y trace of the fastest race lap
  car positions   per driver X/Y from FastF1 position telemetry,
                  resampled onto a shared 1 second time grid, so the
                  distance between dots on screen is the real time gap
  pit intervals   from measured PitInTime/PitOutTime, used to style dots
                  while the car is physically in the pit lane (the X/Y
                  trace itself already follows the real pit lane path)
  leaderboard     per lap positions and gaps from timing data

Telemetry transponder dropouts are linearly interpolated; the fraction of
interpolated samples per driver is reported in the file, not hidden.

First build per race takes seconds when the position data is already in
the FastF1 cache (it is, for every cached race). The result is stored in
data/store/replays/ and served instantly afterwards.
"""

import json
import pickle

import numpy as np
import pandas as pd

from config.settings import CACHE_DIR, STORE_DIR
from data.fastf1_client import load_race

REPLAY_DIR = STORE_DIR / "replays"
DT = 1.0          # seconds between samples
SCALE = 1000      # coordinates normalised to a 0..SCALE square
FORMAT = 3        # bump to invalidate previously built replay files

# Track status codes from F1 live timing.
STATUS_NAMES = {"1": "green", "2": "yellow", "4": "safety car",
                "5": "red flag", "6": "vsc", "7": "vsc ending"}


def _pos_from_cache(year: int, event: str):
    """Fast path: read the position telemetry pickle straight from the
    FastF1 cache without triggering a full telemetry parse."""
    ydir = CACHE_DIR / str(year)
    if not ydir.exists():
        return None
    want = event.replace(" ", "_")
    for event_dir in ydir.iterdir():
        parts = event_dir.name.split("_", 1)
        if len(parts) == 2 and parts[1] == want:
            for race_dir in event_dir.glob("*_Race"):
                f = race_dir / "position_data.ff1pkl"
                if f.exists():
                    try:
                        return pickle.loads(f.read_bytes())["data"]
                    except Exception:
                        return None
    return None


def _pit_intervals(laps: pd.DataFrame, driver_code: str, t0: float) -> list:
    d = laps[laps["Driver"] == driver_code].sort_values("LapNumber")
    ins = d["PitInTime"].dropna().dt.total_seconds().tolist()
    outs = d["PitOutTime"].dropna().dt.total_seconds().tolist()
    intervals = []
    for t_in in ins:
        t_out = next((o for o in outs if o > t_in), t_in + 30.0)
        intervals.append([round((t_in - t0) / DT), round((t_out - t0) / DT)])
    return intervals


def _leaderboard(laps: pd.DataFrame) -> dict:
    board = {}
    for lap_no, at_lap in laps.groupby("LapNumber"):
        at_lap = at_lap.dropna(subset=["Position", "Time"]).sort_values("Position")
        if at_lap.empty:
            continue
        t_leader = at_lap["Time"].iloc[0]
        rows = []
        for _, r in at_lap.iterrows():
            rows.append({
                "code": r["Driver"],
                "pos": int(r["Position"]),
                "gap": round((r["Time"] - t_leader).total_seconds(), 2),
            })
        board[int(lap_no)] = rows
    return board


def build_replay(year: int, event: str, force: bool = False) -> dict:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPLAY_DIR / f"{year}_{event.replace(' ', '_')}.json"
    if out_file.exists() and not force:
        cached = json.loads(out_file.read_text(encoding="utf-8"))
        if cached.get("format") == FORMAT:
            return cached

    session = load_race(year, event, weather=True, messages=True)
    laps = session.laps
    results = session.results

    pos = _pos_from_cache(year, event)
    if pos is None:
        # Slow path: full telemetry parse through the public API.
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        pos = session.pos_data

    info = {}
    for _, r in results.iterrows():
        color = str(r.get("TeamColor") or "888888").lstrip("#")
        info[str(r["DriverNumber"])] = {
            "code": r["Abbreviation"],
            "team": r.get("TeamName", ""),
            "color": f"#{color}" if len(color) == 6 else "#888888",
        }

    # Clean per driver frames and find the shared time span. The Time
    # column repeats in blocks (coarse session clock), so dedupe leaves
    # roughly 1 Hz coverage, which matches the 1 s replay grid. The raw
    # ~4.5 Hz rows are kept separately for the track outline.
    frames = {}
    raw_frames = {}
    t_starts, t_ends = [], []
    for num, df in pos.items():
        num = str(num)
        if num not in info:
            continue
        d = df[((df["X"] != 0) | (df["Y"] != 0))].copy()
        if len(d) < 100:
            continue
        d["t"] = d["Time"].dt.total_seconds()
        d = d.sort_values("t")
        raw_frames[num] = d
        d = d.drop_duplicates(subset="t")
        frames[num] = d
        t_starts.append(d["t"].iloc[0])
        t_ends.append(d["t"].iloc[-1])
    if not frames:
        raise RuntimeError(f"No position telemetry in cache for {year} {event}")

    t0, t1 = min(t_starts), max(t_ends)
    grid = np.arange(t0, t1, DT)

    # Global bounds for normalisation, from all cars.
    all_x = np.concatenate([f["X"].values for f in frames.values()])
    all_y = np.concatenate([f["Y"].values for f in frames.values()])
    minx, maxx = float(all_x.min()), float(all_x.max())
    miny, maxy = float(all_y.min()), float(all_y.max())
    span = max(maxx - minx, maxy - miny)

    def norm_x(v):
        return np.round(np.nan_to_num((v - minx) / span * SCALE)).astype(int)

    def norm_y(v):
        return np.round(np.nan_to_num(SCALE - (v - miny) / span * SCALE)).astype(int)

    drivers, xs, ys, pits, interp_frac = [], {}, {}, {}, {}
    for num, d in frames.items():
        code = info[num]["code"]
        gx = np.interp(grid, d["t"], d["X"], left=np.nan, right=np.nan)
        gy = np.interp(grid, d["t"], d["Y"], left=np.nan, right=np.nan)
        # Coverage: fraction of grid points more than 2s from a real sample.
        nearest = np.searchsorted(d["t"].values, grid)
        nearest = np.clip(nearest, 1, len(d) - 1)
        dist = np.minimum(
            np.abs(grid - d["t"].values[nearest - 1]),
            np.abs(grid - d["t"].values[nearest]))
        interp_frac[code] = round(float((dist > 2.0).mean()), 4)

        xi = norm_x(gx).astype(object)
        yi = norm_y(gy).astype(object)
        nan_mask = np.isnan(gx) | np.isnan(gy)
        xi[nan_mask] = None
        yi[nan_mask] = None
        xs[code] = xi.tolist()
        ys[code] = yi.tolist()
        pits[code] = _pit_intervals(laps, code, t0)
        drivers.append({"code": code, "team": info[num]["team"],
                        "color": info[num]["color"]})

    # Track outline from the fastest race lap, using the raw ~4.5 Hz rows
    # (consecutive duplicate positions dropped) for corner fidelity.
    fastest = laps.pick_fastest()
    fl_drv = str(fastest["DriverNumber"])
    fl_start = fastest["LapStartTime"].total_seconds()
    fl_end = fastest["Time"].total_seconds()
    fd = raw_frames.get(fl_drv, next(iter(raw_frames.values())))
    lapd = fd[(fd["t"] >= fl_start) & (fd["t"] <= fl_end)]
    lapd = lapd[(lapd["X"].diff() != 0) | (lapd["Y"].diff() != 0)]
    step = max(1, len(lapd) // 350)
    lapd = lapd.iloc[::step]
    track = list(map(list, zip(norm_x(lapd["X"].values).tolist(),
                               norm_y(lapd["Y"].values).tolist())))

    # Real pit lane path: a car's telemetry trace between crossing the pit
    # entry line (PitInTime) and the pit exit line (PitOutTime) IS the pit
    # lane. Use the longest available stop for the cleanest trace.
    pit_lane = []
    best = None
    for _, lp in laps.dropna(subset=["PitInTime"]).iterrows():
        num = str(lp["DriverNumber"])
        if num not in raw_frames:
            continue
        t_in = lp["PitInTime"].total_seconds()
        nxt = laps[(laps["Driver"] == lp["Driver"])
                   & (laps["LapNumber"] == lp["LapNumber"] + 1)]
        if nxt.empty or pd.isna(nxt["PitOutTime"].iloc[0]):
            continue
        t_out = nxt["PitOutTime"].iloc[0].total_seconds()
        if 10 < t_out - t_in < 60 and (best is None or t_out - t_in > best[2] - best[1]):
            best = (num, t_in, t_out)
    if best is not None:
        num, t_in, t_out = best
        fd = raw_frames[num]
        # Extend past the entry/exit line crossings so the ramps that
        # connect the pit lane to the track are part of the drawn path.
        trace = fd[(fd["t"] >= t_in - 10) & (fd["t"] <= t_out + 10)]
        trace = trace[(trace["X"].diff() != 0) | (trace["Y"].diff() != 0)]
        step = max(1, len(trace) // 200)
        trace = trace.iloc[::step]
        pit_lane = list(map(list, zip(norm_x(trace["X"].values).tolist(),
                                      norm_y(trace["Y"].values).tolist())))

    # Track status segments mapped onto the sample grid.
    status_segments = []
    ts = session.track_status
    if ts is not None and len(ts) > 0:
        rows = ts.copy()
        rows["t"] = rows["Time"].dt.total_seconds()
        rows = rows.sort_values("t").reset_index(drop=True)
        for i, r in rows.iterrows():
            code = str(r["Status"])
            start = max(int((r["t"] - t0) / DT), 0)
            end = int((rows["t"].iloc[i + 1] - t0) / DT) if i + 1 < len(rows) \
                else int(len(grid))
            if end > start:
                status_segments.append(
                    [start, end, STATUS_NAMES.get(code, "green")])

    # Weather samples mapped onto the grid (roughly one per minute).
    weather = []
    wd = session.weather_data
    if wd is not None and len(wd) > 0:
        for _, r in wd.iterrows():
            idx = int((r["Time"].total_seconds() - t0) / DT)
            if idx < 0 or idx >= len(grid):
                continue
            weather.append([
                idx,
                round(float(r["TrackTemp"]), 1),
                round(float(r["AirTemp"]), 1),
                round(float(r["Humidity"]), 0),
                round(float(r["WindSpeed"]), 1),
                round(float(r["WindDirection"]), 0),
                1 if bool(r["Rainfall"]) else 0,
            ])

    # Lap number -> first sample index, from each lap's earliest start time.
    lap_start_idx = {}
    for lap_no, g in laps.groupby("LapNumber"):
        starts = g["LapStartTime"].dropna()
        if starts.empty:
            continue
        idx = int((starts.min().total_seconds() - t0) / DT)
        lap_start_idx[int(lap_no)] = max(idx, 0)

    replay = {
        "format": FORMAT,
        "year": year,
        "event": event,
        "total_laps": int(laps["LapNumber"].max()),
        "dt": DT,
        "n_samples": int(len(grid)),
        "track": track,
        "pit_lane": pit_lane,
        "status_segments": status_segments,
        "weather": weather,
        "drivers": sorted(drivers, key=lambda d: d["code"]),
        "x": xs,
        "y": ys,
        "pit": pits,
        "lap_start_idx": lap_start_idx,
        "leaderboard": _leaderboard(laps),
        "interpolated_fraction": interp_frac,
        "source": "FastF1 position telemetry, 1s resample, "
                  "linear interpolation over transponder gaps",
    }
    out_file.write_text(json.dumps(replay, separators=(",", ":")),
                        encoding="utf-8")
    return replay
