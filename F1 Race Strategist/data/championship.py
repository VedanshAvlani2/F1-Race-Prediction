"""Live championship tracker.

On each run:
  1. Pull current season driver and constructor standings via the jolpica
     Ergast mirror (through fastf1.ergast).
  2. Compare the latest completed round against the last cached pull.
  3. If a new race has happened, append it to standings_history.json.
  4. If nothing new, no-op. If the network is unavailable, return the last
     cached standings marked stale=True instead of erroring.

Rate limits: jolpica allows 4 req/s, 500 req/h. This module makes at most
2 requests per run, so limits are never a concern.
"""

import json
from datetime import datetime, timezone

from config.settings import CURRENT_SEASON, STORE_DIR

STANDINGS_FILE = STORE_DIR / "standings.json"
HISTORY_FILE = STORE_DIR / "standings_history.json"


def _read_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
    return default


def _fetch_live(season: int) -> dict:
    """Fetch standings from the Ergast mirror. Raises on network failure."""
    from fastf1.ergast import Ergast

    ergast = Ergast()
    drivers_resp = ergast.get_driver_standings(season=season)
    constructors_resp = ergast.get_constructor_standings(season=season)

    ddesc = drivers_resp.description
    dcontent = drivers_resp.content[0] if drivers_resp.content else None
    ccontent = constructors_resp.content[0] if constructors_resp.content else None
    if dcontent is None or dcontent.empty:
        raise RuntimeError(f"No standings returned for season {season}")

    latest_round = int(ddesc.iloc[0]["round"]) if "round" in ddesc.columns else 0

    drivers = [
        {
            "position": int(r["position"]),
            "driver": f"{r['givenName']} {r['familyName']}",
            "code": r.get("driverCode") or "",
            "team": (r["constructorNames"][0] if isinstance(r.get("constructorNames"), list) and r["constructorNames"] else ""),
            "points": float(r["points"]),
            "wins": int(r["wins"]),
        }
        for _, r in dcontent.iterrows()
    ]
    constructors = []
    if ccontent is not None and not ccontent.empty:
        constructors = [
            {
                "position": int(r["position"]),
                "team": r["constructorName"],
                "points": float(r["points"]),
                "wins": int(r["wins"]),
            }
            for _, r in ccontent.iterrows()
        ]

    return {
        "season": season,
        "round": latest_round,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "drivers": drivers,
        "constructors": constructors,
    }


def update_standings(season: int = CURRENT_SEASON) -> dict:
    """Main entry point. Returns standings plus tracker metadata:
    status is one of: updated, no_new_race, offline_cached, unavailable."""
    cached = _read_json(STANDINGS_FILE, None)

    try:
        live = _fetch_live(season)
    except Exception as exc:  # network down or API unavailable
        if cached:
            cached["tracker_status"] = "offline_cached"
            cached["stale"] = True
            cached["tracker_note"] = f"Live fetch failed ({type(exc).__name__}). Serving last cached pull."
            return cached
        return {
            "tracker_status": "unavailable",
            "stale": True,
            "tracker_note": "No network and no cached standings exist yet. "
                            "Run once with connectivity to seed the cache.",
            "season": season,
            "drivers": [],
            "constructors": [],
        }

    if cached and cached.get("season") == season and cached.get("round") == live["round"]:
        live["tracker_status"] = "no_new_race"
        live["stale"] = False
        STANDINGS_FILE.write_text(json.dumps(live, indent=2), encoding="utf-8")
        return live

    # New race since last pull: persist and append to history.
    live["tracker_status"] = "updated"
    live["stale"] = False
    STANDINGS_FILE.write_text(json.dumps(live, indent=2), encoding="utf-8")

    history = _read_json(HISTORY_FILE, [])
    known = {(h.get("season"), h.get("round")) for h in history}
    if (season, live["round"]) not in known:
        history.append({
            "season": season,
            "round": live["round"],
            "recorded_at": live["fetched_at"],
            "drivers_top10": live["drivers"][:10],
            "constructors": live["constructors"],
        })
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return live


def get_history() -> list:
    return _read_json(HISTORY_FILE, [])
