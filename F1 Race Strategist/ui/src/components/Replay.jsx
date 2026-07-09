import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const SPEEDS = [1, 5, 20, 60];
const STATUS_STYLE = {
  green: { badge: "green", label: "green flag", track: "#3d444d" },
  yellow: { badge: "yellow", label: "yellow flag", track: "#d29922" },
  "safety car": { badge: "yellow", label: "SAFETY CAR", track: "#d29922" },
  vsc: { badge: "yellow", label: "VIRTUAL SC", track: "#d29922" },
  "vsc ending": { badge: "yellow", label: "VSC ENDING", track: "#d29922" },
  "red flag": { badge: "red", label: "RED FLAG", track: "#e10600" },
};

export default function Replay() {
  const [sessions, setSessions] = useState([]);
  const [year, setYear] = useState("");
  const [event, setEvent] = useState("");
  const [rep, setRep] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(20);
  const [lap, setLap] = useState(1);
  const [status, setStatus] = useState("green");
  const [weatherRow, setWeatherRow] = useState(null);
  const [showLabels, setShowLabels] = useState(true);
  const [visibleCars, setVisibleCars] = useState(0);

  const idxRef = useRef(0);
  const rafRef = useRef(null);
  const lastTs = useRef(null);
  const dotRefs = useRef({});
  const labelRefs = useRef({});
  const playingRef = useRef(false);
  const speedRef = useRef(speed);
  const labelsRef = useRef(true);
  const repRef = useRef(null);

  useEffect(() => {
    api.sessions().then((s) => {
      setSessions(s);
      if (s.length) {
        const last = s[s.length - 1];
        setYear(String(last.year));
        setEvent(last.event);
      }
    });
  }, []);

  const years = [...new Set(sessions.map((s) => s.year))];
  const events = sessions.filter((s) => String(s.year) === String(year));

  const load = async () => {
    setLoading(true);
    setError("");
    setRep(null);
    setPlaying(false);
    cancelAnimationFrame(rafRef.current);
    lastTs.current = null;
    try {
      const r = await api.replay(Number(year), event);
      repRef.current = r;
      idxRef.current = r.lap_start_idx["1"] ?? 0;
      setRep(r);
      setLap(1);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const currentLap = (idx) => {
    const r = repRef.current;
    if (!r) return 1;
    let best = 1;
    for (const [l, i] of Object.entries(r.lap_start_idx)) {
      if (i <= idx && Number(l) > best) best = Number(l);
    }
    return best;
  };

  const currentStatus = (idx) => {
    const r = repRef.current;
    if (!r || !r.status_segments) return "green";
    for (const [a, b, name] of r.status_segments) {
      if (idx >= a && idx < b) return name;
    }
    return "green";
  };

  const currentWeather = (idx) => {
    const r = repRef.current;
    if (!r || !r.weather || !r.weather.length) return null;
    let row = r.weather[0];
    for (const w of r.weather) {
      if (w[0] > idx) break;
      row = w;
    }
    return row;
  };

  const inPit = (code, idx) => {
    const r = repRef.current;
    if (!r || !r.pit[code]) return false;
    return r.pit[code].some(([a, b]) => idx >= a && idx <= b);
  };

  const renderFrame = () => {
    const r = repRef.current;
    if (!r) return;
    const idx = idxRef.current;
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, r.n_samples - 1);
    const f = idx - i0;
    let visible = 0;
    for (const d of r.drivers) {
      const el = dotRefs.current[d.code];
      const lb = labelRefs.current[d.code];
      if (!el) continue;
      const x0 = r.x[d.code]?.[i0], x1 = r.x[d.code]?.[i1];
      const y0 = r.y[d.code]?.[i0], y1 = r.y[d.code]?.[i1];
      if (x0 == null || y0 == null || x1 == null || y1 == null) {
        el.setAttribute("opacity", "0");
        if (lb) lb.setAttribute("opacity", "0");
        continue;
      }
      visible += 1;
      const cx = x0 + (x1 - x0) * f;
      const cy = y0 + (y1 - y0) * f;
      el.setAttribute("opacity", "1");
      el.setAttribute("cx", cx);
      el.setAttribute("cy", cy);
      if (lb) {
        lb.setAttribute("opacity", labelsRef.current ? "1" : "0");
        lb.setAttribute("x", cx);
        lb.setAttribute("y", cy - 15);
      }
      if (inPit(d.code, i0)) {
        el.setAttribute("fill", "none");
        el.setAttribute("stroke", d.color);
        el.setAttribute("stroke-width", "3");
      } else {
        el.setAttribute("fill", d.color);
        el.setAttribute("stroke", "#0d1117");
        el.setAttribute("stroke-width", "1.5");
      }
    }
    setVisibleCars((prev) => (prev === visible ? prev : visible));
  };

  const tick = (ts) => {
    const r = repRef.current;
    if (!r) return;
    if (lastTs.current != null && playingRef.current) {
      const elapsed = (ts - lastTs.current) / 1000;
      idxRef.current = Math.min(
        idxRef.current + (elapsed * speedRef.current) / r.dt,
        r.n_samples - 1);
      if (idxRef.current >= r.n_samples - 1) setPlaying(false);
    }
    lastTs.current = ts;
    const idx = Math.floor(idxRef.current);
    const l = currentLap(idx);
    setLap((prev) => (prev === l ? prev : l));
    const st = currentStatus(idx);
    setStatus((prev) => (prev === st ? prev : st));
    const w = currentWeather(idx);
    setWeatherRow((prev) => (prev === w ? prev : w));
    renderFrame();
    rafRef.current = requestAnimationFrame(tick);
  };

  useEffect(() => {
    playingRef.current = playing;
    speedRef.current = speed;
    labelsRef.current = showLabels;
  }, [playing, speed, showLabels]);

  useEffect(() => {
    if (rep) rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [rep]);

  const scrubToLap = (l) => {
    const r = repRef.current;
    if (!r) return;
    idxRef.current = r.lap_start_idx[String(l)] ?? 0;
    setLap(l);
    renderFrame();
  };

  const board = rep ? rep.leaderboard[String(lap)] || [] : [];
  const trackPath = rep ? rep.track.map((p) => p.join(",")).join(" ") : "";
  const pitPath = rep && rep.pit_lane && rep.pit_lane.length
    ? rep.pit_lane.map((p) => p.join(",")).join(" ")
    : "";

  let viewBox = "-20 -20 1040 1040";
  if (rep) {
    const pts = rep.track.concat(rep.pit_lane || []);
    const xs2 = pts.map((p) => p[0]);
    const ys2 = pts.map((p) => p[1]);
    const minX = Math.min(...xs2) - 45, maxX = Math.max(...xs2) + 45;
    const minY = Math.min(...ys2) - 45, maxY = Math.max(...ys2) + 45;
    viewBox = `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;
  }

  const st = STATUS_STYLE[status] || STATUS_STYLE.green;

  return (
    <div>
      <div className="panel" style={{ marginBottom: 16 }}>
        <h2>Race replay</h2>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
          <div style={{ width: 120 }}>
            <label>Year</label>
            <select value={year} onChange={(e) => setYear(e.target.value)}>
              {years.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label>Race</label>
            <select value={event} onChange={(e) => setEvent(e.target.value)}>
              {events.map((s) => (
                <option key={s.event} value={s.event}>{s.event}</option>
              ))}
            </select>
          </div>
          <button className="primary" style={{ width: 180, marginTop: 0 }}
            onClick={load} disabled={loading}>
            {loading ? "Building replay..." : "Load replay"}
          </button>
        </div>
        <div className="note">
          Positions, flags and weather from real FastF1 telemetry at 1s
          resolution. Dot spacing is the actual time gap.
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {rep && (
        <div className="grid" style={{ gridTemplateColumns: "1fr 250px" }}>
          <div className="panel">
            <h2>
              {rep.event} {rep.year} &middot; lap {lap}/{rep.total_laps}{" "}
              <span className={`badge ${st.badge}`}>{st.label}</span>{" "}
              <span className="badge blue">{visibleCars} cars on track</span>
            </h2>

            {weatherRow && (
              <div className="kv" style={{ marginBottom: 8 }}>
                <div><div className="k">Track</div>
                  <div className="v">{weatherRow[1]}&deg;C</div></div>
                <div><div className="k">Air</div>
                  <div className="v">{weatherRow[2]}&deg;C</div></div>
                <div><div className="k">Humidity</div>
                  <div className="v">{weatherRow[3]}%</div></div>
                <div><div className="k">Wind</div>
                  <div className="v">{weatherRow[4]} m/s{" "}
                    <span style={{ display: "inline-block",
                      transform: `rotate(${weatherRow[5]}deg)` }}>&#8593;</span>
                  </div></div>
                <div><div className="k">Rain</div>
                  <div className="v">{weatherRow[6] ? "YES" : "no"}</div></div>
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "center" }}>
              <svg viewBox={viewBox} preserveAspectRatio="xMidYMid meet"
                style={{ maxHeight: "62vh", maxWidth: "100%" }}>
                {pitPath && (
                  <>
                    <polyline points={pitPath} fill="none" stroke="#2b3138"
                      strokeWidth="10" strokeLinejoin="round"
                      strokeLinecap="round" />
                    <text x={rep.pit_lane[Math.floor(rep.pit_lane.length / 2)][0]}
                      y={rep.pit_lane[Math.floor(rep.pit_lane.length / 2)][1] + 26}
                      fill="#8b949e" fontSize="15" textAnchor="middle">
                      pit lane
                    </text>
                  </>
                )}
                <polyline points={trackPath} fill="none" stroke={st.track}
                  strokeWidth="22" strokeLinejoin="round" strokeLinecap="round" />
                <polyline points={trackPath} fill="none" stroke="#161b22"
                  strokeWidth="14" strokeLinejoin="round" strokeLinecap="round" />
                <circle cx={rep.track[0][0]} cy={rep.track[0][1]} r="6"
                  fill="#e6edf3" />
                {rep.drivers.map((d) => (
                  <g key={d.code}>
                    <circle r="10" opacity="0"
                      ref={(el) => { dotRefs.current[d.code] = el; }} />
                    <text opacity="0" fontSize="12" fontFamily="monospace"
                      textAnchor="middle" fill={d.color}
                      ref={(el) => { labelRefs.current[d.code] = el; }}>
                      {d.code}
                    </text>
                  </g>
                ))}
              </svg>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 10,
                          borderTop: "1px solid var(--border)", paddingTop: 10 }}>
              <button className="primary"
                style={{ width: 90, marginTop: 0, padding: "8px" }}
                onClick={() => setPlaying(!playing)}>
                {playing ? "Pause" : "Play"}
              </button>
              <input type="range" min={1} max={rep.total_laps} value={lap}
                style={{ flex: 1 }}
                onChange={(e) => scrubToLap(Number(e.target.value))} />
              <span style={{ fontFamily: "monospace", fontSize: 12 }}>
                lap {lap}
              </span>
              {SPEEDS.map((s) => (
                <button key={s} onClick={() => setSpeed(s)}
                  style={{
                    background: "var(--bg)", cursor: "pointer",
                    color: speed === s ? "var(--text)" : "var(--dim)",
                    border: speed === s ? "1px solid var(--red)"
                                        : "1px solid var(--border)",
                    borderRadius: 6, padding: "4px 8px", fontSize: 12,
                  }}>
                  {s}x
                </button>
              ))}
              <label style={{ display: "flex", gap: 4, alignItems: "center",
                              margin: 0, fontSize: 12, color: "var(--dim)" }}>
                <input type="checkbox" checked={showLabels}
                  style={{ width: "auto" }}
                  onChange={(e) => setShowLabels(e.target.checked)} />
                labels
              </label>
            </div>
          </div>

          <div className="panel">
            <h2>Leaderboard, lap {lap}</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: 3,
                          fontFamily: "monospace", fontSize: 12 }}>
              {board.map((row) => {
                const d = rep.drivers.find((x) => x.code === row.code);
                const pit = inPit(row.code, Math.floor(idxRef.current));
                return (
                  <div key={row.code}
                    style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ color: "var(--dim)", width: 18 }}>
                      {row.pos}
                    </span>
                    <span style={{
                      width: 10, height: 10, borderRadius: "50%",
                      display: "inline-block",
                      background: pit ? "transparent" : (d?.color || "#888"),
                      border: pit ? `2px solid ${d?.color || "#888"}` : "none",
                    }} />
                    <span>{row.code}</span>
                    <span style={{ marginLeft: "auto",
                                   color: pit ? "var(--yellow)" : "var(--dim)" }}>
                      {pit ? "IN PIT" : row.pos === 1 ? "LEADER"
                             : `+${row.gap.toFixed(1)}s`}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="note">
              Hollow ring = in pit lane. Cars missing from the board have
              retired or not been classified on this lap.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
