import { useEffect, useState } from "react";
import { api } from "../api.js";

function GridSlot({ entry, index, mode, onDragStart, onDragOver, onDrop,
                    dragging }) {
  const side = index % 2 === 0 ? "left" : "right";
  return (
    <div
      draggable={mode === "custom"}
      onDragStart={(e) => onDragStart(e, index)}
      onDragOver={(e) => { e.preventDefault(); onDragOver(index); }}
      onDrop={(e) => { e.preventDefault(); onDrop(index); }}
      style={{
        display: "flex", alignItems: "center", gap: 8,
        marginLeft: side === "right" ? "52%" : 0,
        width: "48%",
        background: dragging === index ? "rgba(225,6,0,.15)" : "var(--bg)",
        border: `1px solid ${dragging === index ? "var(--red)" : "var(--border)"}`,
        borderLeft: `4px solid ${entry.color || "#888"}`,
        borderRadius: 6, padding: "6px 10px", marginBottom: 4,
        cursor: mode === "custom" ? "grab" : "default",
        userSelect: "none",
      }}>
      <span style={{ fontFamily: "monospace", color: "var(--dim)",
                     width: 26, fontSize: 12 }}>P{entry.pos}</span>
      <span style={{ width: 12, height: 12, borderRadius: 3,
                     background: entry.color || "#888", flexShrink: 0 }} />
      <div style={{ minWidth: 0 }}>
        <div style={{ fontWeight: 600, fontSize: 13 }}>{entry.code}</div>
        <div style={{ fontSize: 11, color: "var(--dim)",
                      whiteSpace: "nowrap", overflow: "hidden",
                      textOverflow: "ellipsis" }}>
          {entry.team || entry.driver}
        </div>
      </div>
      {mode === "custom" && (
        <span style={{ marginLeft: "auto", color: "var(--dim)" }}>&#8942;</span>
      )}
    </div>
  );
}

export default function PreRace() {
  const [circuits, setCircuits] = useState([]);
  const [event, setEvent] = useState("");
  const [temp, setTemp] = useState(35);
  const [laps, setLaps] = useState(57);
  const [driver, setDriver] = useState("");
  const [mode, setMode] = useState("predicted");
  const [gridInfo, setGridInfo] = useState(null);
  const [grid, setGrid] = useState([]);
  const [dragFrom, setDragFrom] = useState(null);
  const [dragOverIdx, setDragOverIdx] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.preRaceCircuits().then((c) => {
      setCircuits(c);
      if (c.length && !event) setEvent(c[0].event);
    });
    api.preRaceGrid().then((g) => {
      setGridInfo(g);
      setGrid(g.grid || []);
      if (g.grid?.length) setDriver(g.grid[0].code);
    }).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!event) return;
    api.sessionInfo(
      circuits.find((c) => c.event === event)?.year, event
    ).then((i) => setLaps(i.total_laps)).catch(() => {});
  }, [event, circuits]);

  const renumber = (list) =>
    list.map((e, i) => ({ ...e, pos: i + 1 }));

  const onDrop = (toIdx) => {
    if (dragFrom == null || dragFrom === toIdx) {
      setDragFrom(null); setDragOverIdx(null); return;
    }
    const next = [...grid];
    const [moved] = next.splice(dragFrom, 1);
    next.splice(toIdx, 0, moved);
    setGrid(renumber(next));
    setDragFrom(null);
    setDragOverIdx(null);
  };

  const resetPredicted = () => {
    if (gridInfo) setGrid(gridInfo.grid || []);
  };

  const run = async () => {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const res = await api.preRacePlan({
        event,
        track_temp: Number(temp),
        total_laps: Number(laps),
        focus_driver: driver,
        grid: grid.map(({ pos, code }) => ({ pos, code })),
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const compoundChip = (c) => (
    <span key={c} className={`compound ${c}`} style={{ marginRight: 4 }}>
      {c[0]}
    </span>
  );

  return (
    <div className="grid" style={{ gridTemplateColumns: "380px 1fr" }}>
      <div className="col">
        <div className="panel">
          <h2>Race setup</h2>
          <label>Circuit</label>
          <select value={event} onChange={(e) => setEvent(e.target.value)}>
            {circuits.map((c) => (
              <option key={c.event} value={c.event}>{c.event}</option>
            ))}
          </select>
          <div style={{ display: "flex", gap: 10 }}>
            <div style={{ flex: 1 }}>
              <label>Forecast track temp (C)</label>
              <input type="number" value={temp} min={10} max={60}
                onChange={(e) => setTemp(e.target.value)} />
            </div>
            <div style={{ flex: 1 }}>
              <label>Race laps</label>
              <input type="number" value={laps} min={30} max={90}
                onChange={(e) => setLaps(e.target.value)} />
            </div>
          </div>
          <label>Focus driver</label>
          <select value={driver} onChange={(e) => setDriver(e.target.value)}>
            {grid.map((g) => (
              <option key={g.code} value={g.code}>{g.code}</option>
            ))}
          </select>
          <button className="primary" onClick={run} disabled={busy || !event}>
            {busy ? "Agents deliberating..." : "Generate strategy brief"}
          </button>
        </div>

        <div className="panel">
          <h2>
            Starting grid{" "}
            <span style={{ float: "right", display: "flex", gap: 4 }}>
              {["predicted", "custom"].map((m) => (
                <button key={m} onClick={() => {
                    setMode(m);
                    if (m === "predicted") resetPredicted();
                  }}
                  style={{
                    background: "var(--bg)", cursor: "pointer",
                    color: mode === m ? "var(--text)" : "var(--dim)",
                    border: mode === m ? "1px solid var(--red)"
                                       : "1px solid var(--border)",
                    borderRadius: 6, padding: "3px 10px", fontSize: 11,
                  }}>
                  {m}
                </button>
              ))}
            </span>
          </h2>
          {mode === "predicted" && gridInfo && (
            <div className="note" style={{ marginBottom: 8 }}>
              {gridInfo.note} Source: {gridInfo.source}.
            </div>
          )}
          {mode === "custom" && (
            <div className="note" style={{ marginBottom: 8 }}>
              Drag drivers into your qualifying order.
            </div>
          )}
          <div>
            {grid.map((entry, i) => (
              <GridSlot key={entry.code} entry={entry} index={i} mode={mode}
                dragging={dragOverIdx}
                onDragStart={(e, idx) => setDragFrom(idx)}
                onDragOver={(idx) => setDragOverIdx(idx)}
                onDrop={onDrop} />
            ))}
          </div>
        </div>
      </div>

      <div className="col">
        {error && <div className="error">{error}</div>}

        {result && (
          <>
            <div className="panel">
              <h2>Strategy brief: {result.plan_table.event}</h2>
              {result.brief && (
                <div className="rec">
                  <div className="big">
                    PLAN A: {result.brief.plan_a.sequence.map(compoundChip)}
                    {" "}pit {result.brief.plan_a.pit_laps.join(", ")}
                  </div>
                  <div className="sub">
                    fallback: {result.brief.plan_b.sequence.join("-")} pit{" "}
                    {result.brief.plan_b.pit_laps.join(", ")}
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <span className={`badge ${
                      result.brief.confidence === "high" ? "green"
                      : result.brief.confidence === "medium" ? "yellow" : "red"}`}>
                      confidence: {result.brief.confidence}
                    </span>
                  </div>
                  <ul className="reasons">
                    {result.brief.switch_triggers.map((t, i) => (
                      <li key={i}>{t}</li>
                    ))}
                  </ul>
                  {result.brief.skeptic_dissent && (
                    <div className="note">
                      Skeptic dissent on record: {result.brief.skeptic_dissent}
                    </div>
                  )}
                </div>
              )}
              {result.final_narrative && (
                <div className="narrative" style={{ whiteSpace: "pre-wrap",
                  color: "var(--dim)", fontSize: 13 }}>
                  {result.final_narrative}
                </div>
              )}
            </div>

            <div className="panel">
              <h2>Ranked plans (Monte Carlo, {result.plan_table.n_monte_carlo} sims)</h2>
              <table>
                <thead>
                  <tr>
                    <th>#</th><th>Plan</th><th>Pit laps</th>
                    <th>Clean race</th><th>MC mean</th>
                    <th>Gap</th><th>SC helps</th>
                  </tr>
                </thead>
                <tbody>
                  {result.plan_table.plans.map((p) => (
                    <tr key={p.rank}>
                      <td>{p.rank}</td>
                      <td>{p.sequence.map(compoundChip)}</td>
                      <td>{p.pit_laps.join(", ")}</td>
                      <td>+{p.clean_time_delta_s}s</td>
                      <td>+{p.mc_mean_time_delta_s}s</td>
                      <td>{p.gap_to_best_s > 0 ? `+${p.gap_to_best_s}s` : "best"}</td>
                      <td>{(p.p_sc_helps * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="note">
                Times are tire plus pit loss deltas, not full race times.{" "}
                {result.plan_table.assumptions.join(". ")}.
              </div>
            </div>

            <div className="panel">
              <h2>Agent deliberation</h2>
              {result.transcript.map((t, i) => (
                <div key={i} className={`turn ${t.node}`}>
                  <div className="who">{t.node}</div>
                  <div className="summary">{t.summary}</div>
                  {t.narrative && (
                    <div className="narrative">{t.narrative}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}

        {!result && !error && (
          <div className="panel">
            <h2>Pre-race strategy planner</h2>
            <div className="note">
              Set the circuit, forecast temperature and grid, then generate.
              The planner enumerates every legal 1 and 2 stop plan, prices
              each with the degradation model and measured pit loss, stress
              tests against safety car timing, and the agent panel writes
              the brief with visible dissent.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
