import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function SessionPicker({ onChange, onRun, busy }) {
  const [sessions, setSessions] = useState([]);
  const [year, setYear] = useState("");
  const [event, setEvent] = useState("");
  const [info, setInfo] = useState(null);
  const [lap, setLap] = useState(20);
  const [driver, setDriver] = useState("");

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

  useEffect(() => {
    if (!year || !event) return;
    api.sessionInfo(year, event).then((i) => {
      setInfo(i);
      if (i.drivers.length && !i.drivers.includes(driver)) {
        setDriver(i.drivers[0]);
      }
    });
  }, [year, event]);

  useEffect(() => {
    if (year && event && driver) {
      onChange({ year: Number(year), event, lap: Number(lap), driver });
    }
  }, [year, event, lap, driver]);

  const years = [...new Set(sessions.map((s) => s.year))];
  const events = sessions.filter((s) => String(s.year) === String(year));

  return (
    <div className="panel">
      <h2>Session</h2>
      <label>Season</label>
      <select value={year} onChange={(e) => setYear(e.target.value)}>
        {years.map((y) => (
          <option key={y} value={y}>{y}</option>
        ))}
      </select>

      <label>Grand Prix</label>
      <select value={event} onChange={(e) => setEvent(e.target.value)}>
        {events.map((s) => (
          <option key={s.event} value={s.event}>{s.event}</option>
        ))}
      </select>

      <label>Lap {info ? `(1-${info.total_laps})` : ""}</label>
      <input
        type="number"
        min={1}
        max={info ? info.total_laps : 78}
        value={lap}
        onChange={(e) => setLap(e.target.value)}
      />

      <label>Driver</label>
      <select value={driver} onChange={(e) => setDriver(e.target.value)}>
        {(info ? info.drivers : []).map((d) => (
          <option key={d} value={d}>{d}</option>
        ))}
      </select>

      <button className="primary" onClick={onRun} disabled={busy}>
        {busy ? "Agents deliberating..." : "Run strategy deliberation"}
      </button>
      <div className="note">
        Deterministic models compute every number; Groq agents debate and
        explain. Disagreements stay visible below.
      </div>
    </div>
  );
}
