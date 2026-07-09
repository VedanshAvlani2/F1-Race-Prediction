import { useEffect, useState } from "react";
import { api } from "./api.js";
import SessionPicker from "./components/SessionPicker.jsx";
import RaceState from "./components/RaceState.jsx";
import Transcript from "./components/Transcript.jsx";
import Recommendation from "./components/Recommendation.jsx";
import Standings from "./components/Standings.jsx";
import Metrics from "./components/Metrics.jsx";
import Replay from "./components/Replay.jsx";
import PreRace from "./components/PreRace.jsx";

export default function App() {
  const [tab, setTab] = useState("strategy");
  const [selection, setSelection] = useState(null); // {year,event,lap,driver}
  const [raceState, setRaceState] = useState(null);
  const [deliberation, setDeliberation] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!selection) return;
    setError("");
    api
      .raceState(selection.year, selection.event, selection.lap, selection.driver)
      .then(setRaceState)
      .catch((e) => setError(String(e)));
  }, [selection]);

  const runDeliberation = async () => {
    if (!selection) return;
    setBusy(true);
    setError("");
    setDeliberation(null);
    try {
      const res = await api.deliberate(selection);
      setDeliberation(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>F1 STRATEGY ENGINE</h1>
        <span className="sub">
          multi-agent pit strategy over validated models
        </span>
      </header>

      <div className="tabs">
        {["strategy", "pre-race", "replay", "championship", "models"].map((t) => (
          <button
            key={t}
            className={tab === t ? "active" : ""}
            onClick={() => setTab(t)}
          >
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}

      {tab === "strategy" && (
        <div className="grid">
          <div className="col">
            <SessionPicker
              onChange={setSelection}
              onRun={runDeliberation}
              busy={busy}
            />
            {deliberation && (
              <Recommendation rec={deliberation.recommendation}
                narrative={deliberation.final_narrative} />
            )}
          </div>
          <div className="col">
            {raceState && <RaceState state={raceState} />}
            {deliberation && (
              <Transcript transcript={deliberation.transcript} />
            )}
          </div>
        </div>
      )}

      {tab === "pre-race" && <PreRace />}
      {tab === "replay" && <Replay />}
      {tab === "championship" && <Standings />}
      {tab === "models" && <Metrics />}
    </div>
  );
}
