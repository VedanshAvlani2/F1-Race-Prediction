export default function Recommendation({ rec, narrative }) {
  if (!rec) return null;
  const confColor =
    rec.confidence === "high" ? "green"
    : rec.confidence === "medium" ? "yellow" : "red";

  return (
    <div className="panel">
      <h2>Final recommendation</h2>
      <div className="rec">
        <div className="big">
          {rec.pit_lap ? `BOX LAP ${rec.pit_lap}` : "STAY OUT"}
        </div>
        <div className="sub">
          window {rec.window?.[0]}-{rec.window?.[1]} &middot; onto{" "}
          <span className={`compound ${rec.compound}`}>{rec.compound}</span>
        </div>
        <div style={{ marginTop: 10 }}>
          <span className={`badge ${confColor}`}>
            confidence: {rec.confidence}
          </span>
        </div>
        <ul className="reasons">
          {rec.confidence_reasons.map((r, i) => <li key={i}>{r}</li>)}
          {rec.adjustments.map((a, i) => <li key={`a${i}`}>{a}</li>)}
        </ul>
        {rec.agent_positions?.skeptic?.disagrees && (
          <div className="note">
            Skeptic dissent on record: {rec.agent_positions.skeptic.reason}
          </div>
        )}
      </div>
      {narrative && <div className="narrative">{narrative}</div>}
    </div>
  );
}
