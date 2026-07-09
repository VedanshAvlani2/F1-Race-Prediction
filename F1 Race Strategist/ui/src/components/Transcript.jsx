const ROLE_LABELS = {
  data: "Data node",
  strategist: "Strategist (race strategist)",
  "strategist (revised)": "Strategist, revised proposal",
  risk: "Risk (team principal)",
  skeptic: "Skeptic (data scientist)",
  synthesis: "Synthesis (chief strategist)",
};

export default function Transcript({ transcript }) {
  return (
    <div className="panel">
      <h2>Agent deliberation transcript</h2>
      {transcript.map((t, i) => {
        const cls = t.node.startsWith("strategist") ? "strategist" : t.node;
        return (
          <div key={i} className={`turn ${cls}`}>
            <div className="who">{ROLE_LABELS[t.node] || t.node}</div>
            <div className="summary">{t.summary}</div>
            {t.narrative && <div className="narrative">{t.narrative}</div>}
            {t.numbers && (
              <div className="nums">{JSON.stringify(t.numbers)}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}
