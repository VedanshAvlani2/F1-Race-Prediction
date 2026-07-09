export default function RaceState({ state }) {
  const statusColor =
    state.track_status === "green" ? "green"
    : state.track_status === "safety car" || state.track_status === "red flag" ? "red"
    : "yellow";

  return (
    <div className="panel">
      <h2>
        Race state: {state.event} {state.year}, lap {state.lap}/{state.total_laps}{" "}
        <span className={`badge ${statusColor}`}>{state.track_status}</span>
      </h2>

      {state.weather && state.weather.track_temp !== undefined && (
        <div className="kv" style={{ marginBottom: 10 }}>
          <div><div className="k">Track</div><div className="v">{state.weather.track_temp.toFixed(0)}&deg;C</div></div>
          <div><div className="k">Air</div><div className="v">{state.weather.air_temp.toFixed(0)}&deg;C</div></div>
          <div><div className="k">Humidity</div><div className="v">{state.weather.humidity.toFixed(0)}%</div></div>
          <div><div className="k">Rain</div><div className="v">{state.weather.rainfall ? "YES" : "no"}</div></div>
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>P</th><th>Driver</th><th>Tire</th><th>Age</th>
            <th>Gap ahead</th><th>Last lap</th>
          </tr>
        </thead>
        <tbody>
          {state.order.map((e) => (
            <tr key={e.driver} className={e.driver === state.focus_driver ? "focus" : ""}>
              <td>{e.position}</td>
              <td>{e.driver}</td>
              <td className={`compound ${e.compound}`}>{e.compound}</td>
              <td>{e.tyre_age ?? "-"}</td>
              <td>{e.gap_ahead_s == null ? "LEADER" : `+${e.gap_ahead_s.toFixed(2)}s`}</td>
              <td>{e.lap_time_s ? e.lap_time_s.toFixed(3) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
