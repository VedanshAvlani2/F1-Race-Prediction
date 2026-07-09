import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function Standings() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.standings().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="panel">Loading standings...</div>;

  const statusBadge =
    data.tracker_status === "updated" ? ["green", "new race recorded"]
    : data.tracker_status === "no_new_race" ? ["blue", "up to date, no new race"]
    : data.tracker_status === "offline_cached" ? ["yellow", "offline, cached data"]
    : ["red", data.tracker_note || "unavailable"];

  return (
    <div className="grid">
      <div className="panel">
        <h2>
          Drivers, season {data.season}{" "}
          <span className={`badge ${statusBadge[0]}`}>{statusBadge[1]}</span>
        </h2>
        {data.tracker_note && <div className="note">{data.tracker_note}</div>}
        <table>
          <thead>
            <tr><th>P</th><th>Driver</th><th>Team</th><th>Pts</th><th>Wins</th></tr>
          </thead>
          <tbody>
            {data.drivers.map((d) => (
              <tr key={d.position}>
                <td>{d.position}</td><td>{d.driver}</td>
                <td>{d.team}</td><td>{d.points}</td><td>{d.wins}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="panel">
        <h2>Constructors</h2>
        <table>
          <thead>
            <tr><th>P</th><th>Team</th><th>Pts</th><th>Wins</th></tr>
          </thead>
          <tbody>
            {(data.constructors || []).map((c) => (
              <tr key={c.position}>
                <td>{c.position}</td><td>{c.team}</td>
                <td>{c.points}</td><td>{c.wins}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
