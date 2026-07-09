import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function Metrics() {
  const [m, setM] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.metrics().then(setM).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!m) return <div className="panel">Loading model metrics...</div>;

  const deg = m.tire_degradation;
  const sc = m.safety_car;

  return (
    <div className="col">
      <div className="panel">
        <h2>Tire degradation model</h2>
        <div className="kv">
          <div><div className="k">CV MAE</div><div className="v">{deg.cv_mae_s}s</div></div>
          <div><div className="k">Baseline MAE</div><div className="v">{deg.baseline_mae_s}s</div></div>
          <div><div className="k">CV R2</div><div className="v">{deg.cv_r2}</div></div>
          <div><div className="k">Laps</div><div className="v">{deg.training_rows.toLocaleString()}</div></div>
          <div><div className="k">Races</div><div className="v">{deg.races}</div></div>
        </div>
        <div className="note">{deg.validation}. {deg.caveats.join(" ")}</div>
      </div>

      <div className="panel">
        <h2>Compound degradation rates (fuel separated)</h2>
        <table>
          <thead>
            <tr><th>Compound</th><th>s/lap</th><th>Raw stint slope</th><th>Stints</th><th>Laps</th></tr>
          </thead>
          <tbody>
            {Object.entries(m.compound_deg_rates).map(([c, r]) => (
              <tr key={c}>
                <td className={`compound ${c}`}>{c}</td>
                <td>{r.deg_s_per_lap}</td>
                <td>{r.raw_stint_slope_s_per_lap} <span className="note">(fuel confounded)</span></td>
                <td>{r.n_stints}</td><td>{r.n_laps?.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>
          Safety car model{" "}
          <span className="badge red">low confidence</span>
          {sc.beats_baseline === false && (
            <span className="badge yellow" style={{ marginLeft: 6 }}>
              does not beat base-rate baseline yet
            </span>
          )}
        </h2>
        <div className="kv">
          <div><div className="k">Brier (model)</div><div className="v">{sc.brier_model}</div></div>
          <div><div className="k">Brier (baseline)</div><div className="v">{sc.brier_baseline_predict_base_rate}</div></div>
          <div><div className="k">Races</div><div className="v">{sc.training_rows}</div></div>
          <div><div className="k">SC rate</div><div className="v">{sc.positive_rate}</div></div>
        </div>
        <div className="note">{sc.validation}. {sc.caveats.join(" ")}</div>
      </div>

      <div className="panel">
        <h2>Circuit pit loss (measured from real stops)</h2>
        <table>
          <thead>
            <tr><th>Circuit</th><th>Pit loss</th><th>Std</th><th>Stops</th><th>Races</th><th></th></tr>
          </thead>
          <tbody>
            {(m.pit_loss.rows || []).map((r) => (
              <tr key={r.event}>
                <td>{r.event}</td>
                <td>{r.pit_loss_s}s</td>
                <td>{r.pit_loss_std}</td>
                <td>{r.n_stops}</td>
                <td>{r.n_races}</td>
                <td>{r.low_confidence ? <span className="badge yellow">few samples</span> : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note">{m.pit_loss.method}</div>
      </div>
    </div>
  );
}
