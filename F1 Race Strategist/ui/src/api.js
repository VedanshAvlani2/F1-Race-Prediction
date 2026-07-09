const BASE = "";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}

export const api = {
  sessions: () => get("/api/sessions"),
  sessionInfo: (year, event) =>
    get(`/api/session-info?year=${year}&event=${encodeURIComponent(event)}`),
  raceState: (year, event, lap, driver) =>
    get(`/api/race-state?year=${year}&event=${encodeURIComponent(event)}&lap=${lap}&driver=${driver}`),
  deliberate: async (body) => {
    const res = await fetch("/api/deliberate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  },
  standings: () => get("/api/standings"),
  metrics: () => get("/api/metrics"),
  replay: (year, event) =>
    get(`/api/replay?year=${year}&event=${encodeURIComponent(event)}`),
  preRaceGrid: () => get("/api/pre-race/grid"),
  preRaceCircuits: () => get("/api/pre-race/circuits"),
  preRacePlan: async (body) => {
    const res = await fetch("/api/pre-race/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return res.json();
  },
};
