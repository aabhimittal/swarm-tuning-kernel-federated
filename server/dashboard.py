"""Self-contained live dashboard for the swarm.

Served at ``/`` so the Hugging Face Space is something you can *watch* rather
than only curl. No CDN, no build step, no external requests — one inline HTML
string that polls the server's own JSON endpoints. Renders the loss curve as
hand-built SVG so there is no charting dependency to ship to the Space.
"""

from __future__ import annotations

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm Parameter Server</title>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --line:#30363d; --fg:#e6edf3;
    --muted:#8b949e; --accent:#f0b132; --good:#3fb950; --bad:#f85149;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de; --fg:#1f2328;
            --muted:#656d76; --accent:#bf8700; --good:#1a7f37; --bad:#cf222e; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5
         ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:1100px; margin:0 auto; padding:24px 20px 60px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--muted); margin-bottom:22px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .k { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .v { font-size:22px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }
  section { margin-top:26px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em;
       color:var(--muted); margin:0 0 10px; font-weight:600; }
  .scroll { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th,td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th:first-child,td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; font-size:12px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px;
          background:var(--line); font-size:12px; }
  .live { color:var(--good); } .idle { color:var(--muted); }
  svg { display:block; width:100%; height:220px; }
  button { background:var(--accent); color:#000; border:0; border-radius:6px;
           padding:8px 14px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  pre { background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:12px; white-space:pre-wrap; word-break:break-word; margin:10px 0 0;
        max-height:260px; overflow:auto; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  input { background:var(--panel); border:1px solid var(--line); color:var(--fg);
          border-radius:6px; padding:8px 10px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🐝 Swarm Parameter Server</h1>
  <div class="sub">Gradient accumulation over HTTP — a software interconnect in place of NVLink.</div>

  <div class="grid" id="stats"></div>

  <section>
    <h2>Global loss</h2>
    <div class="card"><svg id="chart" viewBox="0 0 800 220" preserveAspectRatio="none"></svg></div>
  </section>

  <section>
    <h2>Swarm contributors</h2>
    <div class="card scroll"><table id="workers">
      <thead><tr>
        <th>Worker</th><th>Accepted</th><th>Steps</th><th>Local steps</th>
        <th>Stale</th><th>Compression</th><th>Uploaded</th><th>Last loss</th><th>Idle</th>
      </tr></thead><tbody></tbody>
    </table></div>
  </section>

  <section>
    <h2>Sample the global model</h2>
    <div class="card">
      <div class="row">
        <input id="prompt" placeholder="prompt (optional)" size="28">
        <input id="ntok" type="number" value="200" min="1" max="1000" size="5">
        <button id="gen">Generate</button>
      </div>
      <pre id="out">The model starts from random weights — early samples are noise. Watch them
become English as the swarm trains.</pre>
    </div>
  </section>
</div>

<script>
const $ = s => document.querySelector(s);
const fmtBytes = n => {
  if (!n) return "0 B";
  const u = ["B","KB","MB","GB","TB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), u.length - 1);
  return (n / Math.pow(1024, i)).toFixed(i ? 1 : 0) + " " + u[i];
};
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function card(k, v) { return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`; }

function drawChart(hist) {
  const svg = $("#chart");
  if (!hist.length) { svg.innerHTML = `<text x="400" y="110" fill="#8b949e"
      text-anchor="middle" font-size="13">waiting for the first global step…</text>`; return; }
  const W = 800, H = 220, P = 30;
  const ys = hist.map(h => h.loss);
  const lo = Math.min(...ys), hi = Math.max(...ys), span = (hi - lo) || 1;
  const x = i => P + (hist.length === 1 ? 0 : i * (W - 2 * P) / (hist.length - 1));
  const y = v => H - P - ((v - lo) / span) * (H - 2 * P);
  const pts = hist.map((h, i) => `${x(i).toFixed(1)},${y(h.loss).toFixed(1)}`).join(" ");
  const area = `${P},${H - P} ${pts} ${x(hist.length - 1).toFixed(1)},${H - P}`;
  svg.innerHTML = `
    <polygon points="${area}" fill="var(--accent)" opacity="0.12"></polygon>
    <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"></polyline>
    <text x="4" y="${P}" fill="#8b949e" font-size="11">${hi.toFixed(3)}</text>
    <text x="4" y="${H - P + 4}" fill="#8b949e" font-size="11">${lo.toFixed(3)}</text>
    <text x="${W - 4}" y="${H - 8}" fill="#8b949e" font-size="11"
      text-anchor="end">v${hist[hist.length - 1].version}</text>`;
}

async function tick() {
  try {
    const [s, w, h] = await Promise.all([
      fetch("status").then(r => r.json()),
      fetch("workers").then(r => r.json()),
      fetch("history").then(r => r.json()),
    ]);
    const t = w.summary || {};
    $("#stats").innerHTML = [
      card("Global version", s.version),
      card("Optimizer steps", s.step),
      card("Last loss", s.last_loss == null ? "—" : s.last_loss.toFixed(4)),
      card("Active workers", `<span class="${t.active_workers ? "live" : "idle"}">${t.active_workers ?? 0}</span>`),
      card("Pending", `${s.pending}/${s.world_size}`),
      card("Mode", `<span class="pill">${s.mode} · ${s.rule}</span>`),
      card("Parameters", (s.num_params || 0).toLocaleString()),
      card("Bandwidth saved", `${fmtBytes(t.bytes_saved || 0)} <span class="k">${t.compression_ratio || 1}x</span>`),
    ].join("");

    $("#workers tbody").innerHTML = (w.workers || []).map(r => `<tr>
      <td>${esc(r.worker_id)}</td><td>${r.accepted}</td><td>${r.steps_triggered}</td>
      <td>${r.local_steps}</td><td>${r.stale}</td><td>${r.compression_ratio}x</td>
      <td>${fmtBytes(r.wire_bytes)}</td>
      <td>${r.last_loss == null ? "—" : r.last_loss.toFixed(4)}</td>
      <td class="${r.idle_s < 60 ? "live" : "idle"}">${r.idle_s.toFixed(0)}s</td></tr>`).join("")
      || `<tr><td colspan="9" style="color:var(--muted)">no workers yet — start one and it appears here</td></tr>`;

    drawChart(h.history || []);
  } catch (e) { /* transient fetch failure; next tick retries */ }
}

$("#gen").onclick = async () => {
  const b = $("#gen"); b.disabled = true; $("#out").textContent = "sampling…";
  try {
    const q = new URLSearchParams({ prompt: $("#prompt").value, tokens: $("#ntok").value });
    const r = await fetch("generate?" + q);
    const j = await r.json();
    $("#out").textContent = r.ok ? j.text : (j.detail || "generation failed");
  } catch (e) { $("#out").textContent = "generation failed: " + e; }
  b.disabled = false;
};

tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"""
