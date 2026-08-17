"""
nexus_context.dashboard.ui
===========================
Generates the complete inline HTML/CSS/JS dashboard page.

Design: dark professional monospace layout, Inter font,
live SSE-fed metrics, no hardcoded values, no emojis.
"""

from __future__ import annotations

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nexus-Context | Observability Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base:     #0d1117;
    --bg-card:     #161b22;
    --bg-card-alt: #1f2937;
    --bg-input:    #21262d;
    --border:      #30363d;
    --accent:      #58a6ff;
    --accent-dim:  #1f3a5f;
    --green:       #3fb950;
    --yellow:      #d29922;
    --red:         #f85149;
    --text-primary:   #e6edf3;
    --text-secondary: #8b949e;
    --text-muted:     #484f58;
    --font-sans: 'Inter', system-ui, sans-serif;
    --font-mono: 'JetBrains Mono', 'Consolas', monospace;
    --radius: 6px;
    --transition: 180ms ease;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 14px; }
  body {
    background: var(--bg-base);
    color: var(--text-primary);
    font-family: var(--font-sans);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ---- Header ---- */
  header {
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: rgba(13, 17, 23, 0.92);
    backdrop-filter: blur(12px);
    z-index: 100;
  }
  .logo {
    font-weight: 600;
    font-size: 15px;
    letter-spacing: -0.3px;
    color: var(--text-primary);
  }
  .logo span { color: var(--accent); }
  .status-row { display: flex; align-items: center; gap: 18px; }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-mono);
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: background var(--transition);
  }
  .dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot.error { background: var(--red); }

  /* ---- Main layout ---- */
  main { flex: 1; padding: 24px 28px; display: flex; flex-direction: column; gap: 20px; }

  /* ---- Section label ---- */
  .section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
  }

  /* ---- Primary metric grid ---- */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color var(--transition);
  }
  .card:hover { border-color: var(--accent-dim); }
  .card-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    text-transform: uppercase;
  }
  .card-value {
    font-family: var(--font-mono);
    font-size: 26px;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1;
    transition: color 200ms;
  }
  .card-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
  }

  /* ---- Gauge: horizontal bar ---- */
  .gauge-bar-wrap {
    margin-top: 6px;
    height: 5px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .gauge-bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 400ms ease;
    width: 0%;
  }

  /* ---- Sparkline canvas ---- */
  .sparkline-wrap { margin-top: 4px; }
  canvas.sparkline {
    width: 100%;
    height: 44px;
    display: block;
  }

  /* ---- Bottom panels row ---- */
  .panels-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
  }
  @media (max-width: 900px) { .panels-row { grid-template-columns: 1fr; } }

  /* ---- Log panel ---- */
  .log-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .log-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .log-title {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: 0.3px;
  }
  .log-count {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-input);
    padding: 2px 8px;
    border-radius: 10px;
  }
  .log-body {
    overflow-y: auto;
    max-height: 280px;
    flex: 1;
  }
  .log-row {
    display: grid;
    grid-template-columns: 90px 1fr auto;
    align-items: start;
    gap: 8px;
    padding: 8px 16px;
    border-bottom: 1px solid rgba(48, 54, 61, 0.4);
    font-family: var(--font-mono);
    font-size: 11px;
    transition: background var(--transition);
  }
  .log-row:hover { background: var(--bg-card-alt); }
  .log-row:last-child { border-bottom: none; }
  .log-time { color: var(--text-muted); white-space: nowrap; }
  .log-content { color: var(--text-primary); word-break: break-all; }
  .log-badge {
    font-size: 10px;
    padding: 1px 7px;
    border-radius: 10px;
    white-space: nowrap;
    font-weight: 500;
  }
  .badge-blue  { background: rgba(88,166,255,0.15); color: var(--accent); }
  .badge-green { background: rgba(63,185,80,0.15);  color: var(--green); }
  .badge-yellow{ background: rgba(210,153,34,0.15); color: var(--yellow); }
  .badge-red   { background: rgba(248,81,73,0.15);  color: var(--red); }

  /* ---- Empty state ---- */
  .empty-state {
    padding: 32px 16px;
    text-align: center;
    color: var(--text-muted);
    font-size: 12px;
    font-family: var(--font-mono);
  }

  /* ---- Session list ---- */
  .session-list { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
  .session-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-input);
    border-radius: 4px;
    padding: 5px 10px;
    font-family: var(--font-mono);
    font-size: 11px;
  }
  .session-id { color: var(--accent); }
  .session-turns { color: var(--text-muted); }

  /* ---- Flicker animation on new data ---- */
  @keyframes value-flash { 0%,100%{opacity:1} 50%{opacity:0.4} }
  .flash { animation: value-flash 300ms ease; }

  /* ---- Footer ---- */
  footer {
    border-top: 1px solid var(--border);
    padding: 10px 28px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    display: flex;
    justify-content: space-between;
  }
</style>
</head>
<body>

<header>
  <div class="logo">nexus<span>-context</span> <span style="color:var(--text-muted);font-weight:300">/ observability</span></div>
  <div class="status-row">
    <div class="status-pill"><div class="dot" id="conn-dot"></div><span id="conn-label">connecting</span></div>
    <div class="status-pill" style="color:var(--text-muted)">port 9000</div>
  </div>
</header>

<main>
  <!-- Primary Metrics -->
  <div>
    <div class="section-label">Live Metrics</div>
    <div class="metric-grid">

      <div class="card">
        <div class="card-label">Active Sessions</div>
        <div class="card-value" id="m-sessions">-</div>
        <div class="session-list" id="session-list"></div>
      </div>

      <div class="card">
        <div class="card-label">Pipeline Latency (ms)</div>
        <div class="card-value" id="m-latency">-</div>
        <div class="card-sub" id="m-latency-sub">last request</div>
        <div class="sparkline-wrap"><canvas class="sparkline" id="sparkline-latency"></canvas></div>
      </div>

      <div class="card">
        <div class="card-label">Token Budget Used</div>
        <div class="card-value" id="m-tokens">-</div>
        <div class="card-sub" id="m-tokens-sub">tokens this turn</div>
        <div class="gauge-bar-wrap"><div class="gauge-bar-fill" id="token-bar"></div></div>
      </div>

      <div class="card">
        <div class="card-label">KV Cache Hit Rate</div>
        <div class="card-value" id="m-cache">-</div>
        <div class="card-sub">Zone P lock reuse</div>
        <div class="gauge-bar-wrap"><div class="gauge-bar-fill" id="cache-bar" style="background:var(--green)"></div></div>
      </div>

      <div class="card">
        <div class="card-label">Memory Pool (WWW Tuples)</div>
        <div class="card-value" id="m-memory">-</div>
        <div class="card-sub" id="m-memory-sub">active tuples</div>
        <div class="sparkline-wrap"><canvas class="sparkline" id="sparkline-memory"></canvas></div>
      </div>

      <div class="card">
        <div class="card-label">Graph Topology</div>
        <div class="card-value" id="m-nodes">-</div>
        <div class="card-sub" id="m-edges">- edges</div>
      </div>

    </div>
  </div>

  <!-- Bottom panels -->
  <div class="panels-row">

    <!-- Tool Call Interceptions -->
    <div class="log-panel">
      <div class="log-header">
        <span class="log-title">Tool Call Interceptions</span>
        <span class="log-count" id="tool-count">0</span>
      </div>
      <div class="log-body" id="tool-log">
        <div class="empty-state">No interceptions yet</div>
      </div>
    </div>

    <!-- Chunk Boundary Events -->
    <div class="log-panel">
      <div class="log-header">
        <span class="log-title">Chunk Boundary Events</span>
        <span class="log-count" id="chunk-count">0</span>
      </div>
      <div class="log-body" id="chunk-log">
        <div class="empty-state">No boundary events yet</div>
      </div>
    </div>

    <!-- Long-Term Knowledge Base -->
    <div class="log-panel">
      <div class="log-header">
        <span class="log-title">Long-Term Knowledge Base</span>
        <span class="log-count" id="ltkb-count">0 facts</span>
      </div>
      <div class="log-body" id="ltkb-log">
        <div class="empty-state">No facts persisted yet</div>
      </div>
    </div>

  </div>
</main>

<footer>
  <span id="footer-events">0 events received</span>
  <span id="footer-uptime">uptime: 0s</span>
</footer>

<script>
// =========================================================
// State
// =========================================================
const state = {
  sessions: {},         // session_id -> {turn_counter}
  latencyHistory: [],   // last 60 pipeline_ms values
  memoryHistory: [],    // last 60 memory_tuples counts
  cacheHits: 0,
  cacheMiss: 0,
  totalTurns: 0,
  totalEvents: 0,
  toolInterceptions: [],
  chunkBoundaries: [],
  ltkbFacts: [],
  startTime: Date.now(),
};

// =========================================================
// Sparkline renderer
// =========================================================
function drawSparkline(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth * dpr;
  const H = canvas.offsetHeight * dpr;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (data.length < 2) return;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.slice(-60);
  const step = W / (pts.length - 1);

  ctx.beginPath();
  ctx.strokeStyle = color || '#58a6ff';
  ctx.lineWidth = 1.5 * dpr;
  ctx.lineJoin = 'round';

  pts.forEach((v, i) => {
    const x = i * step;
    const y = H - ((v - min) / range) * H * 0.9 - H * 0.05;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under line
  ctx.lineTo((pts.length - 1) * step, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, (color || '#58a6ff') + '33');
  grad.addColorStop(1, (color || '#58a6ff') + '00');
  ctx.fillStyle = grad;
  ctx.fill();
}

// =========================================================
// DOM helpers
// =========================================================
function setText(id, val) {
  const el = document.getElementById(id);
  if (el && el.textContent !== String(val)) {
    el.textContent = val;
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }
}

function setBar(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min(100, Math.max(0, pct)) + '%';
}

function formatTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-GB', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function prependLogRow(containerId, counterId, cells, badge) {
  const container = document.getElementById(containerId);
  if (!container) return;
  // Remove empty state
  const empty = container.querySelector('.empty-state');
  if (empty) empty.remove();

  const row = document.createElement('div');
  row.className = 'log-row';
  row.style.opacity = '0';
  row.innerHTML = `
    <span class="log-time">${cells[0]}</span>
    <span class="log-content">${cells[1]}</span>
    <span class="log-badge ${badge}">${cells[2]}</span>
  `;
  container.insertBefore(row, container.firstChild);
  // Fade in
  requestAnimationFrame(() => { row.style.transition = 'opacity 300ms'; row.style.opacity = '1'; });

  // Cap to 50 rows
  const rows = container.querySelectorAll('.log-row');
  if (rows.length > 50) rows[rows.length - 1].remove();

  // Update count badge
  const countEl = document.getElementById(counterId);
  if (countEl) countEl.textContent = rows.length;
}

// =========================================================
// Render sessions list
// =========================================================
function renderSessions() {
  const list = document.getElementById('session-list');
  if (!list) return;
  list.innerHTML = '';
  const ids = Object.keys(state.sessions);
  setText('m-sessions', ids.length);
  if (ids.length === 0) return;
  ids.slice(-6).forEach(sid => {
    const item = document.createElement('div');
    item.className = 'session-item';
    item.innerHTML = `<span class="session-id">${sid.slice(0, 18)}${sid.length > 18 ? '...' : ''}</span><span class="session-turns">turn ${state.sessions[sid].turn_counter}</span>`;
    list.appendChild(item);
  });
}

// =========================================================
// Process SSE event
// =========================================================
function processEvent(ev) {
  state.totalEvents++;
  document.getElementById('footer-events').textContent = `${state.totalEvents} events received`;

  const type = ev.event_type;
  const sid = ev.session_id;

  if (type === 'session_created' || type === 'session_restored') {
    if (!state.sessions[sid]) state.sessions[sid] = { turn_counter: 0 };
    renderSessions();
  }

  if (type === 'turn_processed') {
    state.totalTurns++;
    if (!state.sessions[sid]) state.sessions[sid] = { turn_counter: 0 };
    state.sessions[sid].turn_counter = ev.turn_counter || 0;

    // Latency
    state.latencyHistory.push(ev.pipeline_ms || 0);
    setText('m-latency', (ev.pipeline_ms || 0).toFixed(1));
    document.getElementById('m-latency-sub').textContent = `turn ${ev.turn_counter || '?'} — session ${sid.slice(0,10)}`;
    drawSparkline('sparkline-latency', state.latencyHistory, '#58a6ff');

    // Token budget
    const budget = 4096;
    const used = ev.tokens_in || 0;
    setText('m-tokens', used);
    document.getElementById('m-tokens-sub').textContent = `/ ${budget} budget`;
    setBar('token-bar', (used / budget) * 100);

    // KV Cache hit rate
    if (ev.zone_p_hit) state.cacheHits++; else state.cacheMiss++;
    const total = state.cacheHits + state.cacheMiss;
    const rate = total > 0 ? ((state.cacheHits / total) * 100) : 0;
    setText('m-cache', rate.toFixed(0) + '%');
    setBar('cache-bar', rate);

    // Memory pool
    state.memoryHistory.push(ev.memory_tuples_count || 0);
    setText('m-memory', ev.memory_tuples_count || 0);
    document.getElementById('m-memory-sub').textContent = `compacted: ${ev.compaction_applied ? 'yes' : 'no'}`;
    drawSparkline('sparkline-memory', state.memoryHistory, '#3fb950');

    // Graph topology
    setText('m-nodes', ev.graph_node_count || 0);
    document.getElementById('m-edges').textContent = `${ev.graph_edge_count || 0} edges`;

    renderSessions();
  }

  if (type === 'tool_call_intercepted') {
    const saved = ev.tokens_saved || 0;
    prependLogRow(
      'tool-log', 'tool-count',
      [formatTime(ev.timestamp), `${sid.slice(0,14)} — saved ${saved} tokens`, `${ev.original_tokens} → ${ev.compressed_tokens}`],
      'badge-yellow'
    );
  }

  if (type === 'chunk_boundary') {
    const suppressed = ev.suppressed_by_syntax;
    prependLogRow(
      'chunk-log', 'chunk-count',
      [
        formatTime(ev.timestamp),
        `dS=${ev.cosine_shift?.toFixed(3)} H=${ev.token_entropy?.toFixed(3)} score=${ev.boundary_score?.toFixed(3)}`,
        suppressed ? 'suppressed' : (ev.is_boundary ? 'split' : 'pass'),
      ],
      suppressed ? 'badge-yellow' : (ev.is_boundary ? 'badge-blue' : 'badge-green')
    );
  }

  if (type === 'ltkb_fact_persisted') {
    const countEl = document.getElementById('ltkb-count');
    const cur = parseInt(countEl?.textContent || '0') + 1;
    if (countEl) countEl.textContent = `${cur} facts`;
    prependLogRow(
      'ltkb-log', 'ltkb-count',
      [
        formatTime(ev.timestamp),
        ev.content_preview || ev.fact_id || '—',
        `w=${ev.weight?.toFixed(3)}`,
      ],
      'badge-green'
    );
  }
}

// =========================================================
// SSE connection
// =========================================================
function connect() {
  const dot = document.getElementById('conn-dot');
  const label = document.getElementById('conn-label');
  dot.className = 'dot';
  label.textContent = 'connecting';

  const es = new EventSource('/dashboard/stream');

  es.onopen = () => {
    dot.className = 'dot live';
    label.textContent = 'live';
  };

  es.onmessage = (e) => {
    try {
      processEvent(JSON.parse(e.data));
    } catch (_) {}
  };

  es.onerror = () => {
    dot.className = 'dot error';
    label.textContent = 'reconnecting';
    es.close();
    setTimeout(connect, 3000);
  };
}

// =========================================================
// Uptime ticker
// =========================================================
setInterval(() => {
  const sec = Math.floor((Date.now() - state.startTime) / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const fmt = h > 0
    ? `${h}h ${m}m ${s}s`
    : m > 0 ? `${m}m ${s}s` : `${s}s`;
  document.getElementById('footer-uptime').textContent = `uptime: ${fmt}`;
}, 1000);

// =========================================================
// Boot
// =========================================================
connect();
</script>
</body>
</html>"""


def get_dashboard_html() -> str:
    """Return the complete dashboard HTML page."""
    return _DASHBOARD_HTML
