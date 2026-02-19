#!/usr/bin/env python3
from __future__ import annotations


from dotenv import load_dotenv
load_dotenv()

import os, json, glob, time, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RUNS_DIR = Path("runs")
APP_NAME = "covariant-stdlib-web"

def latest_file(pattern: str) -> str | None:
    cands = sorted(glob.glob(str(RUNS_DIR / pattern)))
    return cands[-1] if cands else None

def run_cmd(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode, out.strip()

def json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2).encode("utf-8")

INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>covariant — picks console</title>
  <style>
    :root{
      /* Pro palette: Black + Neon Green + Electric Blue */
      --bg0:#05060a;
      --bg1:#070a12;
      --panel: rgba(10,14,26,.72);
      --card: rgba(12,18,34,.72);
      --card2: rgba(10,16,30,.62);

      --text:#e9f0ff;
      --muted:#97a6c7;
      --muted2:#6f7ea2;

      --line: rgba(255,255,255,.08);
      --line2: rgba(255,255,255,.12);

      --blue:#2f7dff;       /* electric-ish blue */
      --blue2:#1b4cff;
      --green:#39ff88;      /* neon green */
      --green2:#16d86a;

      --danger:#ff4d6d;
      --warn:#ffb020;

      --shadow: 0 18px 45px rgba(0,0,0,.55);
      --shadow2: 0 10px 24px rgba(0,0,0,.40);

      --r:18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }

    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:var(--sans);
      color:var(--text);
      min-height:100vh;

      /* deep black base + subtle neon glows */
      background:
        radial-gradient(900px 520px at 12% 2%, rgba(47,125,255,.18), transparent 55%),
        radial-gradient(800px 520px at 88% 8%, rgba(57,255,136,.14), transparent 55%),
        radial-gradient(700px 420px at 45% 110%, rgba(47,125,255,.10), transparent 60%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }

    .wrap{max-width:1040px; margin:0 auto; padding:22px 16px 56px;}

    .topbar{
      display:flex; align-items:flex-end; justify-content:space-between; gap:14px;
      margin-bottom:14px;
    }
    .title{
      font-size:40px; font-weight:900; letter-spacing:.2px;
      margin:6px 0 2px;
    }
    .subtitle{
      margin:0; color:var(--muted); font-size:14px;
    }
    .badge{
      font-family:var(--mono);
      font-size:12px;
      padding:8px 10px;
      border-radius:999px;
      border:1px solid rgba(47,125,255,.35);
      background: rgba(8,12,24,.55);
      color: rgba(233,240,255,.92);
      box-shadow: var(--shadow2);
      white-space:nowrap;
    }

    .panel{
      background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
      border:1px solid var(--line);
      border-radius: var(--r);
      box-shadow: var(--shadow);
      padding:14px;
      position:relative;
      overflow:hidden;
    }
    .panel:before{
      content:"";
      position:absolute; inset:-2px;
      background:
        radial-gradient(380px 180px at 15% 10%, rgba(47,125,255,.18), transparent 60%),
        radial-gradient(340px 160px at 92% 25%, rgba(57,255,136,.16), transparent 60%);
      pointer-events:none;
      filter: blur(8px);
      opacity:.75;
    }
    .panel > *{position:relative}

    .grid{
      display:grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap:10px;
      margin-top:8px;
    }
    @media (max-width:780px){
      .grid{grid-template-columns:1fr 1fr;}
      .title{font-size:34px}
      .topbar{align-items:flex-start; flex-direction:column}
    }
    @media (max-width:520px){ .grid{grid-template-columns:1fr;} }

    label{display:block; font-size:12px; color:var(--muted); margin:0 0 6px; letter-spacing:.25px}
    input, select{
      width:100%;
      padding:12px 12px;
      border-radius:14px;
      border:1px solid var(--line2);
      background: rgba(7,10,20,.58);
      color:var(--text);
      outline:none;
      font-size:16px;
    }
    input:focus, select:focus{
      border-color: rgba(47,125,255,.55);
      box-shadow: 0 0 0 3px rgba(47,125,255,.16);
    }

    .controls{
      display:grid;
      grid-template-columns: 1.2fr 1fr 1fr;
      gap:10px;
      margin-top:10px;
      align-items:end;
    }
    @media (max-width:780px){ .controls{grid-template-columns:1fr;} }

    .btnrow{display:flex; flex-wrap:wrap; gap:10px; margin-top:12px}
    button{
      border:1px solid rgba(255,255,255,.12);
      color:var(--text);
      padding:12px 14px;
      border-radius: 14px;
      font-weight:850;
      letter-spacing:.2px;
      cursor:pointer;
      box-shadow: var(--shadow2);
      transition: transform .06s ease, filter .15s ease, border-color .15s ease;
      background: linear-gradient(180deg, rgba(20,30,60,.9), rgba(12,18,40,.88));
    }
    button:hover{filter: brightness(1.06); border-color: rgba(47,125,255,.28)}
    button:active{transform: translateY(1px)}

    .primary{
      border-color: rgba(47,125,255,.35);
      background:
        linear-gradient(180deg, rgba(47,125,255,.26), rgba(12,18,40,.92));
    }
    .accent{
      border-color: rgba(57,255,136,.30);
      background:
        linear-gradient(180deg, rgba(57,255,136,.18), rgba(12,18,40,.92));
    }
    .ghost{
      background: rgba(7,10,20,.42);
    }

    .status{
      margin-top:10px;
      padding:10px 12px;
      border-radius: 14px;
      border:1px dashed rgba(255,255,255,.14);
      background: rgba(0,0,0,.18);
      color:rgba(233,240,255,.80);
      font-family: var(--mono);
      font-size: 12px;
      white-space: nowrap;
      overflow:hidden;
      text-overflow: ellipsis;
    }
    .hint{margin-top:10px; color:var(--muted2); font-size:12px}

    h2{
      margin:22px 0 10px;
      font-size:13px;
      letter-spacing:.26em;
      color:rgba(151,166,199,.85);
    }

    .cards{display:grid; grid-template-columns:1fr 1fr; gap:12px}
    @media (max-width:780px){ .cards{grid-template-columns:1fr} }

    .card{
      background: linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
      border:1px solid rgba(255,255,255,.09);
      border-radius: var(--r);
      padding:14px;
      box-shadow: var(--shadow2);
      position:relative;
      overflow:hidden;
    }
    .card:before{
      content:"";
      position:absolute; inset:-1px;
      background:
        radial-gradient(240px 120px at 85% 0%, rgba(47,125,255,.16), transparent 60%),
        radial-gradient(200px 120px at 10% 90%, rgba(57,255,136,.12), transparent 60%);
      opacity:.65;
      filter: blur(10px);
      pointer-events:none;
    }
    .card > *{position:relative}

    .match{font-size:18px; font-weight:900; line-height:1.15}
    .meta{margin-top:6px; color:var(--muted); font-size:13px}

    .row{display:flex; gap:10px; flex-wrap:wrap; margin-top:12px}
    .chip{
      background: rgba(6,10,20,.55);
      border:1px solid rgba(255,255,255,.10);
      border-radius: 999px;
      padding:7px 10px;
      font-size:12px;
      color:rgba(233,240,255,.78);
      font-family: var(--mono);
    }

    .pill{
      float:right;
      font-family: var(--mono);
      font-size:12px;
      padding:7px 10px;
      border-radius:999px;
      background: rgba(0,0,0,.22);
      border:1px solid rgba(255,255,255,.12);
    }
    .good{color:var(--green); border-color: rgba(57,255,136,.32)}
    .warn{color: var(--warn); border-color: rgba(255,176,32,.35)}
    .bad{color: var(--danger); border-color: rgba(255,77,109,.35)}

    details{
      margin-top:16px;
      background: rgba(0,0,0,.18);
      border:1px solid rgba(255,255,255,.09);
      border-radius: var(--r);
      padding:12px 12px 2px;
    }
    summary{
      cursor:pointer;
      color:rgba(151,166,199,.95);
      font-weight:900;
      letter-spacing:.06em;
      margin:0 0 10px;
    }
    pre{
      background: rgba(0,0,0,.35);
      border:1px solid rgba(255,255,255,.10);
      border-radius: 14px;
      padding:12px;
      overflow:auto;
      color:#dbe6ff;
      font-family: var(--mono);
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .toggle{
      display:flex; align-items:center; gap:10px;
      padding:10px 12px;
      border-radius:14px;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(7,10,20,.40);
    }
    .toggle input{ width:20px; height:20px; }
    .toggle span{ color:rgba(233,240,255,.86); font-weight:800; }
    .toggle small{ color:var(--muted2); display:block; font-weight:600; margin-top:2px; }

    .toast{
      position:fixed;
      left:50%;
      bottom:18px;
      transform:translateX(-50%);
      background: rgba(0,0,0,.55);
      border:1px solid rgba(255,255,255,.14);
      box-shadow: var(--shadow2);
      color:rgba(233,240,255,.9);
      padding:10px 12px;
      border-radius: 14px;
      font-family: var(--mono);
      font-size:12px;
      opacity:0;
      pointer-events:none;
      transition: opacity .18s ease;
    }
    .toast.show{opacity:1}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <div class="title">covariant — picks console</div>
        <p class="subtitle">black / neon • local engine • ledger picks</p>
      </div>
      <div id="hdrBadge" class="badge">status: idle</div>
    </div>

    <div class="panel">
      <div class="grid">
        <div>
          <label>BANKROLL</label>
          <input id="bankroll" value="500" inputmode="decimal"/>
        </div>
        <div>
          <label>MIN_EV</label>
          <input id="min_ev" value="0.01" inputmode="decimal"/>
        </div>
        <div>
          <label>MIN_EDGE_PCT</label>
          <input id="min_edge" value="1.0" inputmode="decimal"/>
        </div>
      </div>

      <div class="controls">
        <div>
          <label>SORT BY</label>
          <select id="sort_by">
            <option value="edge" selected>Edge % (desc)</option>
            <option value="ev1">EV / $1 (desc)</option>
            <option value="stake">Stake (desc)</option>
          </select>
        </div>

        <div class="toggle">
          <input id="auto_refresh" type="checkbox"/>
          <div>
            <span>Refresh every 30s</span>
            <small>auto-load latest ledger</small>
          </div>
        </div>

        <div>
          <label>&nbsp;</label>
          <button class="accent" style="width:100%" onclick="copyBetSlip()">Copy bet slip</button>
        </div>
      </div>

      <div class="btnrow">
        <button class="ghost" onclick="health()">Health</button>
        <button class="primary" onclick="generate()">Generate Picks</button>
        <button class="ghost" onclick="loadLatest()">Load Latest</button>
        <button class="ghost" onclick="gradeLatest()">Grade Latest</button>
      </div>

      <div id="status" class="status">(ready)</div>
      <div class="hint">If you hit “Address already in use”, bump PORT (e.g. 8095 → 8096).</div>
    </div>

    <h2>SINGLES</h2>
    <div id="singles" class="cards"></div>

    <h2>PARLAYS</h2>
    <div id="parlays" class="cards"></div>

    <details>
      <summary>Raw JSON (latest ledger)</summary>
      <pre id="raw">(none loaded)</pre>
    </details>
  </div>

  <div id="toast" class="toast">copied</div>

<script>
  // ===== JS boot + error trap (shows why buttons don't work) =====
  (function(){
    const show = (msg) => {
      try { document.getElementById("hdrBadge").textContent = "status: " + msg; } catch(e){}
      try { document.getElementById("status").textContent = msg; } catch(e){}
      try { document.getElementById("raw").textContent = String(msg); } catch(e){}
    };
    show("ui script booting…");

    window.addEventListener("error", (e) => {
      const msg = "JS error: " + (e?.message || e);
      show(msg);
    });

    window.addEventListener("unhandledrejection", (e) => {
      const msg = "Promise error: " + (e?.reason?.message || e?.reason || e);
      show(msg);
    });
  })();

const $ = (id)=>document.getElementById(id);

let LAST_LEDGER = null;        // full ledger payload
let LAST_NORM = {singles:[], parlays:[]};
let REFRESH_TIMER = null;

function setStatus(msg){
  $("status").textContent = msg;
  $("hdrBadge").textContent = "status: " + msg.replace(/\s+/g," ").slice(0,80);
}

function toast(msg){
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"), 1200);
}

function pct(x){
  if (x === null || x === undefined || x === "") return "—";
  const n = Number(x);
  if (!isFinite(n)) return String(x);
  if (n > 1 && n <= 100) return n.toFixed(3) + "%";
  if (n >= 0 && n <= 1) return (n*100).toFixed(3) + "%";
  return n.toFixed(3) + "%";
}
function num(x, d=4){
  if (x === null || x === undefined || x === "") return "—";
  const n = Number(x);
  if (!isFinite(n)) return String(x);
  return n.toFixed(d);
}
function nfloat(x){
  const n = Number(x);
  return isFinite(n) ? n : NaN;
}

function edgePill(edge){
  const n = Number(edge);
  let cls = "pill";
  if (isFinite(n)){
    if (n >= 2.0) cls += " good";
    else if (n >= 0.5) cls += " warn";
    else cls += " bad";
  }
  return `<span class="${cls}">edge ${pct(edge)}</span>`;
}

// ---- schema normalizer (old + new ledgers) ----
function normalizeLedger(data){
  let singles = [];
  let parlays = [];

  if (Array.isArray(data?._normalized_picks)) singles = data._normalized_picks;
  else if (Array.isArray(data?.picks)) singles = data.picks;
  else if (Array.isArray(data?.singles?.picks)) singles = data.singles.picks;

  if (Array.isArray(data?.parlays_sized) && data.parlays_sized.length) parlays = data.parlays_sized;
  else if (Array.isArray(data?.parlays)) parlays = data.parlays;

  return {singles, parlays};
}

function pickFields(p){
  const away = p.away || p.away_team || p.event_away || p.a || "";
  const home = p.home || p.home_team || p.event_home || p.h || "";
  const matchup = p.matchup || p.game || ((away && home) ? `${away} @ ${home}` : (p.event || p.name || "Pick"));

  const market = p.market || p.market_key || p.type || "market";
  const selection = p.selection || p.team || p.side || p.outcome || p.pick || p.bet || "—";
  const book = p.book || p.bookmaker || p.sportsbook || "—";
  const odds = p.odds || p.price || p.american_odds || "—";

  const imp = p.implied || p.implied_prob || p.p_implied;
  const model = p.model || p.model_prob || p.p_model || p.p;

  const edge = p.edge || p.edge_pct || p.edge_percent;
  const ev1 = p.ev_per_dollar || p.ev_per_$1 || p.ev_per_1 || p.ev_per_unit || p.ev;
  const stake = p.stake || p.stake_usd || p.size || p.units || p.bet_size;

  const commence = p.commence_time || p.commence || p.start || p.start_time;
  const snap = p.snapshot_ts || p.snapshot || p.odds_ts || p.ts;

  return {matchup, market, selection, book, odds, imp, model, edge, ev1, stake, commence, snap};
}

function escapeHtml(s){
  return String(s ?? "").replace(/[&<>"']/g, (c)=>({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));
}

function renderPickCard(p){
  const f = pickFields(p);
  return `
    <div class="card">
      ${edgePill(f.edge)}
      <div class="match">${escapeHtml(f.matchup)}</div>
      <div class="meta">${escapeHtml(f.market)} → <b>${escapeHtml(f.selection)}</b></div>
      <div class="row">
        <span class="chip">book ${escapeHtml(f.book)}</span>
        <span class="chip">odds ${escapeHtml(String(f.odds))}</span>
        <span class="chip">implied ${pct(f.imp)}</span>
        <span class="chip">model ${pct(f.model)}</span>
        <span class="chip">EV/$1 ${num(f.ev1, 4)}</span>
        <span class="chip">stake ${escapeHtml(String(f.stake ?? "—"))}</span>
      </div>
      <div class="row">
        <span class="chip">commence ${escapeHtml(String(f.commence ?? "—"))}</span>
        <span class="chip">snapshot ${escapeHtml(String(f.snap ?? "—"))}</span>
      </div>
    </div>
  `;
}

function renderParlayCard(p){
  const legs = p.legs || p.parlay_legs || p.picks || [];
  const stake = p.stake || p.stake_usd || p.size || "—";
  const ev = p.ev || p.expected_value || p.ev_total || "—";
  const edge = p.edge || p.edge_pct || p.edge_percent;

  const legsHtml = Array.isArray(legs) && legs.length
    ? legs.map((L,i)=> {
        const lf = pickFields(L);
        return `<div class="meta" style="margin-top:6px">${i+1}. ${escapeHtml(lf.market)} → <b>${escapeHtml(lf.selection)}</b> <span style="color:rgba(111,126,162,.95)">(${escapeHtml(lf.book)} ${escapeHtml(String(lf.odds))})</span></div>`;
      }).join("")
    : `<div class="meta" style="margin-top:6px">No legs field found in this parlay object.</div>`;

  return `
    <div class="card">
      ${edgePill(edge)}
      <div class="match">Parlay</div>
      <div class="row">
        <span class="chip">stake ${escapeHtml(String(stake))}</span>
        <span class="chip">EV ${escapeHtml(String(ev))}</span>
      </div>
      ${legsHtml}
    </div>
  `;
}

function sortSingles(singles){
  const key = $("sort_by").value;
  const arr = singles.slice();

  function val(p){
    const f = pickFields(p);
    if (key === "edge") return nfloat(f.edge);
    if (key === "ev1") return nfloat(f.ev1);
    if (key === "stake") return nfloat(f.stake);
    return NaN;
  }
  arr.sort((a,b)=>{
    const va = val(a), vb = val(b);
    if (!isFinite(va) && !isFinite(vb)) return 0;
    if (!isFinite(va)) return 1;
    if (!isFinite(vb)) return -1;
    return vb - va; // desc
  });
  return arr;
}

function renderAll(){
  const singlesSorted = sortSingles(LAST_NORM.singles);

  $("singles").innerHTML = singlesSorted.length
    ? singlesSorted.map(renderPickCard).join("")
    : `<div class="card"><div class="meta">No singles found (schema mismatch or filters dropped everything).</div></div>`;

  $("parlays").innerHTML = LAST_NORM.parlays.length
    ? LAST_NORM.parlays.map(renderParlayCard).join("")
    : `<div class="card"><div class="meta">No parlays found (or not enabled in this run).</div></div>`;
}

function betSlipText(){
  const s = sortSingles(LAST_NORM.singles);
  if (!s.length) return "No singles loaded.";

  const lines = [];
  lines.push("COVARIANT BET SLIP");
  lines.push(`Sort: ${$("sort_by").value} (desc)`);
  lines.push(`Bankroll=${$("bankroll").value} | MIN_EV=${$("min_ev").value} | MIN_EDGE_PCT=${$("min_edge").value}`);
  lines.push("");

  s.forEach((p, i)=>{
    const f = pickFields(p);
    lines.push(`${i+1}) ${f.matchup}`);
    lines.push(`   ${f.market} → ${f.selection}`);
    lines.push(`   Book: ${f.book} | Odds: ${f.odds}`);
    lines.push(`   Model: ${pct(f.model)} | Implied: ${pct(f.imp)} | Edge: ${pct(f.edge)} | EV/$1: ${num(f.ev1,4)} | Stake: ${f.stake ?? "—"}`);
    lines.push("");
  });

  if (LAST_NORM.parlays && LAST_NORM.parlays.length){
    lines.push("PARLAYS");
    lines.push("");
    LAST_NORM.parlays.forEach((par, i)=>{
      lines.push(`${i+1}) Parlay`);
      const legs = par.legs || par.parlay_legs || par.picks || [];
      if (Array.isArray(legs) && legs.length){
        legs.forEach((L,j)=>{
          const lf = pickFields(L);
          lines.push(`   ${j+1}. ${lf.market} → ${lf.selection} (${lf.book} ${lf.odds})`);
        });
      } else {
        lines.push("   (no legs field found)");
      }
      lines.push(`   Stake: ${par.stake ?? par.size ?? "—"} | EV: ${par.ev ?? par.expected_value ?? "—"} | Edge: ${pct(par.edge ?? par.edge_pct ?? par.edge_percent)}`);
      lines.push("");
    });
  }

  return lines.join("\n").trim();
}

async function copyBetSlip(){
  const txt = betSlipText();
  try{
    await navigator.clipboard.writeText(txt);
    toast("✅ bet slip copied");
  }catch(e){
    // fallback: put in raw box and instruct user
    $("raw").textContent = txt;
    toast("clipboard blocked — slip in Raw JSON");
  }
}

async function health(){
  setStatus("checking health…");
  const r = await fetch("/health");
  const j = await r.json();
  setStatus(j.ok ? `healthy ✓ | ${j.service} | odds_key=${j.odds_api_key_present}` : "health failed ✗");
  $("raw").textContent = JSON.stringify(j, null, 2);
}

async function generate(){
  setStatus("generating picks…");
  const payload = {
    BANKROLL: $("bankroll").value,
    MIN_EV: $("min_ev").value,
    MIN_EDGE_PCT: $("min_edge").value
  };
  const r = await fetch("/api/picks/generate", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const j = await r.json();
  setStatus(j.ok ? `generated ✓ | ledger=${j.latest_ledger || "?"}` : "generate failed ✗ (see raw)");
  $("raw").textContent = JSON.stringify(j, null, 2);
  if (j.ok) await loadLatest();
}

async function loadLatest(){
  setStatus("loading latest…");
  const r = await fetch("/api/picks/latest");
  const j = await r.json();

  if (!j.ok){
    setStatus("load failed ✗");
    $("raw").textContent = JSON.stringify(j, null, 2);
    return;
  }

  LAST_LEDGER = j;
  const data = j.data || {};
  LAST_NORM = normalizeLedger(data);

  renderAll();

  setStatus(`loaded ✓ | singles=${LAST_NORM.singles.length} | parlays=${LAST_NORM.parlays.length} | ${j.path}`);
  $("raw").textContent = JSON.stringify(j, null, 2);
}

async function gradeLatest(){
  setStatus("grading latest…");
  const r = await fetch("/api/grade/latest", {method:"POST"});
  const j = await r.json();
  setStatus(j.ok ? `graded ✓ | ${j.graded_latest || "ok"}` : "grade failed ✗ (check ODDS plan/endpoint)");
  $("raw").textContent = JSON.stringify(j, null, 2);
}

function startAutoRefresh(){
  stopAutoRefresh();
  REFRESH_TIMER = setInterval(()=>loadLatest(), 30000);
}
function stopAutoRefresh(){
  if (REFRESH_TIMER){ clearInterval(REFRESH_TIMER); REFRESH_TIMER = null; }
}

$("sort_by").addEventListener("change", ()=>{
  if (LAST_NORM.singles.length || LAST_NORM.parlays.length) renderAll();
});

$("auto_refresh").addEventListener("change", (e)=>{
  if (e.target.checked){
    startAutoRefresh();
    toast("auto refresh ON");
    loadLatest();
  } else {
    stopAutoRefresh();
    toast("auto refresh OFF");
  }
});

  // ===== END OF SCRIPT MARKER =====
  try {
    document.getElementById("hdrBadge").textContent =
      "status: ui ready ✓";
  } catch(e) {}

</script>

</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if self.path == "/health":
            body = json_bytes({
                "ok": True,
                "service": APP_NAME,
                "time": int(time.time()),
                "cwd": str(Path.cwd()),
                "python": subprocess.getoutput("python -V"),
                "odds_api_key_present": bool(os.environ.get("ODDS_API_KEY","").strip()),
            })
            self._send(200, body, "application/json; charset=utf-8")
            return

        if self.path == "/api/picks/latest":
            latest = latest_file("picks_full_*.json") or latest_file("picks_*.json")
            if not latest:
                self._send(404, json_bytes({"ok": False, "error":"No picks ledger found in runs/."}), "application/json; charset=utf-8")
                return
            data = json.loads(Path(latest).read_text())

              
            # --- normalize schema (ALL generator versions) ---
            picks = []

            # v1 schema
            if isinstance(data.get("picks"), list):
                picks = data["picks"]

            # v2 schema
            elif isinstance(data.get("singles"), dict):
                if isinstance(data["singles"].get("picks"), list):
                    picks = data["singles"]["picks"]

            # v3 schema (current engine)
            elif isinstance(data.get("singles"), list):
                picks = data["singles"]

            data["_normalized_picks"] = picks

            self._send(200, json_bytes({"ok": True, "path": latest, "data": data}), "application/json; charset=utf-8")
            return

        self._send(404, json_bytes({"ok": False, "error":"not found", "path": self.path}), "application/json; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length","0") or "0")
        raw = self.rfile.read(length) if length > 0 else b""
        payload = {}
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}

        if self.path == "/api/picks/generate":
            RUNS_DIR.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            if "BANKROLL" in payload: env["BANKROLL"] = str(payload["BANKROLL"])
            if "MIN_EV" in payload: env["MIN_EV"] = str(payload["MIN_EV"])
            if "MIN_EDGE_PCT" in payload: env["MIN_EDGE_PCT"] = str(payload["MIN_EDGE_PCT"])

            code, out = run_cmd(["python", "generate_picks_full.py", "--top", "50"], env=env)
            latest = latest_file("picks_full_*.json") or latest_file("picks_*.json")

            self._send(200 if code == 0 else 500, json_bytes({
                "ok": code == 0,
                "exit_code": code,
                "latest_ledger": latest,
                "log_tail": out[-4000:],
            }), "application/json; charset=utf-8")
            return

        if self.path == "/api/grade/latest":
            if not os.environ.get("ODDS_API_KEY","").strip():
                self._send(400, json_bytes({"ok": False, "error":"Set ODDS_API_KEY in environment first."}), "application/json; charset=utf-8")
                return

            latest = latest_file("picks_full_*.json") or latest_file("picks_*.json")
            if not latest:
                self._send(404, json_bytes({"ok": False, "error":"No picks ledger found in runs/."}), "application/json; charset=utf-8")
                return

            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            code, out = run_cmd(["python", "grade_outcomes.py", "--ledger", latest, "--csv", "auto"])

            graded = latest_file("graded_*.json")
            self._send(200 if code == 0 else 500, json_bytes({
                "ok": code == 0,
                "exit_code": code,
                "graded_latest": graded,
                "log_tail": out[-4000:],
            }), "application/json; charset=utf-8")
            return

        self._send(404, json_bytes({"ok": False, "error":"not found", "path": self.path}), "application/json; charset=utf-8")

def main():
    port = int(os.environ.get("PORT","8088"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[{APP_NAME}] listening on http://127.0.0.1:{port}  (and 0.0.0.0:{port})")
    server.serve_forever()

if __name__ == "__main__":
    main()
