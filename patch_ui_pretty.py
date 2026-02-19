from __future__ import annotations
from pathlib import Path
import re

path = Path("web_server_stdlib.py")
txt = path.read_text(encoding="utf-8")

# ---------- JS to inject/replace ----------
EDGE_FN = r"""
function edgeColor(edge) {
  const e = Number(edge || 0);
  if (e >= 3) return "#22c55e";   // strong edge
  if (e >= 1) return "#eab308";   // medium
  return "#ef4444";               // weak
}
""".strip()

RENDER_FN = r"""
function renderPicks(picks) {
  const container = document.getElementById("singles");
  if (!container) return;
  container.innerHTML = "";

  if (!picks || picks.length === 0) {
    container.innerHTML = '<div class="empty">No singles found.</div>';
    return;
  }

  picks.forEach(p => {
    const edgeRaw = (p.edge_pct ?? p.edge ?? 0);
    const edge = Number(edgeRaw || 0);
    const edgeStr = edge.toFixed(3);

    const card = document.createElement("div");
    card.className = "pick-card";

    const modelProb = (p.model_prob != null ? (Number(p.model_prob) * 100.0) : null);
    const evd = (p.ev_per_dollar != null ? Number(p.ev_per_dollar) : null);

    card.innerHTML = `
      <div class="pick-header">
        <div>
          <div class="game">${p.away || "?"} @ ${p.home || "?"}</div>
          <div class="market">${p.market || "market"} → ${p.selection || "selection"}</div>
        </div>
        <div class="edge-pill"
             style="background:${edgeColor(edge)}22;
                    color:${edgeColor(edge)};
                    border:1px solid ${edgeColor(edge)};">
          edge ${edgeStr}%
        </div>
      </div>

      <div class="pick-grid">
        <div><label>Book</label>${p.book || "-"}</div>
        <div><label>Odds</label>${(p.odds ?? "-")}</div>
        <div><label>Implied</label>${(p.implied_prob != null ? (Number(p.implied_prob)*100).toFixed(2)+"%" : "-")}</div>
        <div><label>Model</label>${(modelProb != null ? modelProb.toFixed(2)+"%" : "-")}</div>
        <div><label>EV / $1</label>${(evd != null ? evd.toFixed(4) : "-")}</div>
        <div><label>Stake</label>${(p.stake ?? "-")}</div>
        <div><label>Commence</label>${(p.commence_time ?? "-")}</div>
        <div><label>Snapshot</label>${(p.snapshot_time ?? "-")}</div>
      </div>
    `;

    container.appendChild(card);
  });
}
""".strip()

def replace_or_inject_function(src: str, name: str, body: str) -> tuple[str, bool]:
  # Replace `function name(...) { ... }` (non-greedy) if present
  pat = re.compile(rf"function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{.*?\n\}}", re.DOTALL)
  m = pat.search(src)
  if m:
    return pat.sub(body, src, count=1), True
  return src, False

# Inject into <script>...</script>
script_m = re.search(r"(<script>)(.*?)(</script>)", txt, flags=re.DOTALL)
if not script_m:
  raise SystemExit("❌ Could not find <script>...</script> block in web_server_stdlib.py")

script_open, script_body, script_close = script_m.group(1), script_m.group(2), script_m.group(3)

changed = False

# Ensure edgeColor exists
script_body2, did = replace_or_inject_function(script_body, "edgeColor", EDGE_FN)
if did:
  changed = True
  script_body = script_body2
else:
  # only inject if not already present
  if "function edgeColor" not in script_body:
    script_body = script_body + "\n\n" + EDGE_FN + "\n"
    changed = True

# Replace or inject renderPicks
script_body2, did = replace_or_inject_function(script_body, "renderPicks", RENDER_FN)
if did:
  changed = True
  script_body = script_body2
else:
  script_body = script_body + "\n\n" + RENDER_FN + "\n"
  changed = True

# ---------- CSS to inject ----------
CSS = r"""
.pick-card {
  background:#0b1220;
  border-radius:14px;
  padding:14px;
  margin:12px 0;
  border:1px solid #1f2937;
}
.pick-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:10px;
}
.game {
  font-weight:600;
  font-size:16px;
}
.market {
  opacity:.7;
  font-size:13px;
}
.edge-pill {
  padding:6px 10px;
  border-radius:999px;
  font-weight:600;
}
.pick-grid {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10px;
}
.pick-grid label {
  display:block;
  font-size:11px;
  opacity:.6;
}
.empty {
  opacity:.75;
  padding:12px;
  border:1px dashed #334155;
  border-radius:12px;
}
""".strip()

# Insert CSS before </style> if not present
if ".pick-card" not in txt:
  style_m = re.search(r"(<style>)(.*?)(</style>)", txt, flags=re.DOTALL)
  if style_m:
    style_open, style_body, style_close = style_m.group(1), style_m.group(2), style_m.group(3)
    style_body = style_body.rstrip() + "\n\n" + CSS + "\n"
    txt = re.sub(r"<style>.*?</style>", style_open + style_body + style_close, txt, count=1, flags=re.DOTALL)
    changed = True

# Write back with updated script block
new_script_block = script_open + script_body + script_close
txt2 = txt[:script_m.start()] + new_script_block + txt[script_m.end():]

if txt2 != txt:
  changed = True
  txt = txt2

if not changed:
  print("ℹ️ No changes needed (already patched).")
else:
  path.write_text(txt, encoding="utf-8")
  print("✅ Patched UI rendering + CSS successfully.")
