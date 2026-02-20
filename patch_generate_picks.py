from pathlib import Path
import re

p = Path("generate_picks.py")
s = p.read_text()

# ---------------------------------------------------
# FIX 1 — remove hardcoded API key (causes 401 errors)
# ---------------------------------------------------
s = re.sub(
    r'DEFAULT_API_KEY\s*=\s*".*?"',
    'DEFAULT_API_KEY = os.environ.get("ODDS_API_KEY", "")',
    s
)

# ---------------------------------------------------
# FIX 2 — safe odds parsing (books sometimes return str/None)
# ---------------------------------------------------
s = s.replace(
    "odds = int(odds)",
    "odds = int(float(odds)) if odds is not None else None"
)

# ---------------------------------------------------
# FIX 3 — protect against missing bookmaker data
# ---------------------------------------------------
s = s.replace(
    'bm.get("markets", []) or []',
    '(bm.get("markets") or [])'
)

# ---------------------------------------------------
# FIX 4 — prevent crash when price missing
# ---------------------------------------------------
s = s.replace(
    '[int(x["price"]) for x in grouped.get(a_name, []) if x.get("price") is not None]',
    '[int(float(x["price"])) for x in grouped.get(a_name, []) if x.get("price") not in (None,"")]'
)

s = s.replace(
    '[int(x["price"]) for x in grouped.get(b_name, []) if x.get("price") is not None]',
    '[int(float(x["price"])) for x in grouped.get(b_name, []) if x.get("price") not in (None,"")]'
)

# ---------------------------------------------------
# FIX 5 — safer HTTP error message
# ---------------------------------------------------
s = s.replace(
    'raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:200]}")',
    'raise RuntimeError(f"Odds API error {r.status_code}: {r.text[:200]} (check API key / quota)")'
)

p.write_text(s)
print("✅ generate_picks.py patched successfully")
