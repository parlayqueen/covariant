from pathlib import Path
import re

p = Path("web_server_stdlib.py")
txt = p.read_text()

# Find the GET handler for /api/picks/latest and inject normalization right after JSON load.
pattern = r'(data\s*=\s*json\.loads\(Path\(latest\)\.read_text\(\)\)\s*\n)'
m = re.search(pattern, txt)
if not m:
    print("❌ Couldn't find JSON load line in web_server_stdlib.py")
    raise SystemExit(1)

inject = m.group(1) + """
            # --- normalize schema (old + new generators) ---
            picks = []
            if isinstance(data.get("picks"), list):
                picks = data["picks"]
            elif isinstance(data.get("singles"), dict):
                if isinstance(data["singles"].get("picks"), list):
                    picks = data["singles"]["picks"]
            data["_normalized_picks"] = picks

"""

# Only inject once
if "data[\"_normalized_picks\"]" in txt or "data['_normalized_picks']" in txt:
    print("ℹ️ Normalization already present; no changes made.")
else:
    txt = re.sub(pattern, inject, txt, count=1)
    p.write_text(txt)
    print("✅ Patch v2 applied successfully.")
