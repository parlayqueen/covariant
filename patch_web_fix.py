from pathlib import Path

f = Path("web_server_stdlib.py")
txt = f.read_text()

old = """data = json.loads(Path(latest).read_text())
            self._send(200, json_bytes({
                "ok": True,
                "path": latest,
                "data": data
            }), "application/json; charset=utf-8")"""

new = """data = json.loads(Path(latest).read_text())

            # --- normalize schema (old + new generators) ---
            picks = []

            if isinstance(data.get("picks"), list):
                picks = data["picks"]

            elif isinstance(data.get("singles"), dict):
                if isinstance(data["singles"].get("picks"), list):
                    picks = data["singles"]["picks"]

            data["_normalized_picks"] = picks

            self._send(200, json_bytes({
                "ok": True,
                "path": latest,
                "data": data
            }), "application/json; charset=utf-8")"""

if old not in txt:
    print("❌ Could not find target block — file layout slightly different.")
else:
    txt = txt.replace(old, new)
    f.write_text(txt)
    print("✅ Patch applied successfully.")
