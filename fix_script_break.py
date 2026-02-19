from pathlib import Path
import re

f = Path("web_server_stdlib.py")
txt = f.read_text()

# Find accidental JS leaking outside script tag
pattern = r"</script>\s*;.*?gradeLatest\(\)\s*\}"
match = re.search(pattern, txt, re.DOTALL)

if match:
    leaked = match.group(0)

    # rebuild correctly inside script tag
    fixed = leaked.replace("</script>", "")
    txt = txt.replace(leaked, fixed + "\n</script>")

    f.write_text(txt)
    print("✅ Script block repaired.")
else:
    print("⚠️ Could not auto-detect — script location slightly different.")
