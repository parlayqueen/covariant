from pathlib import Path
import re

f = Path("web_server_stdlib.py")
txt = f.read_text()

# Locate the INDEX_HTML triple-quoted string safely
m = re.search(r'INDEX_HTML\s*=\s*"""(.*?)"""', txt, re.DOTALL)
if not m:
    raise SystemExit("❌ Could not find INDEX_HTML block in web_server_stdlib.py")

html = m.group(1)
orig = html

# 1) If any JS leaked AFTER the HTML ends, cut it
end_html = html.rfind("</html>")
if end_html != -1:
    html = html[: end_html + len("</html>")]

# 2) If JS leaked right after </script> as visible text, try to pull it back in
leak = re.search(r"</script>\s*([^<].*?)</body>", html, re.DOTALL)
if leak:
    leaked_js = leak.group(1).strip()
    if any(k in leaked_js for k in ("function", "const ", "let ", "var ", "async ")):
        html = re.sub(r"</script>\s*([^<].*?)</body>", "</script></body>", html, flags=re.DOTALL)
        html = html.replace("</script>", "\n" + leaked_js + "\n</script>", 1)

if html == orig:
    print("⚠️ No obvious leak pattern found. (UI might already be fixed.)")
else:
    new_txt = txt[:m.start(1)] + html + txt[m.end(1):]
    f.write_text(new_txt)
    print("✅ UI leak fix applied to INDEX_HTML.")
