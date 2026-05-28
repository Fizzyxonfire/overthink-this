"""One-off: extract the exact spiral prompt pieces from server.py and emit
a base64 JSON blob the Supabase Edge Function can decode at runtime. This
guarantees the migrated spiral prompt is byte-identical to the Render one."""
import ast, json, base64, os

src = open(os.path.join(os.path.dirname(__file__), "server.py"), encoding="utf-8").read()
tree = ast.parse(src)
lines = src.splitlines(keepends=True)

def seg(node):
    return "".join(lines[node.lineno - 1:node.end_lineno])

want = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in ("TONE_PROMPTS", "FEW_SHOT_EXAMPLE"):
                want[t.id] = seg(node)
    if isinstance(node, ast.FunctionDef) and node.name == "_build_prompt":
        want["_build_prompt"] = seg(node)

ns: dict = {}
exec(want["FEW_SHOT_EXAMPLE"], ns)
exec(want["TONE_PROMPTS"], ns)
exec(want["_build_prompt"], ns)
build = ns["_build_prompt"]

out = {}
for tone in ("gentle", "balanced", "brutal", "roast"):
    out[tone] = build("__SITUATION__", tone, "__CATEGORY__")

b64 = base64.b64encode(json.dumps(out).encode("utf-8")).decode("ascii")
dest = os.path.join(os.path.dirname(__file__), "..", "supabase", "functions", "ai", "_prompts.b64")
open(dest, "w", encoding="utf-8").write(b64)
print("OK len(b64) =", len(b64))
