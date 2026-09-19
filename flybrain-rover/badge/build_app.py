"""
build_app.py -- glue the generated circuit into the app, producing the single file the IDE imports.

    python badge/export_badge_circuit.py       # 1. the connectome -> badge/flybadge_circuit.lua
    python badge/build_app.py                  # 2. + app_template.lua -> badge/flybadge_app.lua
    lua badge/test_lua.lua                     # 3. run it against the Python reference

The badge IDE's "Import app" takes one file whose leading long comment is the manifest, and the guide is
explicit that an app must carry its helpers inline with no require() dependencies. So the circuit cannot
stay a module: this substitutes it into the template at `--@CIRCUIT@` and checks the result against the
documented limits (64 KiB of main.lua, and a source small enough to compile inside the Lua heap).
"""
from __future__ import annotations

import argparse
import os
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="badge/app_template.lua")
    ap.add_argument("--circuit", default="badge/flybadge_circuit.lua")
    ap.add_argument("--out", default="badge/flybadge_app.lua")
    a = ap.parse_args()

    tpl = open(a.template).read()
    cir = open(a.circuit).read()
    if "--@CIRCUIT@" not in tpl:
        raise SystemExit(f"{a.template} has no --@CIRCUIT@ marker")

    # the module becomes a plain local table: `local M = {}` .. `return M` -> `local C = {}`
    cir = cir.replace("local M = {}", "local C = {}").replace("M.", "C.")
    cir = re.sub(r"^return M\s*$", "", cir, flags=re.M).rstrip() + "\n"

    out = tpl.replace("--@CIRCUIT@", cir)
    with open(a.out, "w") as f:
        f.write(out)

    size = len(out.encode())
    manifest = re.search(r"--\[==\[badge-app\n(.*?)\n\]==\]", out, re.S)
    keys = dict(l.split("=", 1) for l in manifest.group(1).splitlines() if "=" in l) if manifest else {}
    print(f"{a.out}: {size / 1024:.1f} KB  ({size} bytes of the 65,536 main.lua limit)")
    print(f"  slug={keys.get('slug')}  api={keys.get('api')}  heap_kb={keys.get('heap_kb')}  "
          f"wake_lock={keys.get('wake_lock')}")
    for need in ("on_enter", "on_tick", "on_button", "on_exit"):
        assert f"function {need}(" in out, f"missing {need}"
    print(f"  lifecycle: on_enter on_tick on_button on_exit")
    for banned in ("pcall", "coroutine", "os", "io", "require", "setmetatable", "load", "dofile"):
        if re.search(rf"(?<![\w.]){banned}\s*[.(]", out):   # not badge.radio. or a word ending in io
            print(f"  WARNING: uses {banned}, which the badge sandbox does not provide")
    if size > 65536:
        raise SystemExit("over the 64 KiB main.lua limit")
    if size > 40000:
        print("  WARNING: large source; the badge compiles main.lua in RAM. Shrink the circuit.")
    print(f"\nPaste {a.out} into the badge IDE: Import app -> Connect -> Push.")


if __name__ == "__main__":
    main()
