"""
build_app.py -- glue the generated circuit into the app, producing the single file the IDE imports.

    python badge/export_badge_circuit.py       # 1. the connectome -> badge/flybadge_circuit.lua
    python badge/build_app.py                  # 2. + app_template.lua -> badge/flybadge_app.lua
    lua badge/test_lua.lua                     # 3. run it against the Python reference

The badge IDE's "Import app" takes one file whose leading long comment is the manifest, and the guide is
explicit that an app must carry its helpers inline with no require() dependencies. So the circuit cannot
stay a module: this substitutes it into the template at `--@CIRCUIT@` and checks the result against the
documented limits (64 KiB of main.lua, and a source small enough to compile inside the Lua heap).

Whole-line comments are stripped from the OUTPUT by default (`--keep-comments` turns that off). The
badge compiles main.lua in RAM and a serial push of a larger file is the flakiest step in the whole
loop, and every byte of prose is a byte pushed and lexed on the device for nothing. The documentation
lives in badge/app_template.lua and badge/flybadge_circuit.lua, which is what a human reads; the built
file is a push artifact. Trailing comments on code lines are left alone, because a `--` inside a string
literal is not worth the risk for the few hundred bytes it would save.
"""
from __future__ import annotations

import argparse
import os
import re


BANNER = """-- FlyBadge / STARTLE. Comments stripped for the push; badge/app_template.lua is the documented source.
-- 150 real neurons and 900 real synapses of the male Drosophila CNS connectome (neuPrint male-cns:v1.0,
-- Berg et al. 2026). No statement in here decides the escape; only the giant fibre's spike does.
"""


def strip_comments(src: str) -> str:
    """Drop whole-line `--` comments, keeping the manifest and any trailing comment on a code line.

    Only lines whose first non-space characters are `--` are removed, and only after the manifest long
    comment has ended, so a `--` inside a string literal is never touched. Runs of blank lines left
    behind are collapsed to one. The app carries no long strings, which is checked below.
    """
    if "[[" in src or "[==[" in src.split("]==]", 1)[1]:
        raise SystemExit("long bracket string in the app: comment stripping is not safe, use --keep-comments")
    head, _, body = src.partition("]==]\n")
    kept, blank = [], False
    for line in body.splitlines():
        if line.lstrip().startswith("--"):
            continue
        if not line.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        kept.append(line)
    return head + "]==]\n" + BANNER + "\n".join(kept).strip("\n") + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="badge/app_template.lua")
    ap.add_argument("--circuit", default="badge/flybadge_circuit.lua")
    ap.add_argument("--out", default="badge/flybadge_app.lua")
    ap.add_argument("--keep-comments", action="store_true",
                    help="do not strip whole-line comments from the output")
    a = ap.parse_args()

    tpl = open(a.template).read()
    cir = open(a.circuit).read()
    if "--@CIRCUIT@" not in tpl:
        raise SystemExit(f"{a.template} has no --@CIRCUIT@ marker")

    # the module becomes a plain local table: `local M = {}` .. `return M` -> `local C = {}`
    cir = cir.replace("local M = {}", "local C = {}").replace("M.", "C.")
    cir = re.sub(r"^return M\s*$", "", cir, flags=re.M).rstrip() + "\n"

    out = tpl.replace("--@CIRCUIT@", cir)
    full = len(out.encode())
    if not a.keep_comments:
        out = strip_comments(out)
    with open(a.out, "w") as f:
        f.write(out)

    size = len(out.encode())
    manifest = re.search(r"--\[==\[badge-app\n(.*?)\n\]==\]", out, re.S)
    keys = dict(l.split("=", 1) for l in manifest.group(1).splitlines() if "=" in l) if manifest else {}
    print(f"{a.out}: {size / 1024:.1f} KB  ({size} bytes of the 65,536 main.lua limit)"
          + ("" if a.keep_comments else f"; {full / 1024:.1f} KB of commented source"))
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
