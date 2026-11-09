"""Run every program in the course and report which ones fail.

    python tests/run_all.py              # run everything
    python tests/run_all.py u04 u05      # only units whose folder starts so
    python tests/run_all.py --save OUT   # also save each program's output

A program passes if it finishes without an error within the time limit.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"gridworld.py"}                 # imported by others, not run
args = sys.argv[1:]
save = None
if "--save" in args:
    i = args.index("--save")
    save = Path(args[i + 1]).resolve()
    del args[i:i + 2]

env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
failed, n = [], 0
for unit in sorted(ROOT.glob("u[0-9][0-9]-*")):
    if args and not any(unit.name.startswith(a) for a in args):
        continue
    for prog in sorted(unit.glob("*.py")):
        if prog.name in MODULES:
            continue
        n += 1
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, prog.name], cwd=unit,
                               env=env, capture_output=True, text=True,
                               timeout=600)
            ok, out = r.returncode == 0, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            ok, out = False, "timed out"
        print(f"{'PASS' if ok else 'FAIL'}  {unit.name}/{prog.name}"
              f"  ({time.time() - t0:.1f} s)")
        if not ok:
            failed.append(f"{unit.name}/{prog.name}")
            print(out[-2000:])
        if save:
            dest = save / unit.name
            dest.mkdir(parents=True, exist_ok=True)
            (dest / (prog.stem + ".txt")).write_text(r.stdout)
print(f"\n{n - len(failed)} of {n} programs passed")
sys.exit(1 if failed else 0)
