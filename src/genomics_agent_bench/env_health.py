"""Method-environment health: the node's shared uv cache, and what each agent run actually ran into.

Equipped runs get their method environments as symlinks into one uv cache per node, shared by every run on it.
A package copy in that cache that has lost files breaks every later environment that links to it, silently: the
run still scores, but the agent finds SPACEL, Spateo or pandas unimportable and spends its time rebuilding them.
Found on 2026-10-23 on compute-128 (14 damaged archives: six, python-dateutil, scipy, scikit-learn, numpy, ...).

Two checks, standard library only so the job script can run them with the node's own python3:

  cache  -- every archive in <cache>/archive-v0 against its wheel RECORD. "missing" (files gone) breaks imports;
            with --repair the damaged archive is removed so uv extracts a fresh copy on the next install (verified
            in the gen2bench image). "changed" (a size differs) is reported only: our own post-install hooks patch
            some packages in place (PASTE, SPACEL), which is deliberate and the same for every run.
  runs   -- every run under a runs root, three kinds of problem kept apart:
            env_broken (cache damage, quarantine the run): a pure-Python module is missing from inside a
              .method-venv* (e.g. No module named 'six', 'scipy._lib._array_api'), or uv/pip hit "No space left on
              device". These depend on the node, so runs that hit them are not comparable with the others.
            build_failed: packages the prebootstrap could not compile ("Failed to build annoy hnswlib": the image
              has no g++, so STAligner's annoy/hnswlib fail on every run). The same everywhere; recorded only.
            image_missing_lib: a shared library the image lacks (libGL.so.1, which Spateo needs). Same everywhere.
            report_mentions_broken: the agent's own report.json says environments were broken (secondary signal).

    python3 src/genomics_agent_bench/env_health.py cache /tmp/p2b-uv-cache-$USER [--repair]
    python3 src/genomics_agent_bench/env_health.py runs agent_runs/hicks_sa5xx_20261023 --out env_health.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMPORT_FAIL = re.compile(r"(ModuleNotFoundError|ImportError)[^\n]*")
SYSTEM_LIB = re.compile(r"lib[\w.+-]*\.so(\.\d+)*")
MISSING_MODULE = re.compile(r"No module named '([\w.]+)'")
FAILED_BUILD = re.compile(r"Failed to build ([\w .-]+)")
MISSING_SOURCE = re.compile(r"Distribution not found at: file://\S*/([A-Za-z][\w]*)-[0-9a-f]{7,}")
CORE_PACKAGES = {"six", "dateutil", "sklearn", "scipy", "numpy", "pandas", "packaging", "joblib", "threadpoolctl",
                 "narwhals", "cloudpickle", "matplotlib", "h5py", "anndata", "cv2"}
BUNDLED_LIB = re.compile(r"lib(gfortran|gomp|openblas|quadmath|scipy_openblas)[\w.-]*-[0-9a-f]{8}")
IMAGE_LIBS = re.compile(r"lib(GL|GLX|EGL|X11|Xext|Xrender|SM|ICE|glib)[\w.-]*\.so")
IN_METHOD_ENV = re.compile(r"/\.method-venv[^/\s]*/")
NO_SPACE = re.compile(r"No space left on device")
REPORT_BROKEN = re.compile(r"truncated install|broken install|tmpfs|install(s|ation)? (was|were|are|is) (broken|truncated|incomplete)"
                           r"|(envs?|environments?)[^.\n]{0,80}(broken|truncated|failed to import|missing modules)", re.I)


def check_cache(cache: Path) -> list[dict]:
    """Archives whose files no longer match their RECORD."""
    root = cache / "archive-v0"
    out = []
    if not root.is_dir():
        return out
    for arch in sorted(root.iterdir()):
        records = list(arch.glob("*.dist-info/RECORD"))
        if not records:
            infos = list(arch.glob("*.dist-info"))
            if infos:   # a package whose own file list is gone: nothing left to link to (seen: pandas 3.0.6, 0 files)
                out.append({"archive": arch.name, "package": infos[0].name.removesuffix(".dist-info"),
                            "files": 0, "missing": 1, "changed": 0, "no_record": True})
            continue    # no dist-info at all: a cached uv environment, not a package
        total = missing = changed = 0
        for row in csv.reader(open(records[0], newline="", encoding="utf-8", errors="replace")):
            if not row or not row[0] or row[0].endswith("RECORD"):
                continue
            total += 1
            p = arch / row[0]
            if not p.exists():
                missing += 1
            elif len(row) >= 3 and row[2].isdigit() and p.is_file() and p.stat().st_size != int(row[2]):
                changed += 1
        if missing or changed:
            out.append({"archive": arch.name, "package": records[0].parent.name.removesuffix(".dist-info"),
                        "files": total, "missing": missing, "changed": changed})
    return out


def heavily_changed(d: dict) -> bool:
    """Many files differ from RECORD: overwritten through a symlink by another version (seen: scipy 1.11.4 with 611 of
    1267 files from a newer scipy). One to three changed files are our own post-install patches (PASTE, SPACEL) or
    small edits; those copies are left alone, since healthy running jobs may link to them."""
    return d["changed"] >= 10 or (d["changed"] >= 3 and d["files"] and d["changed"] / d["files"] >= 0.05)


def repair_cache(cache: Path, damaged: list[dict]) -> list[str]:
    """Remove archives with missing files, a lost RECORD, or heavily overwritten files; uv re-extracts them."""
    removed = []
    for d in damaged:
        if d["missing"] or heavily_changed(d):
            shutil.rmtree(cache / "archive-v0" / d["archive"], ignore_errors=True)
            removed.append(f'{d["package"]} ({d["archive"]}, {d["missing"]}/{d["files"]} files missing, {d["changed"]} changed)')
    return removed


def _tool_results(trace: Path):
    """Text of every tool result in a stream-json trace (claude, gemini and codex all write JSON lines)."""
    try:
        lines = trace.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    if len(lines) == 1 and lines[0][:2] in ("b'", 'b"'):      # a timed-out run's raw stdout is a bytes literal
        import ast
        try:
            lines = ast.literal_eval(lines[0]).decode("utf-8", "replace").splitlines()
        except (ValueError, SyntaxError):
            pass
    for line in lines:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        msg = d.get("message") if isinstance(d.get("message"), dict) else {}
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                t = c.get("content")
                yield t if isinstance(t, str) else json.dumps(t)
        for key in ("output", "aggregated_output", "stdout", "stderr"):     # codex / gemini tool events
            v = d.get(key) or (d.get("item") or {}).get(key) if isinstance(d.get("item"), dict) else d.get(key)
            if isinstance(v, str):
                yield v


def scan_run(run_dir: Path) -> dict:
    row = {"run": run_dir.name, "env_broken": False, "cache_damage_imports": 0, "no_space": 0, "build_failed": "",
           "image_missing_lib": "", "version_conflict": "", "wrong_env_imports": 0, "report_mentions_broken": False, "evidence": ""}
    failed_builds = set()
    boot = run_dir / "method_env_prebootstrap_stdout.txt"
    if boot.exists():
        for m in FAILED_BUILD.finditer(boot.read_text(encoding="utf-8", errors="replace")):
            failed_builds.update(m.group(1).split())
    err = run_dir / "method_env_prebootstrap_stderr.txt"
    if err.exists():
        for m in MISSING_SOURCE.finditer(err.read_text(encoding="utf-8", errors="replace")):
            failed_builds.add(m.group(1))       # a git source the container cannot see (STalign)
    row["build_failed"] = " ".join(sorted(failed_builds))
    conflicts = set()
    libs = set()
    for text in _tool_results(run_dir / "raw_stdout.jsonl"):
        if IN_METHOD_ENV.search(text):
            for m in IMPORT_FAIL.finditer(text):
                line = m.group(0)
                lib = IMAGE_LIBS.search(line)
                if lib:
                    libs.add(lib.group(0)); continue        # the image lacks it: same on every node
                if "cannot enable executable stack" in line:
                    libs.add("execstack:" + (lib.group(0) if (lib := SYSTEM_LIB.search(line)) else "?"))
                    continue                                # glibc 2.41 in the image refuses torch 2.3's execstack .so
                if re.search(r"requires '[\w.-]+'", line):
                    conflicts.add(line[:120]); continue     # an optional extension the env never installed (e.g. torch-cluster)
                if "cannot import name" in line:
                    conflicts.add(line[:120]); continue     # the module is there, its API is not: a version clash
                mod = MISSING_MODULE.search(line)
                if mod and mod.group(1).split(".")[0].lower() in {b.lower() for b in failed_builds}:
                    continue                      # never built, so nothing was damaged
                # Damage shows as an installed package failing to import its own pieces or dependencies: the last
                # frame before the error is inside site-packages. When it is the agent's own script, the agent
                # picked an environment without that package, which is a choice, not damage.
                frames = re.findall(r'File "([^"]+)"', text[:m.start()])
                if not frames or "/site-packages/" not in frames[-1]:
                    row["wrong_env_imports"] += 1
                    continue
                # Damage means a piece of an installed core package is gone (six, dateutil.*, sklearn.*, scipy.*,
                # a library bundled in a wheel); a whole optional package missing (rpy2, nrrd, torch_scatter, ...)
                # is the env spec's limitation and the same on every node.
                top = mod.group(1).split(".")[0] if mod else ""
                if not (top in CORE_PACKAGES or BUNDLED_LIB.search(line)):
                    conflicts.add(line[:120]); continue
                row["cache_damage_imports"] += 1
                row["evidence"] = row["evidence"] or line[:160]
        if NO_SPACE.search(text):
            row["no_space"] += 1
            row["evidence"] = row["evidence"] or "No space left on device"
    row["image_missing_lib"] = " ".join(sorted(libs))
    row["version_conflict"] = " | ".join(sorted(conflicts))
    rep = run_dir / "workspace_snapshot/submission/report.json"
    if rep.exists() and REPORT_BROKEN.search(rep.read_text(encoding="utf-8", errors="replace")):
        row["report_mentions_broken"] = True
    row["env_broken"] = bool(row["cache_damage_imports"] or row["no_space"])
    return row


def scan_runs(root: Path) -> list[dict]:
    rows = []
    for row_csv in sorted(root.glob("*/row.csv")):
        job = row_csv.parent.name
        agent, model, task, condition, rep = row_csv.read_text().strip().split(",")[:5]
        node = ""
        try:
            node = subprocess.run(["sacct", "-j", job, "-X", "-n", "-o", "NodeList"], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        for man in row_csv.parent.glob("*/manifest.json"):
            status = json.loads(man.read_text()).get("status")
            r = scan_run(man.parent)
            rows.append({"job": job, "node": node, "agent": agent, "model": model, "task": task,
                         "condition": condition, "rep": rep, "status": status, **r})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("cache", type=Path); c.add_argument("--repair", action="store_true")
    r = sub.add_parser("runs"); r.add_argument("root", type=Path); r.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.cmd == "cache":
        damaged = check_cache(a.cache)
        for d in damaged:
            print("damaged", d["package"], d["archive"], f'{d["missing"]}/{d["files"]} missing, {d["changed"]} changed')
        if a.repair:
            for x in repair_cache(a.cache, damaged):
                print("removed", x)
        n = sum(1 for _ in (a.cache / "archive-v0").iterdir()) if (a.cache / "archive-v0").is_dir() else 0
        print(f"{n} archives, {sum(1 for d in damaged if d['missing'] or heavily_changed(d))} damaged (missing, no RECORD or heavily changed), "
              f"{sum(1 for d in damaged if d['changed'] and not d['missing'])} with patched files")
        return
    rows = scan_runs(a.root)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["job"]); w.writeheader(); w.writerows(rows)
    broken = [x for x in rows if x["env_broken"]]
    print(f"{len(rows)} runs, {len(broken)} env_broken")
    for x in broken:
        print("BROKEN", x["job"], x["node"], x["model"], x["task"][:6], x["condition"], "|", x["evidence"])


if __name__ == "__main__":
    sys.exit(main())
