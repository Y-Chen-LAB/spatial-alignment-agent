"""Coverage matrix for the all-tasks method-baseline campaign.

For every non-smoke task x each of the 17 non-agent methods, one row: done / failed / timeout / oom /
not_applicable / running / pending, with the failure stage and reason, runtime, node and device.
Reads only the specific per-cell paths (results/baselines/<task>/<variant>/, the campaign status JSONs and logs)
plus sacct for the campaign arrays; no filesystem scans.

    .venv/bin/python scripts/baseline_campaign_report.py [--out results/baselines_coverage.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = Path("/dcs07/chen/data/alignment-agent/baselines_campaign")
RESULTS = REPO / "results" / "baselines"
TASKS = REPO / "benchmarks" / "spatial_alignment" / "tasks"

# method -> (primary output dir, extra variants that the same run writes or that were run alongside)
METHODS = {
    "identity": ("identity", []), "centroid_similarity": ("centroid_similarity", []),
    "pca_similarity": ("pca_similarity", []), "icp_similarity": ("icp_similarity", []),
    "PASTE": ("paste", ["paste_noflip"]), "PASTE2": ("paste2", []), "Spateo": ("spateo", ["spateo_rigidpart"]),
    "SPACEL": ("spacel", []), "GEASO": ("geaso", ["geaso_nonrigid", "geaso_rigid"]),
    "CAST": ("cast", ["cast_affine", "cast_rescale", "cast_rescale_affine"]), "STAligner": ("staligner", []),
    "SLAT": ("slat", ["slat_lsq"]), "STalign": ("stalign", ["stalign_expr"]), "STAIR": ("stair", []),
    "SANTO": ("santo", ["santo_coarse"]), "GPSA": ("gpsa", []), "Moscot": ("moscot", ["moscot_warp"]),
}
ERR_PAT = re.compile(r"(Error|Exception|SystemExit|NOT_APPLICABLE|Killed|Illegal instruction|oom|out of memory|"
                     r"CANCELLED|DUE TO TIME LIMIT|Traceback)", re.I)


def sacct_states() -> dict[str, tuple[str, int]]:
    ids = []
    log = CAMPAIGN / "submissions.log"
    if log.exists():
        ids = [ln.split(" array ")[1].split()[0] for ln in log.read_text().splitlines() if " array " in ln]
    out = {}
    if not ids:
        return out
    r = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "--format=JobID,State,ElapsedRaw"],
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        jid, st, el = (ln.split("|") + ["", "", ""])[:3]
        out[jid] = (st.split()[0] if st else "", int(el or 0))
    return out


def log_reason(path: Path) -> str:
    if not path or not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()[-200:]
    except OSError:
        return ""
    hits = [ln.strip() for ln in lines if ERR_PAT.search(ln) and "Warning" not in ln]
    return (hits[-1] if hits else (lines[-1].strip() if lines else ""))[:300]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "results" / "baselines_coverage.csv")
    args = ap.parse_args()
    na = {}
    if (CAMPAIGN / "not_applicable.tsv").exists():
        for r in csv.DictReader(open(CAMPAIGN / "not_applicable.tsv"), delimiter="\t"):
            na[(r["task"], r["variant"])] = r["reason"]
    sa = sacct_states()
    rows = []
    tasks = sorted(d.name for d in TASKS.iterdir() if (d / "task.toml").exists() and not d.name.endswith("-smoke"))
    for tid in tasks:
        for method, (primary, extras) in METHODS.items():
            d = RESULTS / tid / primary
            row = {"task": tid, "method": method, "variant": primary, "state": "", "stage": "", "reason": "",
                   "runtime_s": "", "device": "", "job": "", "variants_done": ""}
            done_vars = [v for v in [primary] + extras if (RESULTS / tid / v / "scores.json").exists()]
            row["variants_done"] = " ".join(done_vars)
            st = {}
            sp = CAMPAIGN / "status" / f"{tid}__{primary}.json"
            if sp.exists():
                try:
                    st = json.loads(sp.read_text())
                except json.JSONDecodeError:
                    st = {}
            row["job"] = st.get("job", "")
            if (d / "scores.json").exists():
                row["state"] = "done"
                try:
                    man = json.loads((d / "run_manifest.json").read_text())
                    row["runtime_s"] = round(float(man.get("runtime_seconds") or man.get("runtime_s") or 0), 1)
                    gpu = man.get("gpu") or (man.get("settings", {}) or {}).get("device") or man.get("device")
                    row["device"] = gpu or "cpu"
                except Exception:
                    row["runtime_s"] = st.get("elapsed_s", "")
                    row["device"] = "cpu" if method in METHODS and primary in (
                        "identity", "centroid_similarity", "pca_similarity", "icp_similarity") else ""
            elif (tid, primary) in na:
                row.update(state="not_applicable", reason=na[(tid, primary)])
            elif st:
                slurm_state, el = sa.get(st.get("job", ""), ("", 0))
                logp = Path(st["log"]) if st.get("log") else None
                if logp is None and st.get("job"):
                    arr, idx = st["job"].split("_")
                    cand = list((CAMPAIGN / "logs").glob(f"p2b-bl-*-{arr}_{idx}.out"))
                    logp = cand[0] if cand else None
                reason = log_reason(logp) if logp else ""
                if "NOT_APPLICABLE" in reason:
                    row.update(state="not_applicable", reason=reason.split("NOT_APPLICABLE:")[-1].strip())
                elif slurm_state in ("RUNNING", "PENDING", "REQUEUED") or (st.get("state") == "running" and not slurm_state):
                    row["state"] = "running"
                else:
                    state = st.get("state", "failed")
                    if slurm_state == "OUT_OF_MEMORY" or st.get("exit_code") in (137,) or "out of memory" in reason.lower():
                        state = "oom"
                    elif slurm_state == "TIMEOUT" or st.get("exit_code") == 124:
                        state = "timeout"
                    elif state == "running":
                        state = "failed"
                    stage = "scoring" if (d / "aligned_coordinates").is_dir() else "method"
                    row.update(state=state, stage=stage, reason=reason or slurm_state,
                               runtime_s=st.get("elapsed_s", el))
            else:
                row["state"] = "pending" if not tid.startswith("sa-3") else "missing"
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["state"] for r in rows)
    print(f"{len(tasks)} tasks x {len(METHODS)} methods = {len(rows)} cells:", dict(c))
    by_m = {}
    for r in rows:
        by_m.setdefault(r["method"], Counter())[r["state"]] += 1
    for m, cc in by_m.items():
        print(f"  {m:20s}", dict(cc))


if __name__ == "__main__":
    main()
