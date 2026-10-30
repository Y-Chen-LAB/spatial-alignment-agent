#!/usr/bin/env python3
"""Every failed or not-applicable cell of the baselines campaign, with its cause and resolution.

Reads only sacct, the submitted rows files and the logs of failed tasks (no filesystem scans).
Rerun any time; the last attempt of each (method, task) cell decides its status.

    python3 scripts/document_baseline_failures.py

Writes results/baselines_failures.csv (one row per cell that failed at least once or is not applicable) and
prints a summary by method and category.
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import re
import subprocess

C = "/dcs07/chen/data/alignment-agent/baselines_campaign"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/results/baselines_failures.csv"

# (category, pattern on the error text, how it is resolved, counts as a genuine method failure?)
CATEGORIES = [
    ("not_applicable_no_shared_features", r"NOT_APPLICABLE", "not applicable: slices share no features (e.g. RNA genes vs chromatin peaks)", "not applicable"),
    ("not_applicable_special_task", r"does not define runnable `slices`", "not applicable: task needs its own runner (sa-005 resolution recovery)", "not applicable"),
    ("runner_star_tasks", r"star tasks not implemented", "runner fixed (star mode added); cell resubmitted", "no (our runner)"),
    ("duplicate_gene_names", r"uniquely valued Index", "shared gene step fixed (duplicate names dropped, first kept); resubmitted", "no (data layout)"),
    ("gpu_out_of_memory", r"CUDA out of memory|OutOfMemoryError", "retried on larger GPUs, then with --max-obs-per-slice 20000", "no if a retry succeeds"),
    ("cpu_out_of_memory", r"OUT_OF_MEMORY|oom-kill|MemoryError|Killed", "retry once with more memory", "yes if the retry fails"),
    ("timeout", r"TIMEOUT|exit 124|DUE TO TIME LIMIT", "retry once with a longer limit", "yes if the retry fails"),
    ("illegal_instruction", r"Illegal instruction|SIGILL|exit 132", "restricted to icelake/sapphirerapids nodes", "no (build)"),
    ("degenerate_output", r"non-finite coordinates|NotEnoughPoints|Need at least 3 points|qhull Delaunay", "method returned NaN or collapsed coordinates; the scorer rejects them; recorded as a method failure", "yes"),
    ("method_internal_error", r"Traceback|Error|error", "method raised an error on this input; recorded as a method failure", "yes"),
]


def classify(text: str, state: str) -> tuple[str, str, str]:
    blob = f"{state}\n{text}"
    for cat, pat, how, genuine in CATEGORIES:
        if re.search(pat, blob):
            return cat, how, genuine
    return "unknown", "inspect the log", "unknown"


def last_error(log: str) -> str:
    if not os.path.exists(log):
        return ""
    lines = [x for x in open(log, errors="replace").read().splitlines()
             if x.strip() and "it/s]" not in x and "findfont" not in x]
    degen = [x for x in lines if re.search(r"non-finite coordinates|NotEnoughPoints|qhull Delaunay", x)]
    if degen:
        return degen[-1].strip()[:300]
    raw = open(log, errors="replace").read()
    killed = re.findall(r"exit 124 after \d+s", raw)  # the wall-time line often trails a progress bar
    if killed and not re.search(r"Traceback", raw[raw.rfind("exit 124"):]):
        return killed[-1]
    errs = [x for x in lines if re.search(r"Error|error:|NOT_APPLICABLE|Killed|CUDA|not implemented|does not define|exit \d+", x)]
    pick = [x for x in errs if not re.match(r"\[\d{4}-", x)] or errs
    return (pick[-1] if pick else (lines[-1] if lines else "")).strip()[:300]


def main() -> None:
    files = {re.search(r"_(\d+)\.tsv$", f).group(1): f for f in glob.glob(f"{C}/rows/submitted_*.tsv")}
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-o", "JobID,State,Elapsed,NodeList", "-j", ",".join(files)],
                         capture_output=True, text=True).stdout
    rows_cache: dict[str, list[str]] = {}
    attempts = collections.defaultdict(list)   # (method, task.toml) -> [(array, idx, state, elapsed, node)]
    for line in out.splitlines():
        jid, st, el, node = line.split("|")
        m = re.match(r"(\d+)_(\d+)$", jid)
        idxs = [int(m.group(2))] if m else []
        if not m:
            m = re.match(r"(\d+)_\[(.+?)(%\d+)?\]$", jid)   # still-pending ranges, e.g. 123_[4-9%2]
            if not m:
                continue
            for part in m.group(2).split(","):
                x, _, y = part.partition("-")
                idxs += list(range(int(x), int(y or x) + 1))
        a = m.group(1)
        if a not in rows_cache:
            rows_cache[a] = open(files[a]).read().splitlines()
        for i in idxs:
            r = rows_cache[a][i].split("\t")
            attempts[(r[0], r[2])].append((a, i, st.split()[0], el, node))

    na_file = f"{C}/not_applicable.tsv"
    na = []
    if os.path.exists(na_file):
        for line in open(na_file):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and not line.startswith("#") and p[0] != "task":
                na.append(p)

    records = []
    for (meth, task), atts in sorted(attempts.items()):
        failed = [x for x in atts if x[2] not in ("COMPLETED", "RUNNING", "PENDING", "CANCELLED")]
        if not failed:
            continue
        scored = os.path.exists(f"{REPO}/results/baselines/{os.path.basename(os.path.dirname(task))}/{meth}/scores.json")
        done = any(x[2] == "COMPLETED" for x in atts) or scored
        live = any(x[2] in ("RUNNING", "PENDING") for x in atts)
        a, i, st, el, node = failed[-1]
        log = (glob.glob(f"{C}/logs/*-{a}_{i}.out") or [""])[0]
        err = last_error(log)
        cat, how, genuine = classify(err, st)
        status = "resolved: later attempt succeeded" if done else ("retry queued or running" if live else "open")
        if cat.startswith("not_applicable"):
            status = "not applicable"
        records.append({"task": os.path.basename(os.path.dirname(task)), "method": meth, "category": cat,
                        "status": status, "genuine_method_failure": ("no" if done else genuine),
                        "n_failed_attempts": len(failed), "last_state": st, "elapsed": el, "node": node,
                        "error": err, "resolution": how, "log": log})
    for p in na:
        records.append({"task": p[0], "method": p[1], "category": "not_applicable_no_shared_features",
                        "status": "not applicable (not submitted)", "genuine_method_failure": "not applicable",
                        "n_failed_attempts": 0, "last_state": "", "elapsed": "", "node": "",
                        "error": p[2] if len(p) > 2 else "", "resolution": "not submitted", "log": ""})
    # Hand-recorded decisions (e.g. retries cancelled as hopeless), while the cell is still unscored.
    man_file = f"{C}/manual_status.tsv"
    if os.path.exists(man_file):
        for line in open(man_file):
            p = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(p) < 5:
                continue
            if os.path.exists(f"{REPO}/results/baselines/{p[0]}/{p[1]}/scores.json"):
                continue
            for r in records:
                if r["task"] == p[0] and r["method"] == p[1]:
                    r["status"], r["genuine_method_failure"] = p[2], p[3]
                    r["resolution"] = p[4]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]) if records else ["task"])
        w.writeheader()
        w.writerows(records)

    by = collections.Counter((r["method"], r["category"], r["status"]) for r in records)
    print(f"{len(records)} cells with a failure or not-applicable status -> {OUT}")
    for (m, c, s), n in sorted(by.items()):
        print(f"  {m:20s} {c:34s} {s:32s} {n}")


if __name__ == "__main__":
    main()
