#!/usr/bin/env python3
"""Clean agent traces and recover the final solution code for each scored run.

For every run in the index (scripts/agent_strategies/build_index.py) this writes, under
/dcs07/chen/data/alignment-agent/agent_traces/<run_dir>/:

  trace.md          cleaned transcript (agent narration in full, tool calls trimmed to ~40 lines,
                    tool results to ~15 lines, step numbers, timestamps, tokens)
  final_solution/   the code that wrote submission/aligned_coordinates (+ local helpers and
                    upstream stage scripts), .py/.R/.sh only, each < 1 MB, plus SOLUTION.md
  steps.json        normalized steps (tool inputs kept in full, outputs trimmed) for tagging
  meta.json         run metadata + exploration stats (+ strategy tags once tag_strategies.py ran)

Three parsers, one per agent CLI:
  claude  Claude Code stream-json  (assistant text/thinking/tool_use, user tool_result, timestamps)
  codex   `codex exec --json`      (item.completed agent_message/reasoning/command_execution/
                                    file_change; no per-event timestamps)
  gemini  Gemini CLI stream-json   (message deltas, tool_use/tool_result with timestamps)

Resource rules: inside a run dir only manifest/scores/prompt/raw_stdout are read, and
workspace_snapshot is listed at depth <= 2. Only code files (.py/.R/.sh, < 1 MB) are opened;
data files are never opened. Runs already extracted are skipped unless --force.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import TRACES_ROOT, build as build_index  # noqa: E402

CODE_EXT = (".py", ".R", ".r", ".sh")
MAX_CODE_BYTES = 1_000_000
CMD_LINES = 40
OUT_LINES = 15
# Files the harness puts in every workspace; never part of an agent's solution.
HARNESS_FILES = {"agent_python.sh", "bootstrap_method_venv.sh", "spatial_method_envs.toml",
                 "TASK.md", "scores.json", "AGENTS.md", "GEMINI.md", "CLAUDE.md"}
SKIP_DIRS = {".uv-task-cache", ".venv", "venv", "__pycache__", ".cache", "data", ".git",
             "node_modules", ".ipynb_checkpoints"}

# --------------------------------------------------------------------------- cleaning
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
SECRET_PATTERNS = [
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"ya29\.[0-9A-Za-z_\-\.]{20,}"),                      # Google OAuth access token
    re.compile(r"1//0[0-9A-Za-z_\-]{30,}"),                           # Google OAuth refresh token
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"xox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{20,}"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|"
               r"password|secret)\s*[:=]\s*['\"]?)[^\s'\"]{8,}"),
]
PROGRESS = re.compile(r"\d+%\||it/s|s/it|\bEpoch\s*\d+\s*/\s*\d+|\b(?:epoch|iter(?:ation)?|step)\s*[:=]?\s*\d+.*\bloss\b|\bloss\s*[:=]\s*[-\d.e]+", re.I)
BASE64ISH = re.compile(r"[A-Za-z0-9+/=_\-]{200,}")


def redact(s: str) -> str:
    for p in SECRET_PATTERNS:
        if p.groups:
            s = p.sub(lambda m: m.group(1) + "[REDACTED]", s)
        else:
            s = p.sub("[REDACTED]", s)
    return s


def clean_text(s, keep_long_tokens: bool = False) -> str:
    """Strip ANSI, progress-bar carriage returns, base64 blobs, secrets; collapse repeats."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = ANSI.sub("", s)
    lines = []
    for ln in s.split("\n"):
        if "\r" in ln:  # progress bars redraw with \r: keep only the final frame
            parts = [p for p in ln.split("\r") if p.strip()]
            ln = parts[-1] if parts else ""
        if not keep_long_tokens:
            ln = BASE64ISH.sub(lambda m: f"[{len(m.group(0))}-char blob dropped]", ln)
        lines.append(ln.rstrip())
    # Collapse only (a) exact duplicate consecutive lines and (b) runs of progress-bar /
    # training-log lines (tqdm "%|", "it/s", "Epoch k/N", "loss=") that differ only in digits.
    # Everything else is kept; long outputs are trimmed later by line count.
    out, i = [], 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i] and lines[i].strip():
            j += 1
        if j > i:
            out.append(lines[i])
            out.append(f"    [... same line repeated {j - i}x ...]")
            i = j + 1
            continue
        if PROGRESS.search(lines[i]):
            key = re.sub(r"\d+(?:\.\d+)?", "#", lines[i])
            while j + 1 < len(lines) and PROGRESS.search(lines[j + 1]) and re.sub(r"\d+(?:\.\d+)?", "#", lines[j + 1]) == key:
                j += 1
            if j - i + 1 > 3:
                out += [lines[i], f"    [... {j - i - 1} progress lines collapsed ...]", lines[j]]
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return redact("\n".join(out))


def trim(s: str, n: int, head_frac: float = 0.67) -> str:
    lines = s.split("\n")
    if len(lines) <= n:
        out = s
    else:
        h = max(1, int(n * head_frac))
        t = n - h
        out = "\n".join(lines[:h] + [f"[... {len(lines) - n} lines trimmed ...]"] + (lines[-t:] if t else []))
    # cap very long single lines too
    return "\n".join(ln if len(ln) <= 400 else ln[:300] + f" [...{len(ln) - 300} chars]" for ln in out.split("\n"))


# --------------------------------------------------------------------------- parsing
def _step(kind, **kw):
    d = {"kind": kind, "ts": None, "name": None, "text": None, "command": None, "path": None,
         "content": None, "output": None, "ok": None, "out_tokens": None, "sub": False}
    d.update(kw)
    return d


def _tool_input(name: str, inp: dict) -> dict:
    """Normalize a tool call input into command/path/content."""
    n = (name or "").lower()
    if not isinstance(inp, dict):
        return {"command": str(inp)}
    if "command" in inp and isinstance(inp["command"], str):
        return {"command": inp["command"]}
    if n in ("write", "write_file") or ("content" in inp and ("file_path" in inp or "path" in inp)):
        return {"path": inp.get("file_path") or inp.get("path"), "content": inp.get("content", "")}
    if n in ("edit", "multiedit", "replace", "str_replace"):
        p = inp.get("file_path") or inp.get("path")
        if "edits" in inp:
            body = "\n".join(f"- old:\n{e.get('old_string', '')}\n+ new:\n{e.get('new_string', '')}" for e in inp["edits"])
        else:
            body = f"- old:\n{inp.get('old_string', '')}\n+ new:\n{inp.get('new_string', '')}"
        return {"path": p, "command": body, "edit": True}
    if n in ("read", "read_file", "list_directory", "glob", "grep", "search_file_content"):
        return {"command": json.dumps(inp)}
    if n == "notebookedit":
        return {"path": inp.get("notebook_path"), "command": inp.get("new_source", "")}
    if n in ("task", "agent"):
        return {"command": f"[subagent] {inp.get('description', '')}\n{inp.get('prompt', '')}"}
    return {"command": json.dumps(inp)[:4000]}


def parse_claude(lines):
    steps, by_id, seen_msg, msg_tok = [], {}, set(), {}
    for l in lines:
        try:
            e = json.loads(l)
        except Exception:
            continue
        t = e.get("type")
        ts = e.get("timestamp")
        sub = bool(e.get("parent_tool_use_id"))
        if t == "assistant":
            msg = e.get("message") or {}
            mid = msg.get("id")
            # stream-json repeats a message's usage on every content block with a growing
            # count; keep the max and put it on the message's first step
            ot = (msg.get("usage") or {}).get("output_tokens") or 0
            msg_tok[mid] = max(msg_tok.get(mid, 0), ot)
            seen_msg.add(mid)
            for b in msg.get("content") or []:
                bt = b.get("type")
                if bt == "text" and b.get("text", "").strip():
                    steps.append(_step("text", ts=ts, text=b["text"], sub=sub, mid=mid))
                elif bt == "thinking" and (b.get("thinking") or "").strip():
                    steps.append(_step("thinking", ts=ts, text=b["thinking"], sub=sub, mid=mid))
                elif bt == "tool_use":
                    s = _step("tool", ts=ts, name=b.get("name"), sub=sub, mid=mid, **_tool_input(b.get("name"), b.get("input")))
                    steps.append(s)
                    by_id[b.get("id")] = s
        elif t == "user":
            content = (e.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if b.get("type") != "tool_result":
                    continue
                s = by_id.get(b.get("tool_use_id"))
                if s is None:
                    continue
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "[image]") if x.get("type") == "text" else f"[{x.get('type')}]" for x in c)
                s["output"] = c if isinstance(c, str) else str(c)
                s["ok"] = not b.get("is_error", False)
                s["ts_end"] = ts
    done = set()
    for s in steps:
        if s.get("mid") and s["mid"] not in done:
            done.add(s["mid"])
            s["out_tokens"] = msg_tok.get(s["mid"]) or None
    return steps


_BASH_WRAP = re.compile(r"^/bin/(?:ba)?sh\s+-lc\s+")


def _unwrap_codex_cmd(cmd: str) -> str:
    m = _BASH_WRAP.match(cmd or "")
    if not m:
        return cmd
    rest = cmd[m.end():]
    try:
        parts = shlex.split(rest)
        if len(parts) == 1:
            return parts[0]
    except ValueError:
        pass
    return rest


def parse_codex(lines):
    steps = []
    for l in lines:
        try:
            e = json.loads(l)
        except Exception:
            continue
        if e.get("type") != "item.completed":
            continue
        it = e.get("item") or {}
        ty = it.get("type")
        if ty == "agent_message":
            steps.append(_step("text", text=it.get("text", "")))
        elif ty == "reasoning":
            if (it.get("text") or "").strip():
                steps.append(_step("thinking", text=it["text"]))
        elif ty == "command_execution":
            steps.append(_step("tool", name="shell", command=_unwrap_codex_cmd(it.get("command", "")),
                               output=it.get("aggregated_output", ""), ok=(it.get("exit_code") == 0),
                               exit_code=it.get("exit_code")))
        elif ty == "file_change":
            ch = it.get("changes") or []
            if isinstance(ch, str):
                desc = ch
                paths = re.findall(r"'path': '([^']+)'", ch)
            else:
                desc = "\n".join(f"{c.get('kind')}: {c.get('path')}" for c in ch)
                paths = [c.get("path") for c in ch]
            steps.append(_step("tool", name="apply_patch", command=desc, path=paths[0] if paths else None,
                               paths=paths, ok=it.get("status") == "completed", edit=True))
        elif ty in ("web_search", "mcp_tool_call", "todo_list"):
            steps.append(_step("tool", name=ty, command=json.dumps({k: v for k, v in it.items() if k not in ("id", "type")})[:2000], ok=True))
        elif ty == "error":
            steps.append(_step("text", text=f"[error] {it.get('message', '')}"))
    return steps


def parse_gemini(lines):
    steps, by_id, buf = [], {}, None
    for l in lines:
        try:
            e = json.loads(l)
        except Exception:
            continue
        t = e.get("type")
        ts = e.get("timestamp")
        if t == "message":
            if e.get("role") != "assistant":
                buf = None
                continue
            if buf is not None and e.get("delta"):
                buf["text"] += e.get("content", "")
            else:
                buf = _step("text", ts=ts, text=e.get("content", ""))
                steps.append(buf)
            continue
        buf = None
        if t == "tool_use":
            s = _step("tool", ts=ts, name=e.get("tool_name"), **_tool_input(e.get("tool_name"), e.get("parameters")))
            steps.append(s)
            by_id[e.get("tool_id")] = s
        elif t == "tool_result":
            s = by_id.get(e.get("tool_id"))
            if s is None:
                continue
            out = e.get("output")
            if e.get("error"):
                out = f"{out or ''}\n[error] {e.get('error')}"
            s["output"] = out if isinstance(out, str) else json.dumps(out)
            s["ok"] = e.get("status") == "success" and not e.get("error")
            s["ts_end"] = ts
    return [s for s in steps if s["kind"] != "text" or s["text"].strip()]


PARSERS = {"claude": parse_claude, "codex": parse_codex, "gemini": parse_gemini}

# --------------------------------------------------------------------------- code recovery
HEREDOC_FILE = re.compile(r"cat\s*>\s*(\S+\.(?:py|R|sh))\s*<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\s*\2\s*(?:\n|$)", re.S)
HEREDOC_PY = re.compile(r"(?:python3?|agent_python\.sh|\.\/agent_python\.sh)[^\n<]*?<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\s*\1\s*(?:\n|$)", re.S)
PY_C = re.compile(r"(?:python3?|agent_python\.sh)\s+-c\s+(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')", re.S)
RUN_SCRIPT = re.compile(r"(?:python3?(?:\.\d+)?|agent_python\.sh|uv\s+run\s+(?:--\S+\s+)*python|Rscript|bash|sh)\s+(?:-[uOBX]\s+)*([\w./\-]+\.(?:py|R|sh))")
CD_PREFIX = re.compile(r"(?:^|&&|;)\s*cd\s+([\w./\-]+)")
WRITES_SUBMISSION = re.compile(r"aligned_coordinates")
# A write of coordinate CSVs (json.dump of report.json and np.save of caches do not count).
SUB_WRITE = re.compile(r"to_csv\(|savetxt\(|csv\.writer|\.writerows?\(|write_csv|copyfile|shutil\.copy|\.write\([^\n]*(?:obs_id|x,y)|['\"]obs_id,x,y")
WRITE_CALL = re.compile(r"to_csv|savetxt|\.write\(|open\([^)]*['\"]w|csv\.writer|write_csv|np\.save|shutil\.copy|cp\s|mv\s")


def inline_code(cmd: str) -> str:
    """Python code embedded in a shell command (heredoc or -c)."""
    parts = [m.group(2) for m in HEREDOC_PY.finditer(cmd or "")]
    for m in PY_C.finditer(cmd or ""):
        q = m.group(1)
        try:
            parts.append(shlex.split(q)[0])
        except ValueError:
            parts.append(q[1:-1])
    return "\n\n".join(parts)


def heredoc_files(cmd: str):
    return [(m.group(1), m.group(3)) for m in HEREDOC_FILE.finditer(cmd or "")]


def rel_to_ws(p: str, ws_abs: str) -> str:
    if not p:
        return p
    p = p.strip().strip("'\"")
    if ws_abs and p.startswith(ws_abs.rstrip("/") + "/"):
        p = p[len(ws_abs.rstrip("/")) + 1:]
    elif p.startswith("/"):
        # some other absolute path that still points into a workspace copy
        m = re.search(r"/p2b_workspaces/[^/]+/[^/]+/(.*)", p)
        if m:
            p = m.group(1)
    if p.startswith("./"):
        p = p[2:]
    return p


def list_snapshot(snap: Path):
    """Depth <= 2 listing of workspace_snapshot. Returns {relpath: (size, mtime)} for code files."""
    files = {}
    try:
        top = list(os.scandir(snap))
    except OSError:
        return files
    for e in top:
        try:
            if e.is_file(follow_symlinks=False):
                if e.name.endswith(CODE_EXT) and e.name not in HARNESS_FILES:
                    st = e.stat()
                    files[e.name] = (st.st_size, st.st_mtime)
            elif e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIRS and not e.name.endswith(("_env", "venv")):
                for f in os.scandir(e.path):
                    if f.is_file(follow_symlinks=False) and f.name.endswith(CODE_EXT):
                        st = f.stat()
                        files[f"{e.name}/{f.name}"] = (st.st_size, st.st_mtime)
        except OSError:
            continue
    return files


def read_code(snap: Path, rel: str) -> str | None:
    p = snap / rel
    try:
        if p.stat().st_size > MAX_CODE_BYTES:
            return None
        return p.read_text(errors="replace")
    except OSError:
        return None


IMPORT_RE = re.compile(r"(?:^|;)[ \t]*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", re.M)
SOURCE_RE = re.compile(r"""(?:source|exec\(open)\(?\s*['"]([\w./\-]+\.(?:R|py))""")


def local_imports(code: str, here: str, snap_files: dict) -> list[str]:
    names = set()
    for m in IMPORT_RE.finditer(code):
        if m.group(1):
            names.add(m.group(1).split(".")[0])
        else:
            for n in m.group(2).split(","):
                n = n.strip().split(" ")[0]
                if n:
                    names.add(n.split(".")[0])
    out = []
    d = os.path.dirname(here)
    for n in names:
        for cand in (f"{d}/{n}.py" if d else f"{n}.py", f"{n}.py", f"work/{n}.py", f"{n}/__init__.py"):
            if cand in snap_files:
                out.append(cand)
                break
    for m in SOURCE_RE.finditer(code):
        c = rel_to_ws(m.group(1), "")
        for cand in (c, f"{d}/{c}" if d else c):
            if cand in snap_files:
                out.append(cand)
    return sorted(set(out))


DATA_LIT = re.compile(r"""['"]([\w./\-{}]+\.(?:npz|npy|json|pkl|pickle|csv|tsv|txt|h5|parquet|joblib|pt))['"]""")
LOAD_HINT = re.compile(r"np\.load|json\.load|pickle\.load|pd\.read_|joblib\.load|torch\.load|read_csv|loadtxt|open\(")


CMD_DATA_ARG = re.compile(r"([\w./\-]+\.(?:npz|npy|json|pkl|pickle|csv|tsv|txt|h5|parquet|joblib|pt))\b")


def upstream_scripts(entry_code: str, entry: str, codes: dict, cmd: str = "") -> list[str]:
    """Scripts that write an intermediate file (by name) which the entry script loads,
    either named in the entry's code or passed to it on the final command line."""
    if not LOAD_HINT.search(entry_code):
        return []
    # f-string names ('theta_{t}.json') are matched on their literal prefix
    lits = {os.path.basename(x).split("{")[0] or os.path.basename(x) for x in DATA_LIT.findall(entry_code)}
    lits |= {os.path.basename(x) for x in CMD_DATA_ARG.findall(cmd or "") if "aligned_coordinates" not in x}
    lits = {x for x in lits if len(x.split(".")[0]) >= 3}
    lits = {x for x in lits if not x.startswith(("s0", "obs")) and "aligned_coordinates" not in x}
    out = []
    for rel, code in codes.items():
        if rel == entry or code is None:
            continue
        for lit in lits:
            if lit in code and re.search(r"(np\.save|savez|json\.dump|pickle\.dump|to_csv|savetxt|joblib\.dump|torch\.save)[^\n]*" + re.escape(lit.split(".")[0].rstrip("_")), code):
                out.append(rel)
                break
    return sorted(set(out))


# --------------------------------------------------------------------------- per run
def _ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def process_run(row: dict, force: bool = False) -> dict:
    out_dir = TRACES_ROOT / "runs" / row["run_id"]
    if (out_dir / "meta.json").exists() and not force:
        return {"run_id": row["run_id"], "status": "skipped"}
    rd = Path(row["path"])
    try:
        man = json.loads((rd / "manifest.json").read_text())
    except Exception:
        man = {}
    ws_abs = man.get("workspace") or ""
    raw = rd / "raw_stdout.jsonl"
    try:
        with open(raw, errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        lines = []
    if len(lines) == 1 and lines[0][:2] in ("b'", 'b"'):
        # On an agent timeout the harness writes str(exc.stdout): the partial output as one bytes literal.
        try:
            lines = ast.literal_eval(lines[0]).decode("utf-8", "replace").splitlines(keepends=True)
        except (ValueError, SyntaxError):
            pass
    steps = PARSERS.get(row["agent"], parse_claude)(lines)
    parse_ok = any(s["kind"] == "tool" for s in steps)

    for i, s in enumerate(steps, 1):
        s["n"] = i
        if s.get("path"):
            s["path"] = rel_to_ws(s["path"], ws_abs)

    snap = rd / "workspace_snapshot"
    snap_files = list_snapshot(snap)
    codes = {rel: read_code(snap, rel) for rel, (sz, _) in snap_files.items() if sz <= MAX_CODE_BYTES}

    # latest full content of each code file written through the trace (fallback if not in snapshot)
    trace_written: dict[str, str] = {}
    written_paths: set[str] = set()
    for s in steps:
        if s["kind"] != "tool":
            continue
        if s.get("content") is not None and s.get("path"):
            trace_written[s["path"]] = s["content"]
            if s["path"].endswith(CODE_EXT):
                written_paths.add(s["path"])
        for p in s.get("paths") or []:
            rp = rel_to_ws(p, ws_abs)
            if rp.endswith(CODE_EXT):
                written_paths.add(rp)
        if s.get("edit") and s.get("path", "") and s["path"].endswith(CODE_EXT):
            written_paths.add(s["path"])
        for p, body in heredoc_files(s.get("command") or ""):
            rp = rel_to_ws(p, ws_abs)
            trace_written[rp] = body
            written_paths.add(rp)

    def code_of(rel):
        return codes.get(rel) if codes.get(rel) is not None else trace_written.get(rel)

    def resolve(script: str, cmd: str):
        script = rel_to_ws(script, ws_abs)
        cands = [script]
        for d in CD_PREFIX.findall(cmd):
            d = rel_to_ws(d, ws_abs).rstrip("/")
            if d and d not in (".", ws_abs):
                cands.insert(0, f"{d}/{script}")
        cands += [f"work/{script}", os.path.basename(script), f"work/{os.path.basename(script)}"]
        for c in cands:
            if c in codes or c in trace_written:
                return c
        return None

    all_files = {**snap_files, **{k: 0 for k in trace_written}}

    def self_writes(c):
        return bool((WRITES_SUBMISSION.search(c) or "submission" in c) and SUB_WRITE.search(c))

    def writer_funcs(c):
        """Top-level functions whose body writes coordinate CSVs."""
        names = set()
        for m in re.finditer(r"^def\s+(\w+)\s*\((.*?)(?=^\S)", c + "\nEND", re.S | re.M):
            if SUB_WRITE.search(m.group(2)):
                names.add(m.group(1))
        return names

    def writes_sub(rel, depth=0):
        """Script writes submission CSVs itself, or calls a writer function of a local helper."""
        c = code_of(rel) or ""
        if self_writes(c):
            return True
        if depth < 2:
            for h in local_imports(c, rel, all_files):
                hc = code_of(h) or ""
                if self_writes(hc) and any(re.search(r"\b" + f + r"\s*\(", c) for f in writer_funcs(hc)):
                    return True
                if writes_sub(h, depth + 1) and re.search(r"\b(main|run)\s*\(", c):
                    return True
        return False

    # classify tool steps: runs which script? writes submission?
    sub_steps = []
    for s in steps:
        if s["kind"] != "tool" or not s.get("command") or s.get("edit"):
            continue
        cmd = s["command"]
        ran = []
        for m in RUN_SCRIPT.finditer(cmd):
            r = resolve(m.group(1), cmd)
            if r and os.path.basename(r) not in HARNESS_FILES:
                ran.append(r)
        s["ran_scripts"] = ran
        inl = inline_code(cmd)
        s["inline"] = bool(inl)
        writes = False
        how = None
        if any(writes_sub(r) for r in ran):
            writes, how = True, "script"
        elif WRITES_SUBMISSION.search(cmd) and (SUB_WRITE.search(inl or "") or re.search(r"(^|[;&|\s])(cp|mv|rsync)\s[^\n]*aligned_coordinates", cmd)):
            writes, how = True, ("inline" if inl and SUB_WRITE.search(inl) else "copy")
        if writes and s.get("ok") is not False:
            s["writes_submission"] = how
            sub_steps.append(s)

    # final solution
    final = {"entry": None, "how": None, "step": None, "command": None, "files": [], "inline_file": None,
             "note": ""}
    if sub_steps:
        last = sub_steps[-1]
        final.update(step=last["n"], command=last["command"], how=last["writes_submission"])
        if last["writes_submission"] == "copy":
            # find the step that produced the copied files: the previous submission write or a script
            prev = [s for s in sub_steps[:-1] if s["writes_submission"] != "copy"]
            src = re.findall(r"(?:cp|mv|rsync)\s+(?:-\w+\s+)*([\w./\-*]+)", last["command"])
            src_dirs = {os.path.basename(os.path.dirname(x.rstrip("/*"))) or x for x in src}
            prod = None
            for rel in sorted(codes, key=lambda r: -snap_files.get(r, (0, 0))[1]):
                c = codes.get(rel) or ""
                if any(d and len(d) > 2 and d in c for d in src_dirs):
                    prod = rel
                    break
            final["note"] = "last write copied candidate files into submission/; entry is the script producing them"
            if prod:
                final["entry"] = prod
            elif prev:
                last = prev[-1]
                final.update(step=last["n"], command=last["command"], how=last["writes_submission"])
        if final["entry"] is None and last.get("ran_scripts"):
            ws = [r for r in last["ran_scripts"] if writes_sub(r)]
            final["entry"] = (ws or last["ran_scripts"])[-1]
        if final["entry"] is None and last.get("inline"):
            final["inline_file"] = f"step{last['n']:03d}_inline.py"
    if final["entry"] is None and final["inline_file"] is None:
        # fallback: most recently modified agent script that references the submission dir
        cands = [r for r in codes if writes_sub(r)]
        if cands:
            final["entry"] = max(cands, key=lambda r: snap_files[r][1])
            final["how"] = final["how"] or "fallback-mtime"
            final["note"] = (final["note"] + "; " if final["note"] else "") + \
                "no submission-writing command matched in the trace; chose the newest script referencing aligned_coordinates"

    files = []
    if final["entry"]:
        todo = [final["entry"]]
        while todo:
            f = todo.pop()
            if f in files:
                continue
            files.append(f)
            todo += local_imports(code_of(f) or "", f, {**snap_files, **{k: 0 for k in trace_written}})
        ups = upstream_scripts(code_of(final["entry"]) or "", final["entry"], {r: code_of(r) for r in set(codes) | set(trace_written) if r.endswith(CODE_EXT)}, final.get("command") or "")
        for u in ups:
            if u not in files:
                files.append(u)
                for m in local_imports(code_of(u) or "", u, {**snap_files, **{k: 0 for k in trace_written}}):
                    if m not in files:
                        files.append(m)
    final["files"] = files
    final["upstream"] = [f for f in files if f in (ups if final["entry"] else [])] if final["entry"] else []

    # ---- exploration stats
    tools = [s for s in steps if s["kind"] == "tool"]
    t0 = _ts(man.get("started_at"))
    first_ts = next((_ts(s["ts"]) for s in steps if s.get("ts")), None)
    t_first_sub, t_method = None, None
    if sub_steps:
        fs = sub_steps[0]
        te = _ts(fs.get("ts_end") or fs.get("ts"))
        base = first_ts or t0
        if te and base:
            t_first_sub, t_method = te - base, "timestamp"
        elif man.get("elapsed_seconds"):
            t_first_sub = man["elapsed_seconds"] * (tools.index(fs) + 1) / max(1, len(tools))
            t_method = "step-fraction-of-wall"
    usage = man.get("usage") or {}
    scripts_written = sorted(set(written_paths) | {r for r, (sz, mt) in snap_files.items() if t0 is None or mt > t0 + 5})
    stats = {
        "n_steps": len(steps),
        "n_tool_calls": len(tools),
        "n_tool_errors": sum(1 for s in tools if s.get("ok") is False),
        "n_text_msgs": sum(1 for s in steps if s["kind"] == "text"),
        "n_scripts_written": len(scripts_written),
        "n_script_runs": sum(len(s.get("ran_scripts") or []) for s in tools),
        "n_inline_code_runs": sum(1 for s in tools if s.get("inline")),
        "n_submission_writes": len(sub_steps),
        "n_submission_rewrites": max(0, len(sub_steps) - 1),
        "t_first_submission_s": round(t_first_sub, 1) if t_first_sub is not None else None,
        "t_first_submission_method": t_method,
        "wall_s": man.get("elapsed_seconds"),
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_output_tokens"),
        "cost_usd": usage.get("cost_usd"),
        "n_web_search": sum(1 for s in tools if re.search(r"web_?search|websearch|webfetch|web_fetch", (s.get("name") or ""), re.I)),
        "n_subagent_steps": sum(1 for s in steps if s.get("sub")),
    }

    # ---- write outputs
    if out_dir.exists() and force:
        shutil.rmtree(out_dir / "final_solution", ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs_dir = out_dir / "final_solution"
    fs_dir.mkdir(exist_ok=True)
    copied = []

    def safe_rel(f):
        # files written outside the workspace (e.g. /tmp/x.py) go under _outside_workspace/
        if f.startswith("/"):
            return "_outside_workspace" + os.path.normpath(f)
        return os.path.normpath(f).lstrip("./") if not f.startswith("..") else "_outside_workspace/" + os.path.basename(f)

    for f in final["files"]:
        c = code_of(f)
        if c is None or len(c.encode()) > MAX_CODE_BYTES or not f.endswith(CODE_EXT):
            continue
        dst = fs_dir / safe_rel(f)
        assert fs_dir in dst.resolve().parents, dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(redact(c))
        copied.append(safe_rel(f) + ("" if f in codes else "  (recovered from trace; not in snapshot)"))
    if final["inline_file"]:
        last = next(s for s in steps if s["n"] == final["step"])
        (fs_dir / final["inline_file"]).write_text(
            f"# Inline code the agent ran at step {final['step']} (the last command that wrote\n"
            f"# submission/aligned_coordinates). Recovered from the trace.\n\n" + redact(inline_code(last["command"])))
        copied.append(final["inline_file"])
    final["copied"] = copied
    write_solution_md(fs_dir / "SOLUTION.md", row, final)

    meta = {k: row.get(k) for k in ("run_id", "run_dir", "job", "agent", "model", "reasoning_effort", "condition",
                                     "prompt_variant", "task", "family", "dataset", "status", "failure_stage",
                                     "agent_returncode", "started_at", "finished_at", "elapsed_seconds",
                                     "leakage_status")}
    fm = man.get("failure_message")
    if fm:
        fm = str(fm)
        # agent-stage failures often carry the raw event stream; keep only a short clean tail
        fm = fm if not fm.lstrip().startswith("{") else "[agent event stream; see trace.md] ..." + fm[-300:]
        meta["failure_message"] = trim(clean_text(fm[:2000]), 15)
    meta.update({k: v for k, v in row.items() if k.startswith("score_")})
    meta["exploration"] = stats
    meta["final_solution"] = {k: final[k] for k in ("entry", "how", "step", "files", "copied", "inline_file", "note", "upstream")}
    meta["parse_ok"] = parse_ok
    meta["snapshot_code_files"] = sorted(snap_files)
    meta["scripts_written"] = scripts_written
    old = out_dir / "meta.json"
    if old.exists():  # keep tags from a previous tagging pass
        try:
            prev = json.loads(old.read_text())
            for k in ("tags", "tags_final", "evidence"):
                if k in prev:
                    meta[k] = prev[k]
        except Exception:
            pass
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    # steps.json for the tagger: full code inputs, trimmed outputs
    slim = []
    for s in steps:
        d = {k: s.get(k) for k in ("n", "kind", "ts", "name", "path", "ok", "sub", "writes_submission", "ran_scripts", "inline")}
        if s["kind"] == "tool":
            d["command"] = redact(s.get("command") or "")[:200_000]
            d["content"] = redact(s.get("content") or "")[:200_000] if s.get("content") else None
            d["output"] = trim(clean_text(s.get("output") or ""), 60)
        else:
            d["text"] = redact(s.get("text") or "")
        slim.append(d)
    (out_dir / "steps.json").write_text(json.dumps(slim))
    (out_dir / "trace.md").write_text(render_trace(row, man, stats, final, steps, ws_abs))
    return {"run_id": row["run_id"], "run_status": row["status"], "status": "ok", "parse_ok": parse_ok, "agent": row["agent"],
            "entry": final["entry"] or final["inline_file"]}


def write_solution_md(path: Path, row: dict, final: dict):
    L = [f"# Final solution: {row['run_dir']}", "", f"- run status: {row.get('status')}",
         f"- model: {row['model']}  condition: {row['condition']}  task: {row['task']}",
         f"- overall score: {row.get('score_overall_mean')}", ""]
    if final["entry"]:
        L += [f"**Entry point:** `{final['entry']}`", ""]
    elif final["inline_file"]:
        L += [f"**Entry point:** `{final['inline_file']}` (inline Python run from the shell; no script file)", ""]
    else:
        L += ["**Entry point:** not identified (no submission-writing command or script found).", ""]
    if final["step"]:
        cmd = trim(clean_text(final["command"] or ""), 12)
        L += [f"Last command that wrote `submission/aligned_coordinates` (step {final['step']}, via {final['how']}):", "",
              "```bash", cmd, "```", ""]
    if final.get("note"):
        L += [f"Note: {final['note']}", ""]
    if final.get("copied"):
        L += ["Files included:"] + [f"- `{f}`" + (" (entry)" if f.split()[0] == final["entry"] else
                                                   " (upstream stage: writes an intermediate the entry loads)" if f.split()[0] in final.get("upstream", []) else
                                                   " (local helper imported)" if f.split()[0] != final.get("inline_file") else "")
                                     for f in final["copied"]]
    L += ["", "Paths are relative to the agent workspace root. Data files (.h5ad/.npz/.pkl/.csv) are never included."]
    path.write_text("\n".join(L) + "\n")


def _fmt_ts(ts):
    if not ts:
        return ""
    return ts.replace("T", " ")[:19]


def render_trace(row, man, stats, final, steps, ws_abs=""):
    ws_re = re.compile(r"/tmp/p2b_workspaces/[^/\s'\"]+/[^/\s'\"]+")

    def ct(x):
        x = ws_re.sub("$WS", x or "")
        x = re.sub(r"(?:^|(?<=\n))cd \$WS\s*(?:;|&&)\s*", "", x)
        return clean_text(x)

    tss = [_ts(s.get("ts")) for s in steps if s.get("ts")]
    t_first = min(tss) if tss else None

    def num(x):
        return f"{x:,}" if isinstance(x, (int, float)) and x is not None else str(x)
    wall = stats["wall_s"]
    H = [f"# Trace: {row['run_dir']}", "", f"Run status: **{row.get('status')}**" + (f" (failure stage: {row.get('failure_stage')})" if row.get("failure_stage") else "") + f"; array task {row.get('job')}", "",
         "| field | value |", "|---|---|",
         f"| model | {row['model']} ({row['agent']}{', effort ' + str(man.get('reasoning_effort')) if man.get('reasoning_effort') else ''}) |",
         f"| condition | {row['condition']} ({row['prompt_variant']}) |",
         f"| task | {row['task']} (family {row['family']}) |",
         f"| score | overall_mean={row.get('score_overall_mean')}, mae={row.get('score_mae')}, mae_reduction={row.get('score_mae_reduction')}, matching_accuracy={row.get('score_matching_accuracy')} |",
         f"| wall time | {round(wall / 60, 1) if wall else '?'} min (started {row.get('started_at')}) |",
         f"| tokens | input {num(stats['input_tokens'])} (cached {num(stats['cached_input_tokens'])}), output {num(stats['output_tokens'])}, reasoning {num(stats['reasoning_tokens'])} |",
         f"| cost | ${stats['cost_usd']:.2f} |" if isinstance(stats["cost_usd"], (int, float)) else "| cost | ? |",
         f"| steps | {stats['n_steps']} ({stats['n_tool_calls']} tool calls, {stats['n_tool_errors']} errors) |",
         f"| submission writes | {stats['n_submission_writes']} (first at {round(stats['t_first_submission_s'] / 60, 1) if stats['t_first_submission_s'] is not None else '?'} min, {stats['t_first_submission_method']}) |",
         f"| final solution | `{final['entry'] or final['inline_file']}` (step {final['step']}) |",
         "",
         "Cleaning: ANSI codes and progress-bar redraws stripped, repeated lines collapsed, blobs over 200 chars dropped, "
         f"secrets redacted; tool inputs trimmed to {CMD_LINES} lines and results to {OUT_LINES} lines. Agent narration is in full. "
         "'gen tok' is an estimate (chars/4) of what the model generated at that step; totals above are from the CLI. "
         "The task prompt is omitted (see prompt.md in the run dir).", ""]
    if row["agent"] == "codex":
        H += ["Codex emits no per-event timestamps; step times are not available.", ""]
    B = []
    for s in steps:
        tsec = _ts(s.get("ts")) if s.get("ts") else None
        tag = f"[{_fmt_ts(s.get('ts'))} UTC, +{(tsec - t_first) / 60:.1f} min] " if tsec is not None and t_first else ""
        sub = " (subagent)" if s.get("sub") else ""
        # Per-step token counts are not reported reliably by the CLIs (Claude stream-json only
        # carries a partial streaming count), so show an estimate of the tokens the model
        # generated for this step: characters of its text or tool input / 4.
        gen = (s.get("text") or "") + (s.get("command") or "") + (s.get("content") or "")
        tok = f" · ≈{max(1, len(gen) // 4):,} gen tok" if gen else ""
        if s["kind"] in ("text", "thinking"):
            label = "Agent" if s["kind"] == "text" else "Agent (thinking)"
            B += [f"### {s['n']}. {label}{sub} {tag}{tok}".rstrip(), "", ws_re.sub("$WS", clean_text(s["text"], keep_long_tokens=True)).strip(), ""]
            continue
        name = s.get("name") or "tool"
        flag = "" if s.get("ok") is not False else " · ERROR"
        ws = f" · **writes submission ({s['writes_submission']})**" if s.get("writes_submission") else ""
        B += [f"### {s['n']}. Tool: {name}{sub} {tag}{tok}{flag}{ws}".rstrip(), ""]
        if s.get("content") is not None:
            lang = "python" if (s.get("path") or "").endswith(".py") else ""
            B += [f"Write `{s.get('path')}`:", "", f"```{lang}", trim(ct(s["content"]), CMD_LINES), "```", ""]
        elif s.get("command"):
            if s.get("path"):
                B += [f"Edit `{s.get('path')}`:", ""]
            B += ["```", trim(ct(s["command"]), CMD_LINES), "```", ""]
            if name == "apply_patch":
                B += ["(Codex does not log patch contents; the file's final version is in the workspace snapshot"
                      " and, if part of the solution, in final_solution/.)", ""]
        if s.get("output"):
            B += ["Result:", "", "```", trim(ct(s["output"]), OUT_LINES), "```", ""]
    return "\n".join(H + B) + "\n"


# --------------------------------------------------------------------------- main
def _safe(row, force):
    try:
        return process_run(row, force)
    except Exception as ex:  # one bad run must not stop the pass
        return {"run_id": row["run_id"], "run_status": row.get("status"), "status": "error", "agent": row.get("agent"),
                "error": f"{type(ex).__name__}: {ex}", "tb": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--runs", nargs="*", help="run_id or run_dir names to process (default: all with a manifest)")
    ap.add_argument("--scored-only", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    rows = [r for r in build_index() if r["status"] != "no_manifest"]
    if a.scored_only:
        rows = [r for r in rows if r["status"] == "scored"]
    if a.runs:
        rows = [r for r in rows if r["run_dir"] in set(a.runs) or r["run_id"] in set(a.runs)]
    if a.limit:
        rows = rows[: a.limit]
    res = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=min(a.workers, 8)) as ex:
            for i, r in enumerate(ex.map(_safe, rows, [a.force] * len(rows), chunksize=4), 1):
                res.append(r)
                if i % 100 == 0:
                    print(f"{i}/{len(rows)}", flush=True)
    else:
        for r in rows:
            res.append(_safe(r, a.force))
    log = TRACES_ROOT / "extract_log.json"
    prev = {}
    if log.exists():
        try:
            prev = {x["run_id"]: x for x in json.loads(log.read_text()) if "run_id" in x}
        except Exception:
            prev = {}
    for r in res:
        if r["status"] != "skipped" or r["run_id"] not in prev:
            prev[r["run_id"]] = r
    log.write_text(json.dumps(list(prev.values()), indent=0))
    from collections import Counter
    print(Counter((r.get("agent"), r.get("run_status"), r["status"], r.get("parse_ok")) for r in res))
    for r in res:
        if r["status"] == "error":
            print(r["run_id"], r["error"])


if __name__ == "__main__":
    main()
