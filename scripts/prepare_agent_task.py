from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from genomics_agent_bench.task_spec import load_task_spec

SANITIZED_H5AD_SCHEMA_VERSION = "3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a SWE-agent-style workspace for a spatial-alignment task.")
    parser.add_argument("task", type=Path, help="Path to task.toml")
    parser.add_argument("--workspace-root", type=Path, default=Path("agent_workspaces"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clean-submission", action="store_true", help="Remove prior coordinate CSVs and scores for a fresh run.")
    parser.add_argument(
        "--warm-start-from", type=Path, default=None,
        help="A directory of coordinate CSVs (a previous run's submission/aligned_coordinates) to place in the "
             "workspace as the current answer. The agent starts from a valid alignment and overwrites it only if "
             "it finds a better one. Nothing is said about how good the prior is.",
    )
    parser.add_argument(
        "--link-source-data",
        action="store_true",
        help="Use raw source H5AD symlinks instead of sanitized agent-visible copies.",
    )
    parser.add_argument(
        "--copy-agent-data",
        action="store_true",
        help="Copy agent-visible H5ADs into the workspace instead of symlinking them.",
    )
    parser.add_argument(
        "--blind-solve",
        action="store_true",
        help="Prepare a solve-only workspace with copied sanitized data and no local scorer/bootstrap helpers.",
    )
    parser.add_argument(
        "--no-method-helper",
        action="store_true",
        help="Omit package-backed method helper scripts and package prompt text for a true no-package control.",
    )
    return parser.parse_args()


def _link_or_copy(source: Path, dest: Path, *, copy: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(source.resolve(), dest)
    else:
        os.symlink(source.resolve(), dest)


def _copy_safe_attrs(source, dest, allowed: set[str] | None = None) -> None:
    private_tokens = ("ground", "truth", "gt", "ccf", "perturb", "transform", "rotation", "translation", "aligned")
    for key, value in source.attrs.items():
        key_text = str(key).lower()
        value_text = str(value).lower()[:500]
        if allowed is not None and key not in allowed:
            continue
        if any(token in key_text or token in value_text for token in private_tokens):
            continue
        dest.attrs[key] = value


def _set_h5ad_array_attrs(node) -> None:
    for key in ("encoding-type", "encoding_type", "encoding-version", "encoding_version"):
        node.attrs.pop(key, None)


def _set_h5ad_string_array_attrs(node) -> None:
    _set_h5ad_array_attrs(node)


def _set_h5ad_dataframe_attrs(group) -> None:
    group.attrs["_index"] = group.attrs.get("_index", "_index")
    group.attrs["column-order"] = np.array([], dtype=float)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.1.0"


def _set_h5ad_dict_attrs(group) -> None:
    for key in ("encoding-type", "encoding_type", "encoding-version", "encoding_version"):
        group.attrs.pop(key, None)


def _set_h5ad_matrix_attrs(node) -> None:
    if isinstance(node, h5py.Group):
        _set_h5ad_dict_attrs(node)
        for child in node.values():
            if isinstance(child, h5py.Dataset):
                _set_h5ad_array_attrs(child)
    else:
        _set_h5ad_array_attrs(node)


def _set_h5ad_group_child_array_attrs(group) -> None:
    _set_h5ad_dict_attrs(group)
    for child in group.values():
        _set_h5ad_matrix_attrs(child)


def _write_sanitized_h5ad(source: Path, dest: Path) -> None:
    """Write an agent-visible H5AD that omits scorer-only labels, GT coords, and transforms.

    The scorer still reads the original task H5ADs. Agent workspaces should only expose:
    observation IDs, expression matrices, gene metadata, and original spatial x/y coordinates.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent, delete=False) as handle:
        tmp = Path(handle.name)

    try:
        with h5py.File(source, "r") as src, h5py.File(tmp, "w") as out:
            _copy_safe_attrs(src, out)
            _set_h5ad_dict_attrs(out)
            out.attrs["genomics-agent-sanitized-version"] = SANITIZED_H5AD_SCHEMA_VERSION

            for key in ("X", "var", "layers"):
                if key in src:
                    src.copy(key, out, name=key)
            if "X" in out:
                _set_h5ad_matrix_attrs(out["X"])
            if "var" in out:
                _set_h5ad_dataframe_attrs(out["var"])
                if "_index" in out["var"]:
                    _set_h5ad_string_array_attrs(out["var/_index"])
            if "layers" in out:
                _set_h5ad_group_child_array_attrs(out["layers"])

            obs = out.create_group("obs")
            if "obs" in src:
                _copy_safe_attrs(src["obs"], obs, allowed={"_index", "encoding-type", "encoding-version"})
                if "_index" in src["obs"]:
                    src["obs"].copy("_index", obs, name="_index")
                obs.attrs["_index"] = src["obs"].attrs.get("_index", "_index")
            _set_h5ad_dataframe_attrs(obs)
            if "_index" in obs:
                _set_h5ad_string_array_attrs(obs["_index"])

            obsm = out.create_group("obsm")
            _set_h5ad_dict_attrs(obsm)
            if "obsm" in src:
                _copy_safe_attrs(src["obsm"], obsm)
                if "spatial" in src["obsm"]:
                    src["obsm"].copy("spatial", obsm, name="spatial")
                    _set_h5ad_array_attrs(obsm["spatial"])

        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _sanitized_cache_path(source: Path, cache_root: Path) -> Path:
    resolved = source.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{source.stem}-{digest}{source.suffix}"


def _sanitized_cache_is_current(cached: Path, source_stat: os.stat_result) -> bool:
    if not cached.exists() or cached.stat().st_mtime < source_stat.st_mtime:
        return False
    try:
        with h5py.File(cached, "r") as handle:
            return str(handle.attrs.get("genomics-agent-sanitized-version", "")) == SANITIZED_H5AD_SCHEMA_VERSION
    except OSError:
        return False


def _ensure_agent_visible_data(source: Path, cache_root: Path, *, use_source_link: bool) -> Path:
    if use_source_link:
        return source.resolve()
    cached = _sanitized_cache_path(source, cache_root)
    source_stat = source.stat()
    if not _sanitized_cache_is_current(cached, source_stat):
        _write_sanitized_h5ad(source, cached)
    return cached.resolve()


PRIVATE_INSTRUCTION_PATTERNS = (
    r"\b(label[_ -]?key|landmark|region[- ]?overlap|matching_accuracy|overlap_ratio_region)\b",
    r"\b(anatomical label|annotation|cell[_ -]?class|cell[_ -]?type|cluster annotation|cluster labels?|Region annotation)\b",
    r"\b(label-aware|annotation-bearing|curated cluster annotations?|compatible cluster labels?|region labels?|scoring labels?)\b",
    r"\b(metric profile|gene-only metric|landmark/region metrics?|region metrics?|deterministic scorer)\b",
    r"\b(obs/Region|obs/Cluster|obs/Cell_class|obs/class_label|obs/subclass|obs/label|obs/annotation)\b",
    r"\b(x_GT|y_GT|z_GT|x_ccf|y_ccf|z_ccf|X_CCF|spatial_3d_aligned)\b",
    r"\b(ground[- ]?truth|held[- ]?out atlas|MAE|perturbation|rotation|translation)\b",
    # "transform" alone describes a method (sa-005: infer a transform, then apply it); only
    # phrases that point at the answer are private
    r"\b((ground[- ]?truth|true|hidden|rigid|per[- ]slice|applied)\s+transforms?|transforms?\s+(metadata|parameters|matri(x|ces)))\b",
    r"\b(sabench_mae|compiled slices|source:|source tar|source-manifest|split_manifest)\b",
    r"\b(benchmarks/spatial_alignment/compiled_data|hidden_perturbation_grid\.json)\b",
    r"\b(benchmarks/spatial_alignment/tasks/|task\.toml)\b",
    r"(~/Downloads|/Users/[^` )]+)",
)


PRIVATE_INLINE_REPLACEMENTS = (
    (r"`?scripts/[A-Za-z0-9_./-]+\.py`?", "the preprocessing pipeline"),
    (r"`?benchmarks/spatial_alignment/compiled_data/[^\s`)]*`?", "withheld compiled source path"),
    (r"`?benchmarks/spatial_alignment/tasks/[^\s`)]*`?", "withheld task source path"),
    (r"`?~/Downloads/[^\s`)]*`?", "withheld local source cache"),
    (r"`?/Users/[^` )]+`?", "withheld local path"),
    (r"`?hidden_perturbation_grid\.json`?", "withheld perturbation metadata"),
)


def _redact_private_instruction(instruction: str) -> str:
    redacted: list[str] = []
    omitted = 0
    for line in instruction.splitlines():
        for pattern, replacement in PRIVATE_INLINE_REPLACEMENTS:
            line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in PRIVATE_INSTRUCTION_PATTERNS):
            omitted += 1
            continue
        redacted.append(line)
    text = "\n".join(redacted).strip()
    if omitted:
        text = (
            f"{text}\n\n"
            "Note: source-manifest details that referenced scorer-only annotations, ground-truth coordinates, "
            "or transform metadata were omitted from this agent prompt."
        ).strip()
    return text


def _task_prompt(
    task_path: Path,
    instruction: str,
    task_id: str,
    data_specs: list[str],
    output_specs: list[str],
    *,
    workspace: Path,
    has_split_manifest: bool,
    public_score_split: str,
    withheld_score_split: str,
    blind_solve: bool,
    expose_method_helper: bool = True,
    warm_start_files: list[str] | None = None,
) -> str:
    warm_start_block = ""
    if warm_start_files:
        listed = "\n".join(f"- `submission/aligned_coordinates/{name}`" for name in warm_start_files)
        warm_start_block = (
            "## A Prior Alignment Is Already On Disk\n\n"
            "The required coordinate CSVs already hold an alignment produced by an earlier attempt on this same "
            "task, working from the same inputs and without the method packages available here:\n\n"
            f"{listed}\n\n"
            "Treat it as the current answer, not as a constraint. Read it, judge it with the allowed inputs only, "
            "and overwrite a file when you have something better. If nothing you try beats it, leave it in place: "
            "an unchanged submission is a valid outcome. Nothing is claimed about how good it is.\n\n"
        )

    slice_rows = []
    for spec in data_specs:
        slice_id, path = spec.split("=", 1)
        source_name = Path(path).name
        linked_name = f"{slice_id}{Path(path).suffix}"
        if blind_solve:
            slice_rows.append(f"- `{slice_id}`: `data/{linked_name}`")
        else:
            slice_rows.append(f"- `{slice_id}`: `data/{linked_name}` (source file `{source_name}`)")

    output_rows = []
    for spec in output_specs:
        slice_id, path = spec.split("=", 1)
        output_rows.append(f"- `submission/aligned_coordinates/{slice_id}.csv`")

    split_section = ""
    if has_split_manifest:
        if blind_solve:
            split_section = """
## Scoring Split

Final scoring is run only by the benchmark harness after your agent process exits. Submit coordinates for all observations.
"""
        else:
            split_section = f"""
## Train/Eval Split

A deterministic 50/50 observation split exists for scoring.
During solving, `./score.sh` reports only the public `{public_score_split}` split for debugging.
The `{withheld_score_split}` split is withheld and is scored only by the benchmark runner after your agent process exits.
Submit coordinates for all observations.
"""

    privacy_section = """
## Data Visibility

The H5AD files under `data/` are sanitized agent-visible inputs.

Allowed inputs:

- Observation IDs from `obs/_index`
- Gene expression from `X`, `var`, and any provided expression `layers`
- Original two-dimensional coordinates from `obsm/spatial`
- Slice identities and required output filenames from this task prompt

Not allowed and not present in the solve workspace:

- Hidden gold aligned coordinates or 3D atlas coordinates
- Ground-truth transforms, perturbation metadata, or perturbation configuration files
- Evaluation landmarks, scorer-only split manifests, or local scorer source
- Anatomical or scoring labels reserved for evaluation
- Private benchmark source paths, local download caches, previous run artifacts, or benchmark repository files

Use only the allowed inputs above. Do not infer, search for, or rely on hidden source paths, scorer internals, prior submissions, or withheld metadata.
"""
    if expose_method_helper:
        execution_section = """
## Execution Strategy

First create a simple valid submission for every required output CSV, even if it only uses a conservative baseline transform. Then improve methodically: try richer models, package-backed methods, or refinements only after a complete returnable answer is already on disk. Whenever you find a better alignment, overwrite the same prediction CSVs. Keep the fallback CSVs valid while you explore stronger approaches.
"""
    else:
        execution_section = """
## Execution Strategy

First create a simple valid submission for every required output CSV, even if it only uses a conservative baseline transform. Then improve methodically: try richer custom models or refinements only after a complete returnable answer is already on disk. Whenever you find a better alignment, overwrite the same prediction CSVs. Keep the fallback CSVs valid while you explore stronger approaches.
"""
    if warm_start_block:
        execution_section = "\n" + warm_start_block + execution_section.lstrip("\n")
    if blind_solve:
        method_note = ""
        if expose_method_helper:
            method_note = """
If `.method-venv/` exists, `./agent_python.sh` automatically uses `.method-venv/bin/python` for the shared baseline stack.

Package-backed spatial alignment methods are allowed. After you have a complete baseline submission on disk, inspect the available package environments and choose the package-backed method or methods that look most promising for this task. There is no quota of packages or method families to try, and PASTE/PASTE2 are not required defaults. A useful package attempt means importing the method package from its listed `.method-venv*/bin/python`, running its alignment or transform API on the task data or a representative adjacent-slice subset, and recording enough output to tell whether it succeeded. If a package is likely infeasible for the task size, fails to import, stalls, or looks worse under allowed input-only diagnostics, move on or fall back to a custom method. Keep the strongest submission.

Method environment rule: if you want to use a package, use the Python executable listed for that package in `.method-cache/METHOD_ENVS.md` or `.method-cache/compat_h5ad/method_envs.tsv`. Do not import paper-method packages through `./agent_python.sh` or base `.method-venv/bin/python`; those are for general data handling only. The environment layout is declared in `spatial_method_envs.toml` and copied to `.method-cache/spatial_method_envs.toml` after bootstrap. The current minimal grouping is: `paste`/`paste2` -> `.method-venv-ot`, `slat`/`scslat` -> `.method-venv-scslat`, and all other method targets use their own listed env because their dependency stacks are not reliably compatible.

Internet access is available for public package documentation and API lookup. You may use web search, `curl`, GitHub READMEs, ReadTheDocs pages, PyPI pages, and `python -m pip show` to learn how to call installed method packages. This is allowed only for package/method documentation; do not search for benchmark source, scorer code, prior runs/results, hidden labels, paper result tables, or anything outside the allowed task inputs. If you use online docs, record the URL and the package/API call you tried in a short local note or `submission/report.json`.

Documentation menu, shown in non-ranked order:

| Helper target | Public docs / examples |
|---|---|
| `staligner` | https://staligner.readthedocs.io/ and https://github.com/zhoux85/STAligner |
| `spateo` | https://spateo-release.readthedocs.io/en/stable/tutorials/notebooks/3_alignment/Basic%20usage%20of%20Spateo%20alignment%20for%202D%20slices.html and https://spateo-release.readthedocs.io/ |
| `gpsa` | https://github.com/andrewcharlesjones/spatial-alignment and https://www.nature.com/articles/s41592-023-01972-2 |
| `cast` | https://cast-tutorial.readthedocs.io/ and https://github.com/wanglab-broad/CAST |
| `stair` | https://stair-tutorial.readthedocs.io/en/latest/ and https://github.com/yuyuanyuana/STAIR |
| `santo` | https://github.com/leihouyeung/SANTO and https://www.nature.com/articles/s41467-024-50308-x |
| `spacel` | https://github.com/QuKunLab/SPACEL |
| `scslat` / `slat` | https://github.com/gao-lab/SLAT |
| `stalign` | https://jef.works/STalign and https://github.com/JEFworks-Lab/STalign |
| `paste2` | https://github.com/raphael-group/paste2 and https://raphael-group.github.io/paste3/notebooks/paste2_tutorial.html |
| `paste` | https://github.com/raphael-group/paste and https://pypi.org/project/paste-bio/ |

API and example entry points:

Use the examples below as starting points for package-backed alignment attempts on the task data. Signature/introspection snippets are included only to handle package-version differences; do not stop at API discovery if a method is usable. The goal is to produce aligned coordinate CSVs from the strongest package-backed or custom approach.

```bash
# Spateo: rigid/nonrigid pairwise slice alignment. Output coordinates are added under key_added.
./.method-venv-spateo/bin/python - <<'PY'
import inspect, spateo as st
print(inspect.signature(st.align.morpho_align))
print(st.align.morpho_align)
PY
```

```bash
# STalign: diffeomorphic image/point alignment from rasterized coordinates/expression.
./.method-venv-stalign/bin/python - <<'PY'
import inspect
import STalign.STalign as stalign
for name in ["L_T_from_points", "LDDMM", "build_transform"]:
    obj = getattr(stalign, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
```

```bash
# CAST: representation learning across samples from coordinate/expression dictionaries.
./.method-venv-cast/bin/python - <<'PY'
import inspect, CAST
print(inspect.signature(CAST.CAST_MARK))
PY
```

```bash
# SPACEL/Scube: pairwise multi-slice alignment; may require spatial-domain labels in .obs.
./.method-venv-spacel/bin/python - <<'PY'
import inspect
from SPACEL import Scube
print(inspect.signature(Scube.alignment.align))
PY
```

```bash
# PASTE2: partial-overlap variant in the merged OT env.
./.method-venv-ot/bin/python - <<'PY'
import inspect
from paste2 import PASTE2
print([n for n in dir(PASTE2) if "align" in n.lower() or "pair" in n.lower()])
print(inspect.signature(PASTE2.partial_pairwise_align))
PY
```

```bash
# STAligner / scSLAT / GPSA: graph/latent alignment families; adapt usable APIs to coordinates.
./.method-venv-staligner/bin/python - <<'PY'
import inspect, STAligner
for name in ["Cal_Spatial_Net", "train_STAligner"]:
    obj = getattr(STAligner, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
./.method-venv-scslat/bin/python - <<'PY'
import pkgutil
print([m.name for m in pkgutil.iter_modules() if "slat" in m.name.lower()][:20])
PY
./.method-venv-gpsa/bin/python - <<'PY'
import inspect, gpsa
print(getattr(gpsa, "VariationalGPSA", None))
print(inspect.signature(gpsa.VariationalGPSA))
PY
```

Package API implementation examples:

```bash
# Common setup after `./bootstrap_method_venv.sh paper-all`: package APIs should
# generally load the sanitized AnnData-compatible files in this cache.
./agent_python.sh - <<'PY'
from pathlib import Path
import anndata as ad
files = sorted(Path(".method-cache/compat_h5ad").glob("*.h5ad"))
print("compat files:", [p.name for p in files])
for path in files[:3]:
    a = ad.read_h5ad(path)
    print(path.name, a.shape, list(a.obsm.keys()), a.obsm["spatial"].shape)
PY
```

```bash
# PASTE / PASTE2: start from pairwise and serial stacking APIs, then adapt them
# to adjacent slices and write transformed coordinates. Both live in the merged OT env.
./.method-venv-ot/bin/python - <<'PY'
import inspect, paste
print("pairwise_align", inspect.signature(paste.pairwise_align))
print("stack_slices_pairwise", inspect.signature(paste.stack_slices_pairwise))
PY
./.method-venv-ot/bin/python - <<'PY'
import inspect
from paste2 import PASTE2
print("PASTE2 functions:", [n for n in dir(PASTE2) if "align" in n.lower() or "pair" in n.lower()])
print("partial_pairwise_align", inspect.signature(PASTE2.partial_pairwise_align))
PY
```

```bash
# Spateo / STalign / CAST / SPACEL: find nearby alignment or registration
# names, then adapt one to a representative adjacent-slice pair and scale to all outputs.
./.method-venv-spateo/bin/python - <<'PY'
import inspect, spateo as st
print([n for n in dir(st.align) if "align" in n.lower() or "morpho" in n.lower()][:30])
print(inspect.signature(st.align.morpho_align))
print("pattern: st.align.morpho_align(model_ref=adata_ref, model_tgt=adata_tgt, spatial_key='spatial', key_added='spateo_align')")
PY
./.method-venv-stalign/bin/python - <<'PY'
import inspect
import STalign.STalign as stalign
for name in ["rasterize", "L_T_from_points", "LDDMM", "build_transform", "transform_points_source_to_target"]:
    obj = getattr(stalign, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
./.method-venv-cast/bin/python - <<'PY'
import inspect, CAST
print(inspect.signature(CAST.CAST_MARK))
print([n for n in dir(CAST) if any(k in n.lower() for k in ["mark", "align", "reg", "project"])][:40])
PY
./.method-venv-spacel/bin/python - <<'PY'
import inspect
from SPACEL import Scube
print(inspect.signature(Scube.alignment.align))
print([n for n in dir(Scube) if "align" in n.lower() or "spatial" in n.lower()][:40])
PY
```

```bash
# STAligner / scSLAT / GPSA / STAIR / SANTO: entry-point names vary by release,
# so find callable alignment-like APIs and adapt a working one to the task.
./.method-venv-staligner/bin/python - <<'PY'
import inspect, STAligner
for name in ["Cal_Spatial_Net", "Stats_Spatial_Net", "train_STAligner", "mclust_R"]:
    obj = getattr(STAligner, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
./.method-venv-scslat/bin/python - <<'PY'
import importlib, inspect, pkgutil
print([m.name for m in pkgutil.iter_modules() if "slat" in m.name.lower()][:30])
for mod_name in ["scSLAT", "SLAT", "slat"]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        print(mod_name, "import failed", type(exc).__name__, exc)
        continue
    names = [n for n in dir(mod) if any(k in n.lower() for k in ["match", "align", "train", "graph"])]
    print(mod_name, names[:40])
    for name in names[:8]:
        obj = getattr(mod, name)
        if callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                pass
PY
./.method-venv-gpsa/bin/python - <<'PY'
import inspect, gpsa
print([n for n in dir(gpsa) if "GPSA" in n or "Variational" in n][:30])
print(inspect.signature(gpsa.VariationalGPSA))
PY
for env in stair santo; do
  ./.method-venv-${env}/bin/python - <<'PY'
import importlib, inspect, pkgutil
for mod_name in ["STAIR", "stair", "SANTO", "santo"]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        continue
    print("module", mod_name)
    names = [n for n in dir(mod) if any(k in n.lower() for k in ["align", "match", "register", "train", "stair", "santo"])]
    print(names[:50])
    for name in names[:10]:
        obj = getattr(mod, name)
        if callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                pass
for m in pkgutil.iter_modules():
    if m.name.lower() in {"stair", "santo"} or "stair" in m.name.lower() or "santo" in m.name.lower():
        print("installed module candidate:", m.name)
PY
done
```

To try packages, use the workspace-local method helper:

```bash
./bootstrap_method_venv.sh paper-all
./bootstrap_method_venv.sh probe
cat .method-cache/METHOD_ENVS.md
cat .method-cache/spatial_method_envs.toml
# or a single target by name. This shuffled order is not a ranking:
./bootstrap_method_venv.sh staligner
./bootstrap_method_venv.sh spateo
./bootstrap_method_venv.sh gpsa
./bootstrap_method_venv.sh cast
./bootstrap_method_venv.sh stair
./bootstrap_method_venv.sh santo
./bootstrap_method_venv.sh spacel
./bootstrap_method_venv.sh scslat
./bootstrap_method_venv.sh stalign
./bootstrap_method_venv.sh slat
./bootstrap_method_venv.sh ot
```

This helper is self-contained in the solve workspace. It may download packages if they are not already cached, and install/import failures should be treated as ordinary method failures.
The helper reads `spatial_method_envs.toml` and creates grouped or isolated method environments such as `.method-venv-ot/`, `.method-venv-staligner/`, `.method-venv-spateo/`, `.method-venv-gpsa/`, `.method-venv-cast/`, `.method-venv-stair/`, `.method-venv-santo/`, `.method-venv-spacel/`, `.method-venv-scslat/`, and `.method-venv-stalign/`, plus compatible AnnData inputs under `.method-cache/compat_h5ad/`. PASTE and PASTE2 share `.method-venv-ot/`; compatibility aliases `.method-venv-paste/` and `.method-venv-paste2/` point there when possible. Prefer the listed method Python for package-backed attempts because several paper-method packages have incompatible dependency stacks; do not treat the shared `.method-venv/` alone as a package-method attempt. `paper-all` best-effort preinstalls every supported method target; check `./bootstrap_method_venv.sh probe`, `.method-cache/METHOD_ENVS.md`, `.method-cache/spatial_method_envs.toml`, and `.method-cache/compat_h5ad/method_install_failures.tsv` for actual availability. Treat the helper target order as a shuffled menu, not a ranking.


```bash
./.method-venv-staligner/bin/python your_method_script.py
./.method-venv-spateo/bin/python your_method_script.py
./.method-venv-gpsa/bin/python your_method_script.py
./.method-venv-stalign/bin/python your_method_script.py
./.method-venv-ot/bin/python your_paste_or_paste2_script.py
cat .method-cache/METHOD_ENVS.md
```
"""
        local_scoring_section = """
## Local Scoring

No local scorer is available in this solve workspace. Final scoring is run by the benchmark harness after your agent process exits.
Do not search parent directories, repository paths, previous runs, benchmark source files, scorer code, Downloads caches, or other locations outside this workspace.

For Python, use the workspace helper:

```bash
./agent_python.sh your_script.py
# or
./agent_python.sh -c "import h5py; print('ok')"
```

The helper runs `uv` with common cached analysis packages: `numpy`, `scipy`, `pandas`, `scikit-learn`, and `h5py`.
{method_note}
""".format(method_note=method_note)
        package_bootstrap_section = ""
    else:
        local_scoring_section = """
## Local Scoring

From this workspace, run:

```bash
./score.sh
```

For quick debugging, inspect `.h5ad` contents with `h5py`.
"""
        package_bootstrap_section = ""
        if expose_method_helper:
            package_bootstrap_section = """
## Package-Backed Method Bootstrap

If you want to try package-backed spatial alignment libraries, use the shared helper in this workspace. Internet access is available for public package documentation and API lookup, including GitHub READMEs, ReadTheDocs pages, PyPI pages, and `python -m pip show`. This is allowed only for package/method documentation; do not search for benchmark source, scorer code, prior runs/results, hidden labels, paper result tables, or anything outside the allowed task inputs.

Method environment rule: if you want to use a package, use the Python executable listed for that package in `.method-cache/METHOD_ENVS.md` or `.method-cache/compat_h5ad/method_envs.tsv`. Do not import paper-method packages through `./agent_python.sh` or base `.method-venv/bin/python`; those are for general data handling only. The environment layout is declared in `spatial_method_envs.toml` and copied to `.method-cache/spatial_method_envs.toml` after bootstrap. The current minimal grouping is: `paste`/`paste2` -> `.method-venv-ot`, `slat`/`scslat` -> `.method-venv-scslat`, and all other method targets use their own listed env because their dependency stacks are not reliably compatible.

Documentation menu, shown in non-ranked order:

| Helper target | Public docs / examples |
|---|---|
| `staligner` | https://staligner.readthedocs.io/ and https://github.com/zhoux85/STAligner |
| `spateo` | https://spateo-release.readthedocs.io/en/stable/tutorials/notebooks/3_alignment/Basic%20usage%20of%20Spateo%20alignment%20for%202D%20slices.html and https://spateo-release.readthedocs.io/ |
| `gpsa` | https://github.com/andrewcharlesjones/spatial-alignment and https://www.nature.com/articles/s41592-023-01972-2 |
| `cast` | https://cast-tutorial.readthedocs.io/ and https://github.com/wanglab-broad/CAST |
| `stair` | https://stair-tutorial.readthedocs.io/en/latest/ and https://github.com/yuyuanyuana/STAIR |
| `santo` | https://github.com/leihouyeung/SANTO and https://www.nature.com/articles/s41467-024-50308-x |
| `spacel` | https://github.com/QuKunLab/SPACEL |
| `scslat` / `slat` | https://github.com/gao-lab/SLAT |
| `stalign` | https://jef.works/STalign and https://github.com/JEFworks-Lab/STalign |
| `paste2` | https://github.com/raphael-group/paste2 and https://raphael-group.github.io/paste3/notebooks/paste2_tutorial.html |
| `paste` | https://github.com/raphael-group/paste and https://pypi.org/project/paste-bio/ |

API and example entry points:

Use the examples below as starting points for package-backed alignment attempts on the task data. Signature/introspection snippets are included only to handle package-version differences; do not stop at API discovery if a method is usable. The goal is to produce aligned coordinate CSVs from the strongest package-backed or custom approach.

```bash
# Spateo: rigid/nonrigid pairwise slice alignment. Output coordinates are added under key_added.
./.method-venv-spateo/bin/python - <<'PY'
import inspect, spateo as st
print(inspect.signature(st.align.morpho_align))
print(st.align.morpho_align)
PY
```

```bash
# STalign: diffeomorphic image/point alignment from rasterized coordinates/expression.
./.method-venv-stalign/bin/python - <<'PY'
import inspect
import STalign.STalign as stalign
for name in ["L_T_from_points", "LDDMM", "build_transform"]:
    obj = getattr(stalign, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
```

```bash
# CAST: representation learning across samples from coordinate/expression dictionaries.
./.method-venv-cast/bin/python - <<'PY'
import inspect, CAST
print(inspect.signature(CAST.CAST_MARK))
PY
```

```bash
# SPACEL/Scube: pairwise multi-slice alignment; may require spatial-domain labels in .obs.
./.method-venv-spacel/bin/python - <<'PY'
import inspect
from SPACEL import Scube
print(inspect.signature(Scube.alignment.align))
PY
```

```bash
# PASTE2: partial-overlap variant in the merged OT env.
./.method-venv-ot/bin/python - <<'PY'
import inspect
from paste2 import PASTE2
print([n for n in dir(PASTE2) if "align" in n.lower() or "pair" in n.lower()])
print(inspect.signature(PASTE2.partial_pairwise_align))
PY
```

```bash
# STAligner / scSLAT / GPSA: graph/latent alignment families; adapt usable APIs to coordinates.
./.method-venv-staligner/bin/python - <<'PY'
import inspect, STAligner
for name in ["Cal_Spatial_Net", "train_STAligner"]:
    obj = getattr(STAligner, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
./.method-venv-scslat/bin/python - <<'PY'
import pkgutil
print([m.name for m in pkgutil.iter_modules() if "slat" in m.name.lower()][:20])
PY
./.method-venv-gpsa/bin/python - <<'PY'
import inspect, gpsa
print(getattr(gpsa, "VariationalGPSA", None))
print(inspect.signature(gpsa.VariationalGPSA))
PY
```

Package API implementation examples:

```bash
# Common setup after `./bootstrap_method_venv.sh paper-all`: package APIs should
# generally load the sanitized AnnData-compatible files in this cache.
./agent_python.sh - <<'PY'
from pathlib import Path
import anndata as ad
files = sorted(Path(".method-cache/compat_h5ad").glob("*.h5ad"))
print("compat files:", [p.name for p in files])
for path in files[:3]:
    a = ad.read_h5ad(path)
    print(path.name, a.shape, list(a.obsm.keys()), a.obsm["spatial"].shape)
PY
```

```bash
# PASTE / PASTE2: start from pairwise and serial stacking APIs, then adapt them
# to adjacent slices and write transformed coordinates. Both live in the merged OT env.
./.method-venv-ot/bin/python - <<'PY'
import inspect, paste
print("pairwise_align", inspect.signature(paste.pairwise_align))
print("stack_slices_pairwise", inspect.signature(paste.stack_slices_pairwise))
PY
./.method-venv-ot/bin/python - <<'PY'
import inspect
from paste2 import PASTE2
print("PASTE2 functions:", [n for n in dir(PASTE2) if "align" in n.lower() or "pair" in n.lower()])
print("partial_pairwise_align", inspect.signature(PASTE2.partial_pairwise_align))
PY
```

```bash
# Spateo / STalign / CAST / SPACEL: find nearby alignment or registration
# names, then adapt one to a representative adjacent-slice pair and scale to all outputs.
./.method-venv-spateo/bin/python - <<'PY'
import inspect, spateo as st
print([n for n in dir(st.align) if "align" in n.lower() or "morpho" in n.lower()][:30])
print(inspect.signature(st.align.morpho_align))
print("pattern: st.align.morpho_align(model_ref=adata_ref, model_tgt=adata_tgt, spatial_key='spatial', key_added='spateo_align')")
PY
./.method-venv-stalign/bin/python - <<'PY'
import inspect
import STalign.STalign as stalign
for name in ["rasterize", "L_T_from_points", "LDDMM", "build_transform", "transform_points_source_to_target"]:
    obj = getattr(stalign, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
./.method-venv-cast/bin/python - <<'PY'
import inspect, CAST
print(inspect.signature(CAST.CAST_MARK))
print([n for n in dir(CAST) if any(k in n.lower() for k in ["mark", "align", "reg", "project"])][:40])
PY
./.method-venv-spacel/bin/python - <<'PY'
import inspect
from SPACEL import Scube
print(inspect.signature(Scube.alignment.align))
print([n for n in dir(Scube) if "align" in n.lower() or "spatial" in n.lower()][:40])
PY
```

```bash
# STAligner / scSLAT / GPSA / STAIR / SANTO: entry-point names vary by release,
# so find callable alignment-like APIs and adapt a working one to the task.
./.method-venv-staligner/bin/python - <<'PY'
import inspect, STAligner
for name in ["Cal_Spatial_Net", "Stats_Spatial_Net", "train_STAligner", "mclust_R"]:
    obj = getattr(STAligner, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
PY
./.method-venv-scslat/bin/python - <<'PY'
import importlib, inspect, pkgutil
print([m.name for m in pkgutil.iter_modules() if "slat" in m.name.lower()][:30])
for mod_name in ["scSLAT", "SLAT", "slat"]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        print(mod_name, "import failed", type(exc).__name__, exc)
        continue
    names = [n for n in dir(mod) if any(k in n.lower() for k in ["match", "align", "train", "graph"])]
    print(mod_name, names[:40])
    for name in names[:8]:
        obj = getattr(mod, name)
        if callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                pass
PY
./.method-venv-gpsa/bin/python - <<'PY'
import inspect, gpsa
print([n for n in dir(gpsa) if "GPSA" in n or "Variational" in n][:30])
print(inspect.signature(gpsa.VariationalGPSA))
PY
for env in stair santo; do
  ./.method-venv-${env}/bin/python - <<'PY'
import importlib, inspect, pkgutil
for mod_name in ["STAIR", "stair", "SANTO", "santo"]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        continue
    print("module", mod_name)
    names = [n for n in dir(mod) if any(k in n.lower() for k in ["align", "match", "register", "train", "stair", "santo"])]
    print(names[:50])
    for name in names[:10]:
        obj = getattr(mod, name)
        if callable(obj):
            try:
                print(name, inspect.signature(obj))
            except Exception:
                pass
for m in pkgutil.iter_modules():
    if m.name.lower() in {"stair", "santo"} or "stair" in m.name.lower() or "santo" in m.name.lower():
        print("installed module candidate:", m.name)
PY
done
```

```bash
./bootstrap_method_venv.sh paper-all
./bootstrap_method_venv.sh probe
cat .method-cache/METHOD_ENVS.md
cat .method-cache/spatial_method_envs.toml
# or a single target by name. This shuffled order is not a ranking:
./bootstrap_method_venv.sh staligner
./bootstrap_method_venv.sh spateo
./bootstrap_method_venv.sh gpsa
./bootstrap_method_venv.sh cast
./bootstrap_method_venv.sh stair
./bootstrap_method_venv.sh santo
./bootstrap_method_venv.sh spacel
./bootstrap_method_venv.sh scslat
./bootstrap_method_venv.sh stalign
./bootstrap_method_venv.sh slat
./bootstrap_method_venv.sh ot
```

This reads `spatial_method_envs.toml`, creates `.method-venv/`, grouped or isolated `.method-venv*` method environments, and standard AnnData-compatible files under `.method-cache/compat_h5ad/`.
Use the listed method Python for package-backed method runs, for example `.method-venv-ot/bin/python`, `.method-venv-staligner/bin/python`, `.method-venv-spateo/bin/python`, `.method-venv-gpsa/bin/python`, or `.method-venv-stalign/bin/python`.
Use `./bootstrap_method_venv.sh paper-all` to best-effort install every supported package target, then inspect `./bootstrap_method_venv.sh probe`, `.method-cache/METHOD_ENVS.md`, `.method-cache/spatial_method_envs.toml`, and `.method-cache/compat_h5ad/method_install_failures.tsv`.
Probe multiple plausible methods first and keep whichever one actually scores best on train.
If uv virtualenv creation fails and `conda`, `mamba`, or `micromamba` is available, rerun the helper with conda fallback enabled:

```bash
./bootstrap_method_venv.sh --conda-fallback auto --recreate ot
```
"""

    return f"""# Agent Task: {task_id}

{instruction.strip()}

{privacy_section}

## Data

The input `.h5ad` files are available in this workspace:

{chr(10).join(slice_rows)}
{split_section}

## Required Output

Create coordinate CSV files in:

{chr(10).join(output_rows)}

Each CSV must contain exactly these columns:

```text
obs_id,x,y
```

Rules:

- `obs_id` values and row order must match the source `.h5ad` observations.
- `x` and `y` must be finite numeric aligned coordinates.
- Do not modify the source data files.
- You may create code, logs, figures, or a short `submission/report.json`, but scoring uses the coordinate CSV files.

{execution_section}

{local_scoring_section}
{package_bootstrap_section}
"""


def _score_script(repo_root: Path, task_path: Path, workspace: Path, public_score_split: str) -> str:
    task_id = load_task_spec(task_path).id
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {repo_root}
uv run python -m genomics_agent_bench.public_score {task_id} {workspace / "submission"} --output {workspace / "scores.json"} --score-split {public_score_split}
"""


def _bootstrap_script(repo_root: Path, workspace: Path) -> str:
    repo_root = repo_root.resolve()
    workspace = (Path.cwd() / workspace).resolve()
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {workspace}
bash {repo_root / "scripts" / "bootstrap_spatial_method_venv.sh"} --workspace {workspace} "$@"
"""


def _blind_bootstrap_script() -> str:
    # Keep blind workspaces self-contained by copying the repo bootstrap script body.
    # The bootstrap script itself must not rely on repository paths.
    return (Path(__file__).resolve().parent / "bootstrap_spatial_method_venv.sh").read_text(encoding="utf-8")

    return """#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./bootstrap_method_venv.sh [options] [method ...]

Create a workspace-local .method-venv for optional package-backed spatial
alignment methods. Supported method names: cast, gpsa, paste, paste2, santo,
scslat, slat, spacel, spateo, stair, stalign, staligner, probe, paper-common,
paper-all,
all, all-paper.

Options:
  --venv-name NAME   Venv directory name; defaults to .method-venv
  --compat-dir PATH  Compatible h5ad directory; defaults to .method-cache/compat_h5ad
  --recreate         Remove and recreate the selected venv before installing
EOF
}

recreate=0
venv_name=".method-venv"
compat_dir=".method-cache/compat_h5ad"
declare -a methods=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-name)
      venv_name="$2"
      shift 2
      ;;
    --compat-dir)
      compat_dir="$2"
      shift 2
      ;;
    --recreate)
      recreate=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      methods+=("$1")
      shift
      ;;
  esac
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"

venv_path="${venv_name}"
compat_path="${compat_dir}"
mkdir -p "${compat_path}"

if [[ ${recreate} -eq 1 && -d "${venv_path}" ]]; then
  rm -rf "${venv_path}"
fi

if [[ ! -x "${venv_path}/bin/python" ]]; then
  uv venv "${venv_path}" --python 3.11 --seed
fi

uv pip install --python "${venv_path}/bin/python" -U pip setuptools wheel
uv pip install --python "${venv_path}/bin/python" \
  "numpy<2" \
  "pandas<2" \
  "scipy<1.12" \
  "h5py<3.15" \
  "anndata==0.7.8" \
  scikit-learn \
  scikit-image

METHOD_COMPAT_PATH="${compat_path}" "${venv_path}/bin/python" - <<'PY'
from pathlib import Path
import os

import h5py
import numpy as np

STRING_DTYPE = h5py.string_dtype("utf-8")


def decode_array(values):
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray([
            value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
            for value in values
        ], dtype=object)
    return values


def string_array(values):
    return np.asarray(decode_array(values).astype(str), dtype=STRING_DTYPE)


def set_array_attrs(node):
    for key in ("encoding-type", "encoding_type", "encoding-version", "encoding_version"):
        node.attrs.pop(key, None)


def set_string_array_attrs(node):
    set_array_attrs(node)


def set_dataframe_attrs(group):
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray([], dtype=float)
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.1.0"


def copy_matrix(src, out, key="X"):
    node = src[key]
    if isinstance(node, h5py.Group):
        group = out.create_group(key)
        for child in ("data", "indices", "indptr"):
            node.copy(child, group)
        group.attrs["shape"] = tuple(int(x) for x in node.attrs["shape"])
        for key in ("encoding-type", "encoding_type", "encoding-version", "encoding_version"):
            group.attrs.pop(key, None)
        for child in group.values():
            set_array_attrs(child)
    else:
        dataset = out.create_dataset(key, data=node[()])
        set_array_attrs(dataset)


def write_index_dataframe(out, key, index_values):
    group = out.create_group(key)
    set_dataframe_attrs(group)
    dataset = group.create_dataset("_index", data=string_array(index_values))
    set_string_array_attrs(dataset)


def convert_one(input_path, output_path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    with h5py.File(input_path, "r") as src, h5py.File(tmp_path, "w") as out:
        copy_matrix(src, out, "X")
        write_index_dataframe(out, "obs", decode_array(src["obs/_index"][()]))
        write_index_dataframe(out, "var", decode_array(src["var/_index"][()]))
        obsm = out.create_group("obsm")
        spatial = obsm.create_dataset("spatial", data=np.asarray(src["obsm/spatial"][()], dtype=float))
        set_array_attrs(spatial)
    tmp_path.replace(output_path)


compat_path = Path(os.environ["METHOD_COMPAT_PATH"])
compat_path.mkdir(parents=True, exist_ok=True)
for input_path in sorted(Path("data").glob("*.h5ad")):
    output_path = compat_path / input_path.name
    convert_one(input_path, output_path)
    print(f"wrote {output_path}")
PY

install_method() {
  local method="$1"
  case "${method}" in
    probe)
      "${venv_path}/bin/python" - <<'PY'
import importlib

targets = [
    ("cast", ["CAST"]),
    ("gpsa", ["gpsa"]),
    ("paste", ["paste"]),
    ("paste2", ["paste2"]),
    ("santo", ["SANTO", "santo"]),
    ("scslat", ["scSLAT"]),
    ("slat", ["scSLAT"]),
    ("spacel", ["SPACEL", "spacel"]),
    ("spateo", ["spateo"]),
    ("stair", ["STAIR", "stair"]),
    ("stalign", ["STalign"]),
    ("staligner", ["STAligner"]),
]
for target, mods in targets:
    last_error = None
    for mod in mods:
        try:
            importlib.import_module(mod)
            print(f"{target} {mod} OK")
            break
        except Exception as exc:
            last_error = f"{mod} {type(exc).__name__}: {str(exc)[:160]}"
    else:
        print(f"{target} FAIL {last_error}")
PY
      ;;
    paste)
      uv pip install --python "${venv_path}/bin/python" "pot==0.9.0" matplotlib seaborn paste-bio
      ;;
    paste2)
      uv pip install --python "${venv_path}/bin/python" paste2
      ;;
    gpsa)
      uv pip install --python "${venv_path}/bin/python" gpsa-engelhardt
      ;;
    stair)
      uv pip install --python "${venv_path}/bin/python" STAIR-tools
      ;;
    spacel)
      uv pip install --python "${venv_path}/bin/python" SPACEL
      ;;
    spateo)
      uv pip install --python "${venv_path}/bin/python" spateo-release
      ;;
    stalign)
      uv pip install --python "${venv_path}/bin/python" "STalign @ git+https://github.com/JEFworks-Lab/STalign.git"
      ;;
    scslat)
      uv pip install --python "${venv_path}/bin/python" scSLAT
      ;;
    slat)
      echo "Installing scSLAT for SLAT; scSLAT implements the SLAT method family." >&2
      install_method scslat
      ;;
    cast)
      echo "Installing CAST from GitHub; this may require PyTorch/DGL-compatible runtime packages." >&2
      uv pip install --python "${venv_path}/bin/python" "CAST @ git+https://github.com/wanglab-broad/CAST.git"
      ;;
    staligner)
      echo "Installing STAligner from GitHub with PyG extension dependencies; upstream still recommends older Python/PyTorch stacks." >&2
      uv pip install --python "${venv_path}/bin/python" \
        "torch==2.3.1" \
        "torch-geometric==2.5.3" \
        annoy \
        hnswlib
      uv pip install --python "${venv_path}/bin/python" --no-build-isolation \
        torch-scatter \
        torch-sparse
      uv pip install --python "${venv_path}/bin/python" "STAligner @ git+https://github.com/zhoux85/STAligner.git"
      ;;
    santo)
      echo "SANTO's PyPI releases currently depend on old Torch wheels that may not support Python 3.11; attempting install anyway." >&2
      uv pip install --python "${venv_path}/bin/python" santo
      ;;
    all)
      install_method paste
      install_method paste2
      install_method gpsa
      install_method stair
      install_method spacel
      install_method spateo
      install_method stalign
      install_method scslat
      restore_paste_compat
      ;;
    all-paper)
      install_method all
      install_method slat
      install_method cast
      install_method staligner
      install_method santo
      restore_paste_compat
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

restore_paste_compat() {
  uv pip install --python "${venv_path}/bin/python" "pot==0.9.0" paste-bio
}

restore_paste_compat_best_effort() {
  local failure_log="${compat_path}/method_install_failures.tsv"
  if restore_paste_compat; then
    return 0
  fi
  printf '%s\t%s\n' "paste-compat" "failed to restore pot==0.9.0 after bulk installs" >> "${failure_log}"
  echo "WARNING: failed to restore PASTE/POT compatibility after bulk installs." >&2
}

install_method_best_effort() {
  local method="$1"
  local failure_log="${compat_path}/method_install_failures.tsv"
  mkdir -p "${compat_path}"
  if install_method "${method}"; then
    return 0
  fi
  printf '%s\t%s\n' "${method}" "install failed" >> "${failure_log}"
  echo "WARNING: method install failed for ${method}; continuing because this target is best-effort." >&2
}

if (( ${#methods[@]} > 0 )); then
  for method in "${methods[@]}"; do
    case "${method}" in
      paper-common)
        install_method_best_effort gpsa
        install_method_best_effort paste
        install_method_best_effort paste2
        install_method_best_effort scslat
        install_method_best_effort spacel
        install_method_best_effort spateo
        install_method_best_effort stair
        install_method_best_effort stalign
        restore_paste_compat_best_effort
        ;;
      paper-all|all-paper)
        install_method_best_effort gpsa
        install_method_best_effort paste
        install_method_best_effort paste2
        install_method_best_effort scslat
        install_method_best_effort spacel
        install_method_best_effort spateo
        install_method_best_effort stair
        install_method_best_effort stalign
        install_method_best_effort slat
        install_method_best_effort cast
        install_method_best_effort santo
        install_method_best_effort staligner
        restore_paste_compat_best_effort
        ;;
      *)
        install_method "${method}"
        ;;
    esac
  done
fi

echo "Created method venv: ${venv_path}"
echo "Compatible h5ad copies: ${compat_path}"
echo "Use ${venv_path}/bin/python for package-backed method runs."
"""


def _agent_python_script(*, use_method_venv: bool = True) -> str:
    method_venv_block = ""
    if use_method_venv:
        method_venv_block = """if [[ -x ".method-venv/bin/python" ]]; then
  exec ./.method-venv/bin/python "$@"
fi
"""
    return f"""#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="${{UV_CACHE_DIR:-/tmp/uv-cache}}"
export TMPDIR="${{TMPDIR:-/tmp}}"
{method_venv_block}\
exec uv run --with numpy --with scipy --with pandas --with scikit-learn --with h5py python "$@"
"""


def _method_env_manifest_path() -> Path:
    # P2B_METHOD_ENV_PROFILE=legacy reproduces the environments the 121-task results ran with (see that file's header)
    name = "spatial_method_envs_legacy.toml" if os.environ.get("P2B_METHOD_ENV_PROFILE") == "legacy" else "spatial_method_envs.toml"
    return Path(__file__).resolve().parents[1] / "benchmarks" / "spatial_alignment" / name


def main() -> None:
    args = parse_args()
    if args.blind_solve and args.link_source_data:
        raise SystemExit("--blind-solve cannot be combined with --link-source-data")
    task = load_task_spec(args.task)
    repo_root = Path.cwd()
    workspace = args.workspace_root / task.id
    if workspace.exists() and not args.force:
        raise SystemExit(f"{workspace} already exists; pass --force to replace generated files")

    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "data").mkdir(exist_ok=True)
    (workspace / "submission" / "aligned_coordinates").mkdir(parents=True, exist_ok=True)
    if args.clean_submission:
        for path in (workspace / "submission" / "aligned_coordinates").glob("*.csv"):
            path.unlink()
        scores_path = workspace / "scores.json"
        if scores_path.exists():
            scores_path.unlink()
    warm_start_files: list[str] = []
    if args.warm_start_from is not None:
        source = args.warm_start_from
        if not source.is_dir():
            raise SystemExit(f"--warm-start-from {source} is not a directory")
        target = workspace / "submission" / "aligned_coordinates"
        for csv in sorted(source.glob("*.csv")):
            shutil.copy2(csv, target / csv.name)
            warm_start_files.append(csv.name)
        if not warm_start_files:
            raise SystemExit(f"--warm-start-from {source} holds no coordinate CSVs")

    if args.no_method_helper:
        for path in workspace.glob(".method-venv*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(workspace / ".method-cache", ignore_errors=True)

    copy_agent_data = args.copy_agent_data or args.blind_solve
    cache_root = args.workspace_root / ".sanitized_h5ad_cache"
    for spec in task.all_data_specs:
        slice_id, path = spec.split("=", 1)
        source = Path(path)
        agent_visible = _ensure_agent_visible_data(source, cache_root, use_source_link=args.link_source_data)
        linked = workspace / "data" / f"{slice_id}{source.suffix}"
        _link_or_copy(agent_visible, linked, copy=copy_agent_data)

    stale_split = workspace / "data" / "split_manifest.json"
    if stale_split.exists() or stale_split.is_symlink():
        stale_split.unlink()

    instruction_path = args.task.parent / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8") if instruction_path.exists() else task.payload.get("name", task.id)
    if not args.link_source_data:
        instruction = _redact_private_instruction(instruction)
    relative_task_path = args.task
    prompt = _task_prompt(
        relative_task_path,
        instruction,
        task.id,
        task.all_data_specs,
        task.runnable_slice_specs,
        workspace=workspace,
        has_split_manifest=task.split_manifest_path is not None,
        public_score_split=task.public_score_split,
        withheld_score_split=task.score_split,
        blind_solve=args.blind_solve,
        expose_method_helper=not args.no_method_helper,
        warm_start_files=warm_start_files,
    )
    (workspace / "TASK.md").write_text(prompt, encoding="utf-8")

    score_path = workspace / "score.sh"
    bootstrap_path = workspace / "bootstrap_method_venv.sh"
    agent_python_path = workspace / "agent_python.sh"
    method_manifest_path = workspace / "spatial_method_envs.toml"
    method_manifest_source = _method_env_manifest_path()
    if args.blind_solve:
        if score_path.exists() or score_path.is_symlink():
            score_path.unlink()
        if args.no_method_helper:
            if bootstrap_path.exists() or bootstrap_path.is_symlink():
                bootstrap_path.unlink()
            if method_manifest_path.exists() or method_manifest_path.is_symlink():
                method_manifest_path.unlink()
        else:
            bootstrap_path.write_text(_blind_bootstrap_script(), encoding="utf-8")
            bootstrap_path.chmod(0o755)
            shutil.copy2(method_manifest_source, method_manifest_path)
        agent_python_path.write_text(_agent_python_script(use_method_venv=not args.no_method_helper), encoding="utf-8")
        agent_python_path.chmod(0o755)
    else:
        script = _score_script(repo_root, relative_task_path, workspace, task.public_score_split)
        score_path.write_text(script, encoding="utf-8")
        score_path.chmod(0o755)

        if args.no_method_helper:
            if bootstrap_path.exists() or bootstrap_path.is_symlink():
                bootstrap_path.unlink()
            if method_manifest_path.exists() or method_manifest_path.is_symlink():
                method_manifest_path.unlink()
        else:
            bootstrap_script = _bootstrap_script(repo_root, workspace)
            bootstrap_path.write_text(bootstrap_script, encoding="utf-8")
            bootstrap_path.chmod(0o755)
            shutil.copy2(method_manifest_source, method_manifest_path)
        if agent_python_path.exists() or agent_python_path.is_symlink():
            agent_python_path.unlink()

    print(f"Wrote {workspace}")
    print(f"Agent prompt: {workspace / 'TASK.md'}")
    if args.blind_solve:
        print("Scorer: withheld for blind solve")
    else:
        print(f"Scorer: {workspace / 'score.sh'}")


if __name__ == "__main__":
    main()
