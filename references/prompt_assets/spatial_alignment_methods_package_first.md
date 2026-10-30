## Benchmark Context (Neutral Full Package Access)

The benchmark paper used multiple spatial-alignment method families on related
tasks. In this prompt variant, the solve workspace exposes helper targets for
all supported package families. You are not given paper rankings, paper scores,
or task-specific recommendations. Treat every package below as an optional tool
candidate to evaluate with allowed input-only diagnostics.

Before spending substantial time on package experiments, first create a complete
valid submission for every required output CSV. Then inspect the available
package environments and choose the package-backed method or methods that look
most promising for this task. There is no quota of packages or method families
to try, and PASTE/PASTE2 are not required defaults. A useful package attempt
means importing the method package from its listed `.method-venv*/bin/python`,
running its alignment or transform API on the task data or a representative
adjacent-slice subset, and recording enough output to tell whether it succeeded.
If plausible package attempts fail, stall, or are clearly infeasible for the task
size, record that briefly and move on or fall back to a custom method. Keep the
strongest output under allowed input-only diagnostics.
Do not treat quiet first-import startup as an immediate package failure:
Torch/numba/JIT-heavy methods can spend a minute or two importing before they
print anything. Use a deliberate bounded timeout, and only mark an import as
failed after an actual traceback/import error or a timeout that is too expensive
for this task.

Method environment rule: if you want to use a package, use the Python executable
listed for that package in `.method-cache/METHOD_ENVS.md` or
`.method-cache/compat_h5ad/method_envs.tsv`. Do not import paper-method
packages through `./agent_python.sh` or base `.method-venv/bin/python`; those
are for general data handling only. The environment layout is declared in
`spatial_method_envs.toml` and copied to `.method-cache/spatial_method_envs.toml`
after bootstrap. The current minimal grouping is: `paste`/`paste2` ->
`.method-venv-ot`, `slat`/`scslat` -> `.method-venv-scslat`, and all other
method targets use their own listed env because their dependency stacks are not
reliably compatible.

Internet access is available for public method documentation and package API
lookup. You may use web search, `curl`, `python -m pip show`, GitHub READMEs,
ReadTheDocs pages, and PyPI pages to learn how to call the installed packages.
This does not permit searching for benchmark source, scorer code, prior run
artifacts, hidden labels, paper result tables, or anything outside the allowed
task inputs. If you use online docs, record the URL and the package/API call you
tried in a short local note or `submission/report.json`.

### Package Targets

The table below is shown in a shuffled order for this prompt variant. The order
is not a ranking or recommendation.

| Helper target | Method family exposed | Import/module hint | Python/package target |
|---|---|---|---|
| `staligner` | STAligner | `STAligner` case-sensitive | `STAligner @ git+https://github.com/zhoux85/STAligner.git` |
| `spateo` | Spateo rigid / non-rigid family | `spateo` | `spateo-release` |
| `gpsa` | GPSA | `gpsa` | `gpsa-engelhardt` |
| `cast` | CAST | `CAST` | `CAST @ git+https://github.com/wanglab-broad/CAST.git` |
| `stair` | STAIR | `STAIR` or `stair` | `STAIR-tools` |
| `santo` | SANTO | `SANTO` or `santo` | `santo` |
| `spacel` | SPACEL | `SPACEL` or `spacel` | `SPACEL` |
| `scslat` | scSLAT / SLAT family | `scSLAT` | `scSLAT` |
| `stalign` | STalign | `import STalign.STalign as stalign` | `STalign @ git+https://github.com/JEFworks-Lab/STalign.git` |
| `slat` | SLAT family | helper alias for `scslat` | helper alias for `scslat` |
| `paste2` | PASTE2 | `paste2` / `from paste2 import PASTE2` | merged OT env `.method-venv-ot` |
| `paste` | PASTE / PASTE_p0 | `paste` | merged OT env `.method-venv-ot` |

### Documentation Menu

The table below is a neutral menu of public documentation and examples. The
order is still not a ranking. Use the docs to find runnable APIs, then test a
small representative pair or subset before launching expensive full-task runs.

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

### API And Example Entry Points

Use the examples below as starting points for package-backed alignment attempts
on the task data. Signature/introspection snippets are included only to handle
package-version differences; do not stop at API discovery if a method is usable.
The goal is to produce aligned coordinate CSVs from the strongest package-backed
or custom approach.

```bash
# Common setup after `./bootstrap_method_venv.sh paper-all`: load the
# AnnData-compatible files that package APIs should read.
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
# PASTE / PASTE2: optimal-transport pairwise alignment and serial stacking.
# Both are installed in the merged OT env. Compatibility aliases
# .method-venv-paste and .method-venv-paste2 point here too.
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
# Spateo: rigid/nonrigid pairwise slice alignment. Output coordinates are added
# to each AnnData under key_added, key_added + "_rigid", and key_added + "_nonrigid".
./.method-venv-spateo/bin/python - <<'PY'
import inspect, spateo as st
print("align members:", [n for n in dir(st.align) if "align" in n.lower() or "morpho" in n.lower()][:30])
print(inspect.signature(st.align.morpho_align))
print("call pattern: st.align.morpho_align(model_ref=adata_ref, model_tgt=adata_tgt, spatial_key='spatial', key_added='spateo_align')")
PY
```

```bash
# STalign: diffeomorphic image/point alignment. Useful when geometry or tissue
# density is informative; rasterize coordinates/expression into images, then
# use LDDMM and transform helpers to map original points.
./.method-venv-stalign/bin/python - <<'PY'
import inspect
import STalign.STalign as stalign
for name in ["rasterize", "L_T_from_points", "LDDMM", "build_transform", "transform_points_source_to_target"]:
    obj = getattr(stalign, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
print("typical flow: rasterize coords/expression -> fit LDDMM -> build_transform -> map original points")
PY
```

```bash
# CAST: representation learning across samples. CAST_MARK accepts dictionaries
# of sample coordinates and expression matrices; use learned embeddings or any
# documented registration utilities only if they produce mappable coordinates.
./.method-venv-cast/bin/python - <<'PY'
import inspect, CAST
print(inspect.signature(CAST.CAST_MARK))
print("candidate CAST names:", [n for n in dir(CAST) if any(k in n.lower() for k in ["mark", "align", "reg", "project"])][:40])
print("call pattern: CAST.CAST_MARK(coords_raw={sample: xy}, exp_dict={sample: X}, output_path='cast_out', ...)")
PY
```

```bash
# SPACEL/Scube: pairwise multi-slice alignment. It may require spatial-domain
# labels in .obs; if labels are absent, either create unsupervised domains from
# allowed expression/coordinates or mark the method infeasible.
./.method-venv-spacel/bin/python - <<'PY'
import inspect
from SPACEL import Scube
print(inspect.signature(Scube.alignment.align))
print("Scube members:", [n for n in dir(Scube) if "align" in n.lower() or "spatial" in n.lower()][:40])
print("data pattern: AnnData objects with .obsm['spatial']; optional unsupervised domains can be derived from allowed inputs")
PY
```

```bash
# STAligner: graph/embedding alignment. It usually starts by building a spatial
# neighbor graph, training embeddings, then deriving a transform or matching from
# aligned embeddings and original coordinates.
./.method-venv-staligner/bin/python - <<'PY'
import inspect, STAligner
for name in ["Cal_Spatial_Net", "Stats_Spatial_Net", "train_STAligner", "mclust_R"]:
    obj = getattr(STAligner, name, None)
    print(name, inspect.signature(obj) if obj is not None else "missing")
print("expected data pattern: list/concat AnnData with .obsm['spatial']; output embeddings often in .obsm")
PY
```

```bash
# scSLAT / SLAT: graph matching and spatial-domain alignment family. Entry-point
# names vary by release; find callable alignment APIs, then adapt a working one.
./.method-venv-scslat/bin/python - <<'PY'
import importlib, inspect, pkgutil
mods = [m.name for m in pkgutil.iter_modules() if "slat" in m.name.lower()]
print("modules:", mods[:30])
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
                print(mod_name, name, inspect.signature(obj))
            except Exception:
                pass
PY
```

```bash
# GPSA: Gaussian-process spatial alignment. It expects numeric coordinates and
# expression/features in tensors or arrays; fit on a manageable subset first.
./.method-venv-gpsa/bin/python - <<'PY'
import inspect, gpsa
print("gpsa members:", [n for n in dir(gpsa) if "GPSA" in n or "Variational" in n][:30])
print("VariationalGPSA", inspect.signature(gpsa.VariationalGPSA))
print("data pattern: data_dict with spatial_coords and outputs per view, then model.train() / forward() to get aligned coordinates")
PY
```

```bash
# STAIR and SANTO: discover installed entry points and adapt a working alignment API.
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

### Bootstrap Pattern

If `./bootstrap_method_venv.sh` exists, prefer it over ad hoc installs because it
reads `spatial_method_envs.toml`, creates a shared `.method-venv/`, grouped or
isolated `.method-venv*` method environments, and standard AnnData-compatible
inputs in `.method-cache/compat_h5ad/`.
For package APIs that call `scanpy.read_h5ad` or `anndata.read_h5ad`, prefer the
files under `.method-cache/compat_h5ad/`; the task `data/*.h5ad` files are the
same sanitized inputs but are optimized for direct `h5py` access.

Use the all-package target to best-effort install every supported method
environment up front, then probe what imports are actually available:

```bash
./bootstrap_method_venv.sh paper-all
./bootstrap_method_venv.sh probe
cat .method-cache/METHOD_ENVS.md
cat .method-cache/spatial_method_envs.toml
```

Install failures are ordinary package availability decisions, not task failures.
They are recorded under `.method-cache/compat_h5ad/method_install_failures.tsv`
when the best-effort targets are used.
The helper also writes `.method-cache/METHOD_ENVS.md` and
`.method-cache/compat_h5ad/method_envs.tsv` listing the Python executable for
each installed method, plus the copied manifest at
`.method-cache/spatial_method_envs.toml`.

Use the listed method Python executables for package-backed attempts. Compatible
packages are merged where validated, but several method families still need
separate stacks:

```bash
./bootstrap_method_venv.sh --recreate ot
./bootstrap_method_venv.sh --recreate staligner
./bootstrap_method_venv.sh --recreate spateo
./bootstrap_method_venv.sh --recreate gpsa
./bootstrap_method_venv.sh --recreate cast
./bootstrap_method_venv.sh --recreate stair
./bootstrap_method_venv.sh --recreate santo
./bootstrap_method_venv.sh --recreate spacel
./bootstrap_method_venv.sh --recreate scslat
./bootstrap_method_venv.sh --recreate stalign
./bootstrap_method_venv.sh --recreate slat
```

```bash
./.method-venv-staligner/bin/python your_method_script.py
./.method-venv-spateo/bin/python your_method_script.py
./.method-venv-gpsa/bin/python your_method_script.py
./.method-venv-stalign/bin/python your_method_script.py
./.method-venv-ot/bin/python your_paste_or_paste2_script.py
```

If uv virtualenv creation fails and a conda implementation is available on PATH,
the helper can try conda/mamba/micromamba as a last-resort environment creator:

```bash
./bootstrap_method_venv.sh --conda-fallback auto --recreate ot
./bootstrap_method_venv.sh --conda-fallback auto --recreate staligner
```

### Selection Guidance

- Keep a valid submission on disk before trying expensive packages.
- In this full-package prompt, do not stop at merely listing or probing packages
  if a package-backed method looks promising. Use the package environment(s)
  that fit the task best, and invest effort in the strongest candidate rather
  than satisfying an artificial package-count target.
- Compare candidate outputs using only allowed inputs: expression consistency,
  spatial smoothness, neighbor preservation, overlap plausibility, and
  slice-to-slice geometry.
- Do not search for hidden labels, scorer internals, previous runs, source
  benchmark metadata, or paper result tables.
- If a package fails to install, import, or fit the task API, record that briefly
  and move to another package or a custom method.
- If a package import is merely slow, wait through normal first-import/JIT
  startup or run a small representative probe with an explicit timeout before
  declaring the method infeasible.
- Do not treat package name order, import success order, or prebootstrap log
  order as evidence of method quality.
