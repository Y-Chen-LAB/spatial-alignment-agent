#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_spatial_method_venv.sh [options] [method ...]

Create workspace-local Python environments for package-backed spatial alignment.
The default shared `.method-venv` is intentionally small. Compatible method
families are grouped where practical; methods with conflicting pandas/anndata/
POT/torch stacks remain isolated. The layout is declared in
`spatial_method_envs.toml`; this script translates that manifest into uv
virtualenvs, with optional conda/mamba fallback for env creation.

Methods:
  ot          Create `.method-venv-ot` with PASTE / paste-bio and PASTE2
  paste       Alias target for the merged `.method-venv-ot`
  paste2      Alias target for the merged `.method-venv-ot`
  gpsa        Create `.method-venv-gpsa` with GPSA
  stair       Create `.method-venv-stair` with STAIR
  spacel      Best-effort create `.method-venv-spacel`
  spateo      Best-effort create `.method-venv-spateo`
  stalign     Create `.method-venv-stalign` with STalign
  scslat      Best-effort create `.method-venv-scslat`
  slat        Alias for scslat
  cast        Best-effort create `.method-venv-cast`
  staligner   Best-effort create `.method-venv-staligner`
  santo       Best-effort create `.method-venv-santo`
  probe       Print import/API status for all available method envs
  paper-common Install the most commonly usable grouped/isolated method envs
  paper-all   Best-effort install every supported grouped/isolated method env
  all         Alias for paper-common
  all-paper   Alias for paper-all

Options:
  --workspace PATH       Workspace root; defaults to current directory
  --manifest PATH        Method env manifest; defaults to spatial_method_envs.toml
  --venv-name NAME       Override the env name for explicit single-method installs
  --compat-dir PATH      Compatible h5ad directory; defaults to .method-cache/compat_h5ad
  --recreate             Remove and recreate selected envs before installing
  --conda-fallback MODE  Conda fallback policy if uv env creation fails: auto, never (default: auto)
  -h, --help             Show this help text
EOF
}

workspace="."
manifest_path=""
venv_name=".method-venv"
compat_dir=".method-cache/compat_h5ad"
conda_fallback="auto"
recreate=0
custom_venv_name=0
declare -a methods=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      workspace="$2"
      shift 2
      ;;
    --manifest)
      manifest_path="$2"
      shift 2
      ;;
    --venv-name)
      venv_name="$2"
      custom_venv_name=1
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
    --conda-fallback)
      conda_fallback="$2"
      shift 2
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

case "${conda_fallback}" in
  auto|never) ;;
  *)
    echo "Unknown --conda-fallback mode: ${conda_fallback}" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
initial_cwd="$(pwd)"
workspace="$(cd "${workspace}" && pwd)"
venv_path="${workspace}/${venv_name}"
compat_path="${workspace}/${compat_dir}"
cache_path="${workspace}/.method-cache"
failure_log="${compat_path}/method_install_failures.tsv"
env_log="${compat_path}/method_envs.tsv"
readme_path="${cache_path}/METHOD_ENVS.md"
manifest_cache_path="${cache_path}/spatial_method_envs.toml"
cd "${workspace}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${cache_path}/matplotlib}"

mkdir -p "${compat_path}" "${MPLCONFIGDIR}"
touch "${failure_log}" "${env_log}"

resolve_manifest_path() {
  if [[ -n "${manifest_path}" ]]; then
    if [[ -f "${manifest_path}" ]]; then
      (cd "$(dirname "${manifest_path}")" && pwd)
      return 0
    fi
    if [[ "${manifest_path}" != /* && -f "${initial_cwd}/${manifest_path}" ]]; then
      (cd "$(dirname "${initial_cwd}/${manifest_path}")" && pwd)
      return 0
    fi
    if [[ -f "${workspace}/${manifest_path}" ]]; then
      (cd "$(dirname "${workspace}/${manifest_path}")" && pwd)
      return 0
    fi
    echo "Manifest not found: ${manifest_path}" >&2
    return 1
  fi
  if [[ -f "${workspace}/spatial_method_envs.toml" ]]; then
    echo "${workspace}"
    return 0
  fi
  if [[ -f "${script_dir}/../benchmarks/spatial_alignment/spatial_method_envs.toml" ]]; then
    (cd "${script_dir}/../benchmarks/spatial_alignment" && pwd)
    return 0
  fi
  echo "Could not find spatial_method_envs.toml" >&2
  return 1
}

manifest_dir="$(resolve_manifest_path)"
if [[ -n "${manifest_path}" && -f "${manifest_path}" ]]; then
  manifest_path="$(cd "$(dirname "${manifest_path}")" && pwd)/$(basename "${manifest_path}")"
elif [[ -n "${manifest_path}" && "${manifest_path}" != /* && -f "${initial_cwd}/${manifest_path}" ]]; then
  manifest_path="$(cd "$(dirname "${initial_cwd}/${manifest_path}")" && pwd)/$(basename "${manifest_path}")"
elif [[ -n "${manifest_path}" && -f "${workspace}/${manifest_path}" ]]; then
  manifest_path="$(cd "$(dirname "${workspace}/${manifest_path}")" && pwd)/$(basename "${manifest_path}")"
else
  manifest_path="${manifest_dir}/spatial_method_envs.toml"
fi

cp "${manifest_path}" "${manifest_cache_path}"

if command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import tomllib
PY
then
  manifest_python=(python3)
else
  manifest_python=(uv run --python 3.11 python)
fi

manifest_query() {
  local query="$1"
  shift || true
  SPATIAL_METHOD_MANIFEST="${manifest_path}" "${manifest_python[@]}" - "${query}" "$@" <<'PY'
import os
import sys
import tomllib


def fail(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


with open(os.environ["SPATIAL_METHOD_MANIFEST"], "rb") as handle:
    manifest = tomllib.load(handle)

query = sys.argv[1]
args = sys.argv[2:]
methods = manifest.get("methods", {})
aliases = manifest.get("aliases", {})


def canonical(name: str) -> str:
    target = aliases.get(name, name)
    if target not in methods:
        fail(f"Unknown method: {name}")
    return target


def method_config(name: str) -> dict:
    return methods[canonical(name)]


def emit_lines(values) -> None:
    for value in values:
        print(value)


if query == "canonical":
    print(canonical(args[0]))
elif query == "has-method":
    canonical(args[0])
elif query == "env":
    print(method_config(args[0])["env"])
elif query == "python":
    print(method_config(args[0]).get("python", "3.11"))
elif query == "stack":
    print(method_config(args[0]).get("stack", "loose"))
elif query == "records":
    cfg = method_config(args[0])
    emit_lines(cfg.get("records", [canonical(args[0])]))
elif query == "env-aliases":
    emit_lines(method_config(args[0]).get("env_aliases", []))
elif query == "packages":
    field = args[1] if len(args) > 1 else "packages"
    emit_lines(method_config(args[0]).get(field, []))
elif query == "post-install":
    print(method_config(args[0]).get("post_install", ""))
elif query == "group":
    group_name = args[0].replace("-", "_")
    emit_lines(manifest.get("groups", {}).get(group_name, []))
elif query == "probe-order":
    emit_lines(manifest.get("groups", {}).get("probe_order", []))
elif query == "stack-packages":
    stack_name = args[0]
    stacks = manifest.get("stacks", {})
    if stack_name not in stacks:
        fail(f"Unknown stack: {stack_name}")
    emit_lines(stacks[stack_name].get("packages", []))
elif query == "base":
    print(manifest.get("base", {}).get(args[0], ""))
elif query == "known-methods":
    emit_lines(sorted(set(methods) | set(aliases)))
elif query == "manifest-summary":
    print(f"manifest\t{os.environ['SPATIAL_METHOD_MANIFEST']}")
    print(f"base\t{manifest.get('base', {}).get('env', '.method-venv')}\tpython={manifest.get('base', {}).get('python', '3.11')}\tstack={manifest.get('base', {}).get('stack', 'old')}")
    for name in sorted(methods):
        cfg = methods[name]
        packages = len(cfg.get("packages", [])) + len(cfg.get("no_build_isolation_packages", [])) + len(cfg.get("post_packages", []))
        print(f"{name}\t{cfg.get('env')}\tpython={cfg.get('python', '3.11')}\tstack={cfg.get('stack', 'loose')}\tpackages={packages}")
else:
    fail(f"Unknown manifest query: {query}", 2)
PY
}

if [[ ${custom_venv_name} -eq 0 ]]; then
  manifest_base_env="$(manifest_query base env)"
  if [[ -n "${manifest_base_env}" ]]; then
    venv_name="${manifest_base_env}"
    venv_path="${workspace}/${venv_name}"
  fi
fi

find_conda() {
  command -v micromamba 2>/dev/null || command -v mamba 2>/dev/null || command -v conda 2>/dev/null || true
}

create_uv_venv() {
  local env_path="$1"
  local python_version="${2:-3.11}"
  if [[ ${recreate} -eq 1 && -d "${env_path}" ]]; then
    rm -rf "${env_path}"
  fi
  if [[ ! -x "${env_path}/bin/python" ]]; then
    uv venv "${env_path}" --python "${python_version}" --seed
  fi
}

create_conda_env() {
  local env_path="$1"
  local python_version="${2:-3.11}"
  local conda_bin
  conda_bin="$(find_conda)"
  if [[ -z "${conda_bin}" ]]; then
    return 1
  fi
  if [[ ${recreate} -eq 1 && -d "${env_path}" ]]; then
    rm -rf "${env_path}"
  fi
  if [[ -x "${env_path}/bin/python" ]]; then
    return 0
  fi
  case "$(basename "${conda_bin}")" in
    micromamba|mamba)
      "${conda_bin}" create -y -p "${env_path}" "python=${python_version}" pip
      ;;
    conda)
      "${conda_bin}" create -y -p "${env_path}" "python=${python_version}" pip
      ;;
  esac
}

create_env() {
  local env_path="$1"
  local python_version="${2:-3.11}"
  if create_uv_venv "${env_path}" "${python_version}"; then
    return 0
  fi
  if [[ "${conda_fallback}" == "auto" ]]; then
    echo "WARNING: uv venv failed for ${env_path}; trying conda/mamba fallback if available." >&2
    create_conda_env "${env_path}" "${python_version}"
    return $?
  fi
  return 1
}

pip_install() {
  local python_bin="$1"
  shift
  if uv pip install --python "${python_bin}" "$@"; then
    return 0
  fi
  echo "WARNING: uv pip install failed; trying python -m pip fallback for ${python_bin}." >&2
  "${python_bin}" -m pip install "$@"
}

install_common_old_stack() {
  local python_bin="$1"
  pip_install "${python_bin}" -U pip setuptools wheel
  pip_install "${python_bin}" \
    "numpy<2" \
    "pandas<2" \
    "scipy<1.12" \
    "h5py<3.15" \
    "anndata==0.7.8" \
    scikit-learn \
    scikit-image \
    matplotlib \
    seaborn
}

install_common_modern_stack() {
  local python_bin="$1"
  pip_install "${python_bin}" -U pip setuptools wheel
  pip_install "${python_bin}" \
    "numpy<2" \
    "pandas>=2.1,<2.3" \
    "scipy<1.12" \
    "h5py<3.15" \
    "anndata>=0.10,<0.11" \
    scikit-learn \
    scikit-image \
    matplotlib \
    seaborn
}

install_common_loose_stack() {
  local python_bin="$1"
  pip_install "${python_bin}" -U pip setuptools wheel
  pip_install "${python_bin}" \
    "numpy<2" \
    "h5py<3.15" \
    scipy \
    pandas \
    anndata \
    scikit-learn \
    scikit-image \
    matplotlib \
    seaborn
}

install_common_stack() {
  local python_bin="$1"
  local stack_name="$2"
  local packages=()
  local package
  pip_install "${python_bin}" -U pip setuptools wheel
  while IFS= read -r package; do
    [[ -n "${package}" ]] && packages+=("${package}")
  done < <(manifest_query stack-packages "${stack_name}")
  if (( ${#packages[@]} > 0 )); then
    pip_install "${python_bin}" "${packages[@]}"
  fi
}

materialize_compat_h5ad() {
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


def clear_encoding_attrs(node):
    for key in ("encoding-type", "encoding_type", "encoding-version", "encoding_version"):
        node.attrs.pop(key, None)


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
        clear_encoding_attrs(group)
        for child in group.values():
            clear_encoding_attrs(child)
    else:
        dataset = out.create_dataset(key, data=node[()])
        clear_encoding_attrs(dataset)


def write_index_dataframe(out, key, index_values):
    group = out.create_group(key)
    set_dataframe_attrs(group)
    dataset = group.create_dataset("_index", data=string_array(index_values))
    clear_encoding_attrs(dataset)


def convert_one(input_path, output_path):
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    with h5py.File(input_path, "r") as src, h5py.File(tmp_path, "w") as out:
        copy_matrix(src, out, "X")
        write_index_dataframe(out, "obs", decode_array(src["obs/_index"][()]))
        write_index_dataframe(out, "var", decode_array(src["var/_index"][()]))
        obsm = out.create_group("obsm")
        clear_encoding_attrs(obsm)
        spatial = obsm.create_dataset("spatial", data=np.asarray(src["obsm/spatial"][()], dtype=float))
        clear_encoding_attrs(spatial)
    tmp_path.replace(output_path)


compat_path = Path(os.environ["METHOD_COMPAT_PATH"])
compat_path.mkdir(parents=True, exist_ok=True)
inputs = sorted(Path("data").glob("*.h5ad"))
if not inputs:
    raise SystemExit("No input h5ad files found under data/")
for input_path in inputs:
    output_path = compat_path / input_path.name
    convert_one(input_path, output_path)
    print(f"wrote {output_path}")
PY
}

method_env_name() {
  local method="$1"
  manifest_query env "${method}"
}

ensure_env_aliases() {
  local method="$1"
  local env_path="$2"
  local alias
  while IFS= read -r alias; do
    [[ -n "${alias}" ]] || continue
    if [[ "${workspace}/${alias}" == "${env_path}" ]]; then
      continue
    fi
    if [[ -L "${workspace}/${alias}" ]]; then
      rm -f "${workspace}/${alias}"
    elif [[ -e "${workspace}/${alias}" ]]; then
      if [[ ${recreate} -eq 1 ]]; then
        rm -rf "${workspace:?}/${alias}"
      else
        continue
      fi
    fi
    ln -s "$(basename "${env_path}")" "${workspace}/${alias}"
  done < <(manifest_query env-aliases "${method}")
}

record_env() {
  local method="$1"
  local env_path="$2"
  printf '%s\t%s\t%s\t%s\n' "${method}" "${env_path}" "${env_path}/bin/python" "${compat_path}" >> "${env_log}"
}

record_failure() {
  local method="$1"
  local message="$2"
  printf '%s\t%s\n' "${method}" "${message}" >> "${failure_log}"
}

install_method_into_env() {
  local method="$1"
  local env_path="$2"
  local python_bin="${env_path}/bin/python"
  local python_version
  local stack_name
  local post_install
  python_version="$(manifest_query python "${method}")"
  stack_name="$(manifest_query stack "${method}")"
  create_env "${env_path}" "${python_version}"
  install_common_stack "${python_bin}" "${stack_name}"

  local packages=()
  local package
  while IFS= read -r package; do
    [[ -n "${package}" ]] && packages+=("${package}")
  done < <(manifest_query packages "${method}" packages)
  if (( ${#packages[@]} > 0 )); then
    pip_install "${python_bin}" "${packages[@]}"
  fi

  packages=()
  while IFS= read -r package; do
    [[ -n "${package}" ]] && packages+=("${package}")
  done < <(manifest_query packages "${method}" no_build_isolation_packages)
  if (( ${#packages[@]} > 0 )); then
    pip_install "${python_bin}" --no-build-isolation "${packages[@]}"
  fi

  packages=()
  while IFS= read -r package; do
    [[ -n "${package}" ]] && packages+=("${package}")
  done < <(manifest_query packages "${method}" post_packages)
  if (( ${#packages[@]} > 0 )); then
    pip_install "${python_bin}" "${packages[@]}"
  fi

  post_install="$(manifest_query post-install "${method}")"
  case "${post_install}" in
    "")
      ;;
    santo_path)
      "${python_bin}" - <<'PY'
import site
from pathlib import Path

site_packages = next(Path(path) for path in site.getsitepackages() if path.endswith("site-packages"))
(site_packages / "p2b_santo_compat.pth").write_text(str(site_packages / "santo") + "\n")
PY
      ;;
    spacel_single_worker)
      # Scube.align(n_threads>1) hands n_threads to differential_evolution(workers=...), whose forked pool can
      # deadlock after OpenMP or BLAS threads were used (docs/KNOWN_ISSUES.md). The search
      # runs in one process; `updating` still follows n_threads, so the result matches the parallel run.
      METHOD_CACHE_PATH="${cache_path}" "${python_bin}" - <<'PY'
import os
import site
from pathlib import Path

cache_path = Path(os.environ["METHOD_CACHE_PATH"])
cache_path.mkdir(parents=True, exist_ok=True)
status, source_path = "not_found", None
for site_dir in site.getsitepackages():
    candidate = Path(site_dir) / "SPACEL" / "Scube" / "alignment.py"
    if candidate.is_file():
        source_path = candidate
        text = candidate.read_text(encoding="utf-8")
        count = text.count("workers=n_threads,")
        if count:
            # uv may install files as links into its shared cache; replace the link so the cache stays unpatched.
            candidate.unlink()
            candidate.write_text(text.replace("workers=n_threads,", "workers=1,  # p2b: single process, see bootstrap"), encoding="utf-8")
            status = f"patched_{count}"
        else:
            status = "no_workers_argument"
        break
(cache_path / "spacel_single_worker.txt").write_text(f"{status}\n{source_path}\n", encoding="utf-8")
print(f"spacel_single_worker\t{status}\t{source_path}")
PY
      ;;
    paste_pot_compat)
      METHOD_CACHE_PATH="${cache_path}" "${python_bin}" - <<'PY'
from __future__ import annotations

import os
import site
from pathlib import Path


cache_path = Path(os.environ["METHOD_CACHE_PATH"])
cache_path.mkdir(parents=True, exist_ok=True)
report_path = cache_path / "paste_pot_compat.txt"

source_path = None
for site_dir in site.getsitepackages():
    candidate = Path(site_dir) / "paste" / "PASTE.py"
    if candidate.exists():
        source_path = candidate
        break
if source_path is None:
    raise SystemExit("Could not locate paste/PASTE.py in site-packages")
text = source_path.read_text(encoding="utf-8")

patterns = (
    "def line_search(cost, G, deltaG, Mi, cost_G):",
    "def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):",
)

patched = []
for old in patterns:
    if old in text:
        text = text.replace(old, "def line_search(cost, G, deltaG, Mi, cost_G, *args, **kwargs):")
        patched.append(old)

if patched:
    # uv may install files as links into its shared cache; replace the link so the cache stays unpatched.
    source_path.unlink()
    source_path.write_text(text, encoding="utf-8")
    status = "patched"
elif "*args, **kwargs" in text and "def line_search(cost, G, deltaG, Mi, cost_G" in text:
    status = "already_patched"
else:
    status = "no_matching_line_search_pattern"

report_path.write_text(f"{status}\n{source_path}\n" + "\n".join(patched) + "\n", encoding="utf-8")
print(f"paste_pot_compat\t{status}\t{source_path}")
PY
      ;;
    opencv_headless)
      # opencv-python links libGL, which the image lacks, so `import cv2` (and with it spateo) fails; the headless
      # build of the same version has the same API without GUI libraries.
      local cv_version
      cv_version="$("${python_bin}" -c 'import importlib.metadata as m; print(m.version("opencv-python"))' 2>/dev/null || true)"
      if [[ -n "${cv_version}" ]]; then
        uv pip uninstall --python "${python_bin}" opencv-python || "${python_bin}" -m pip uninstall -y opencv-python
        pip_install "${python_bin}" "opencv-python-headless==${cv_version}"
        # Both wheels own the same cv2/ files, so removing one can leave the other recorded as installed with cv2/
        # missing (seen 2026-10-23 on fresh nodes: "No module named 'cv2'" although headless was installed).
        # Verify, and relink the headless wheel if the import fails.
        local cv_status="ok"
        if ! "${python_bin}" -c "import cv2" >/dev/null 2>&1; then
          uv pip install --python "${python_bin}" --reinstall --no-deps "opencv-python-headless==${cv_version}" \
            || "${python_bin}" -m pip install --force-reinstall --no-deps "opencv-python-headless==${cv_version}"
          if "${python_bin}" -c "import cv2" >/dev/null 2>&1; then cv_status="relinked"; else cv_status="cv2_import_failed"; fi
        fi
        mkdir -p "${cache_path}" && echo "${cv_status} ${cv_version}" >> "${cache_path}/opencv_headless.txt"
        echo "opencv_headless	replaced opencv-python ${cv_version} (${cv_status})"
      else
        echo "opencv_headless	opencv-python not installed"
      fi
      ;;
    *)
      echo "Unknown post_install hook for ${method}: ${post_install}" >&2
      return 2
      ;;
  esac
  return 0
}

install_isolated_method() {
  local method="$1"
  local canonical
  canonical="$(manifest_query canonical "${method}")"
  local env_name
  if [[ ${custom_venv_name} -eq 1 ]]; then
    env_name="${venv_name}"
  else
    env_name="$(method_env_name "${canonical}")"
  fi
  local env_path="${workspace}/${env_name}"
  install_method_into_env "${canonical}" "${env_path}"
  ensure_env_aliases "${canonical}" "${env_path}"
  local record
  while IFS= read -r record; do
    [[ -n "${record}" ]] && record_env "${record}" "${env_path}"
  done < <(manifest_query records "${canonical}")
}

install_isolated_best_effort() {
  local method="$1"
  if install_isolated_method "${method}"; then
    return 0
  fi
  record_failure "${method}" "install failed"
  echo "WARNING: method install failed for ${method}; continuing because this target is best-effort." >&2
  return 0
}

probe_one() {
  local method="$1"
  local env_path="$2"
  local python_bin="${env_path}/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    printf '%s\t%s\tNO_ENV\t%s\n' "${method}" "${env_path}" "${python_bin}"
    return 0
  fi
  METHOD_TARGET="${method}" "${python_bin}" - <<'PY'
import importlib
import os

target = os.environ["METHOD_TARGET"]
checks = {
    "cast": [("CAST", "import CAST")],
    "gpsa": [("gpsa", "import gpsa")],
    "ot": [("paste+paste2", "import numpy as np, paste; from anndata import AnnData; from paste2 import PASTE2; assert hasattr(PASTE2, 'partial_pairwise_align'); a=AnnData(X=np.eye(3)); b=AnnData(X=np.eye(3)); a.obsm['spatial']=np.array([[0.,0.],[1.,0.],[0.,1.]]); b.obsm['spatial']=np.array([[0.,0.],[1.,0.],[0.,1.]]); pi,obj=paste.pairwise_align(a,b,dissimilarity='euclidean',norm=True,numItermax=2,return_obj=True,verbose=False,gpu_verbose=False); assert pi.shape==(3,3)")],
    "paste": [("paste", "import numpy as np, paste; from anndata import AnnData; a=AnnData(X=np.eye(3)); b=AnnData(X=np.eye(3)); a.obsm['spatial']=np.array([[0.,0.],[1.,0.],[0.,1.]]); b.obsm['spatial']=np.array([[0.,0.],[1.,0.],[0.,1.]]); pi,obj=paste.pairwise_align(a,b,dissimilarity='euclidean',norm=True,numItermax=2,return_obj=True,verbose=False,gpu_verbose=False); assert pi.shape==(3,3)")],
    "paste2": [("paste2", "import paste2; from paste2 import PASTE2; assert hasattr(PASTE2, 'partial_pairwise_align')")],
    "santo": [("SANTO", "import SANTO"), ("santo", "import santo")],
    "scslat": [("scSLAT", "import importlib.metadata as md; assert int(md.version('zarr').split('.')[0]) < 3; import scSLAT")],
    "slat": [("scSLAT", "import importlib.metadata as md; assert int(md.version('zarr').split('.')[0]) < 3; import scSLAT")],
    "spacel": [("SPACEL", "import SPACEL"), ("spacel", "import spacel")],
    "spateo": [("spateo", "import spateo")],
    "stair": [("STAIR", "import STAIR"), ("stair", "import stair")],
    "stalign": [("STalign.STalign", "import importlib; m = importlib.import_module('STalign.STalign'); assert hasattr(m, 'transform_points_source_to_target')")],
    "staligner": [("STAligner", "import STAligner")],
}
last = None
for module_name, code in checks[target]:
    try:
        exec(code, {})
        print(f"{target}\t{module_name}\tOK")
        break
    except Exception as exc:
        last = f"{module_name}\t{type(exc).__name__}: {str(exc)[:240]}"
else:
    print(f"{target}\tFAIL\t{last}")
PY
}

probe_methods() {
  local method env_path
  while IFS= read -r method; do
    [[ -n "${method}" ]] || continue
    env_path="${workspace}/$(method_env_name "${method}")"
    probe_one "${method}" "${env_path}"
  done < <(manifest_query probe-order)
}

write_method_readme() {
  {
    cat <<'EOF'
# Spatial Method Environments

The package prompt uses grouped method environments where compatible and keeps
conflicting paper-method stacks isolated. The source of truth is the manifest
copied to:

```bash
EOF
    printf '%s\n' "${manifest_cache_path}"
    cat <<'EOF'
```

Rule for agents:

- If you want to use a package, run it with the exact Python executable listed
  for that package in the table below.
- Do not import method packages through `./agent_python.sh` or base
  `.method-venv/bin/python` unless the table explicitly maps that method to
  `base`; the base env is for general data handling only.
- `paste` and `paste2` intentionally share the merged OT env
  `.method-venv-ot`. `slat` is an alias for the `scslat` env. Other listed
  method families remain separate because their dependency stacks are not
  reliably compatible.
- The merged OT env applies a local PASTE/POT compatibility patch when needed;
  details are written to `.method-cache/paste_pot_compat.txt`.
- Method envs are bootstrapped with common data/science packages where
  compatible (`numpy`, `scipy`, `pandas`, `anndata`, `h5py`, `scikit-learn`,
  `scikit-image`, `matplotlib`, `seaborn`), but package installs may adjust
  versions. Check `./bootstrap_method_venv.sh probe` before investing in a
  method.
- Some method imports have heavy one-time startup costs from Torch, numba/JIT,
  or compiled libraries. A quiet import for a minute or two is not by itself a
  package failure; use an explicit bounded timeout and only move on after an
  actual traceback, import error, or timeout that is too expensive for this
  task.

Manifest summary:

```text
EOF
    manifest_query manifest-summary
    cat <<'EOF'
```

Compatible H5AD inputs:

```bash
EOF
    printf '%s\n' "${compat_path}"
    cat <<'EOF'
```

Environment table:

```text
method	env	python	compat_h5ad
EOF
    cat "${env_log}" 2>/dev/null || true
    cat <<'EOF'
```

Use examples:

```bash
./.method-venv-ot/bin/python your_paste_or_paste2_script.py
# Compatibility aliases also point to the merged OT env:
./.method-venv-paste/bin/python your_paste_script.py
./.method-venv-paste2/bin/python your_paste2_script.py
./.method-venv-stalign/bin/python your_stalign_script.py
./bootstrap_method_venv.sh probe
```

If uv virtualenv creation fails and conda/mamba is installed, rerun with:

```bash
./bootstrap_method_venv.sh --conda-fallback auto --recreate <method>
```

On this machine, conda fallback only works if `conda`, `mamba`, or
`micromamba` is on PATH.
EOF
  } > "${readme_path}"
}

initialize_shared_env() {
  local base_python
  local base_stack
  base_python="$(manifest_query base python)"
  base_stack="$(manifest_query base stack)"
  create_env "${venv_path}" "${base_python:-3.11}"
  install_common_stack "${venv_path}/bin/python" "${base_stack:-old}"
}

if (( ${#methods[@]} == 0 )); then
  methods=("base")
fi

initialize_shared_env
materialize_compat_h5ad

for method in "${methods[@]}"; do
  case "${method}" in
    base)
      record_env "base" "${venv_path}"
      ;;
    probe)
      probe_methods
      ;;
    paper-common|all)
      : > "${failure_log}"
      : > "${env_log}"
      record_env "base" "${venv_path}"
      while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        install_isolated_best_effort "${target}"
      done < <(manifest_query group paper-common)
      ;;
    paper-all|all-paper)
      : > "${failure_log}"
      : > "${env_log}"
      record_env "base" "${venv_path}"
      while IFS= read -r target; do
        [[ -n "${target}" ]] || continue
        install_isolated_best_effort "${target}"
      done < <(manifest_query group paper-all)
      ;;
    *)
      if manifest_query has-method "${method}" >/dev/null 2>&1; then
        install_isolated_method "${method}"
      else
        echo "Unknown method: ${method}" >&2
        echo "Known methods:" >&2
        manifest_query known-methods >&2 || true
        usage >&2
        exit 2
      fi
      ;;
  esac
done

write_method_readme

echo "Created shared method env: ${venv_path}"
echo "Compatible h5ad files: ${compat_path}"
echo "Method env summary: ${env_log}"
echo "Method env docs: ${readme_path}"
