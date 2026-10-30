"""The GEASO paper's reported per-method results, mapped onto our tasks.

Source: the paper's source-data workbook (Wang, Liu & Ma, Nat Commun 2026, 41467_2026_71042_MOESM4_ESM.xlsx):

    wget -O source_data.xlsx "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-71042-6/MediaObjects/41467_2026_71042_MOESM4_ESM.xlsx"
    .method-venv-spacel/bin/python scripts/extract_geaso_paper_baselines.py source_data.xlsx

Writes results/paper_baselines/geaso_2026.csv, one row per task, condition, method and metric:

    Fig.2a-b   rotation sweep, PCC / CLC / AAS by angle          -> sa-407
    Fig.2c-d   stitching MAE by overlap x angle                   -> sa-408 (20%), sa-409 (50%), sa-410 (80%)
    Fig.2e-f   3D reconstruction, MERFISH MAE per slice pair      -> sa-023 (same 129 Zhuang ABCA-1 sections)
               3D reconstruction, OpenST CLC per slice pair       -> sa-027 (OpenST lymph node, 19 sections)

The other sheets (Fig.3, 6a, 6d/S19) hold coordinates, not scores; Fig.4 (mouse embryo) and Fig.S8
(GEASO ablations) have no counterpart task. Values are copied as reported, with "N.A." kept as missing.
Stitching MAE is in the section's raw pixels; `value_um` converts it with 0.0946 µm per pixel (the
STARmap PLUS pixel size used to build sa-408 to sa-410). The paper's metrics are not our scorer's:
its MAE is the mean spot distance over the moving slice only; CLC is label consistency over a
neighbourhood; AAS is 360 minus the angle error in degrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
UM_PER_PX = 0.0946
STITCH_TASK = {"20%": "sa-408-starmapplus-sagittal3-stitch-overlap20",
               "50%": "sa-409-starmapplus-sagittal3-stitch-overlap50",
               "80%": "sa-410-starmapplus-sagittal3-stitch-overlap80"}


def main(xlsx: Path) -> None:
    x = pd.ExcelFile(xlsx)
    rows = []

    a = x.parse("Fig.2a-b&Fig.S1-2")
    for r in a.itertuples(index=False):
        rows.append(dict(task="sa-407-starmapplus-ad-rotation-sweep", paper_figure="Fig. 2a-b", condition=f"rotation {r.rotation}",
                         method=r.method, metric=r.metric, value=pd.to_numeric(r.value, errors="coerce"),
                         units="degrees (360 - angle error)" if r.metric == "AAS" else "unitless"))

    c = x.parse("Fig.2c-d&Fig.S4")
    for r in c.itertuples(index=False):
        overlap, rot = [p.strip() for p in str(getattr(r, "_0")).split(",")]
        v = pd.to_numeric(r.value, errors="coerce")
        rows.append(dict(task=STITCH_TASK[overlap], paper_figure="Fig. 2c-d", condition=f"overlap {overlap}, rotation {rot}",
                         method=r.method, metric="MAE", value=v, units="pixels", value_um=v * UM_PER_PX))

    e = x.parse("Fig.2e-f")
    pair = e.groupby(["dataset", "method"]).cumcount() + 1
    for r, k in zip(e.itertuples(index=False), pair):
        if r.dataset == "MERFISH":
            rows.append(dict(task="sa-023-wmb-zhuang-abca1-129-section-3d-mae", paper_figure="Fig. 2e",
                             condition=f"pair {k}", method=r.method, metric="MAE",
                             value=pd.to_numeric(r.value, errors="coerce"), units="as reported (coordinate units of the released data)"))
        else:
            rows.append(dict(task="sa-027-openst-mln-full-19-slice-3d-align", paper_figure="Fig. 2f",
                             condition=f"pair {k}", method=r.method, metric="CLC",
                             value=pd.to_numeric(r.value, errors="coerce"), units="unitless"))

    out = REPO / "results/paper_baselines/geaso_2026.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "source", "Wang, Liu & Ma 2026 Nat Commun (GEASO), source data")
    df.to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
