# Paper Method Priors

This note summarizes method-priority signals extracted from the local SABench
source-data CSVs under `references/paper_source_data/s43588-026-00977/`.
It is intended for prompt design, not for task scoring. The values below are
paper-side method-performance priors; agents still receive only sanitized task
inputs and must not access paper result tables during blind solves.

## Aggregate Paper Rankings

Lower rank is better in `MOESM9` rank tables.

| Context | Strongest paper methods |
|---|---|
| Image datasets, accuracy | `spateo` non-rigid, `SPACEL`, `spateo` rigid, `STAIR`, `CAST`, `PASTE2` |
| Image datasets, scalability | `spateo` rigid/non-rigid, then `PASTE`, `SLAT`, `SPACEL` |
| NGS datasets, accuracy | `PASTE2`, `SPACEL`, `spateo` non-rigid, `spateo` rigid, `STAIR` |
| NGS datasets, scalability | `PASTE`, `SLAT`, `PASTE_p0`, then `SPACEL` / `spateo` |

## Direct Dataset Priors

These come from direct paper-side DLPFC/MHPR metric tables in `MOESM7`.

| Dataset family | Strongest paper methods |
|---|---|
| DLPFC gene + landmark combined | `SPACEL`, `PASTE2`, `CAST`, `STAIR`, `spateo` |
| DLPFC robustness | `SPACEL` is strongest across rotation/crop; `spateo`, `PASTE2`, and `STAIR` are also strong |
| MHPR gene + landmark combined | `spateo` rigid/non-rigid, `SPACEL`, `PASTE2`, `STAligner`, `SLAT`, `STAIR` |
| MHPR robustness | `SPACEL`, `spateo`, `STAIR`, `PASTE2`; `CAST` is also competitive on rotation |

## 3D / MAE Tables

Lower MAE is better in `MOESM10`.

| Dataset family | Strongest paper methods |
|---|---|
| Flysta3D MAE | `SPACEL`, `PASTE`, `PASTE_p0`, `STAIR`, `PASTE2`, `spateo` |
| MLN MAE | `spateo` rigid, `SLAT`, `spateo` non-rigid, `STalign`, `SPACEL` |

## Prompt Implications

- For DLPFC-like Visium serial alignment, nudge agents toward `SPACEL` and
  `PASTE2` first; `CAST`, `STAIR`, and `spateo` are reasonable second-wave
  attempts.
- For MHPR/MERFISH or other image-like serial alignment, nudge agents toward
  `spateo` and `SPACEL` first; `PASTE2`, `STAIR`, and `SLAT` are useful
  backups.
- For generic NGS/Visium/STereo-seq tasks, nudge agents toward `PASTE2`,
  `SPACEL`, and `spateo`, with `PASTE` kept as a faster fallback.
- For 3D serial/MAE-like tasks, nudge agents toward `SPACEL` first, then
  `PASTE`/`PASTE2` or `spateo` depending on task size and API availability.
- Do not hide the other methods. The prompt should still allow `STAligner`,
  `GPSA`, `STalign`, `SANTO`, `CAST`, `STAIR`, and `SLAT` when they fit the
  data or when higher-priority packages fail.
