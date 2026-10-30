# AMBC Visium ↔ STARmap+ Cross-Platform Alignment (Data231 ↔ Data237)

This cross-platform pair comes from the SABench adult mouse brain coronal family. `Data231` is 10x Visium; `Data237` is STARmap+ well11 with preserved tissue-region annotations.

## Task

Align the two slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- The STARmap+ slice preserves `obs/Region`, `obs/Sub_region`, and `obs/Cell_type` from the public Zenodo CSVs via `scripts/compile_starmap_plus_well11.py`.
- STARmap+ raw gene symbols were normalized from all-caps to standard mouse-style case during compilation so cross-platform marker-gene overlap with other mouse AMBC slices is meaningful.
- Even though `Data237` is labeled, the local Visium partner does not ship corresponding region labels, so this task currently runs under the honest SABench gene-only profile rather than claiming landmark metrics it cannot support.
- This still gives a useful AMBC cross-platform benchmark and keeps the labeled STARmap+ asset ready for future pairings if a second labeled AMBC modality becomes available locally.
