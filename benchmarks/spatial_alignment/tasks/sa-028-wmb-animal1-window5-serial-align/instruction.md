# WMB MERFISH Partial Animal-1 Alignment

This task is a partial paper2bench reconstruction of the SABench WMB family using the single locally downloaded animal file.

Included contiguous source sections: C57BL6J-1.071, C57BL6J-1.081, C57BL6J-1.073, C57BL6J-1.083, C57BL6J-1.046

The compiled per-section files expose:
- `obsm/spatial`: the observed section coordinates
- `obs/Region`: a coarse region label mapped from `major_brain_region`
- `obs/ccf_region_name`: fine-grained atlas annotation retained for analysis
- `obs/brain_section_label`: original section identifier

## Task

Align the provided sections into a shared coordinate frame. Submit one coordinate CSV per section with columns `obs_id,x,y`.

## Scoring

`metric_profile = "sabench_light_alignment_v1"`

This is intentionally a starter subset, not a full reproduction of SABench Data99-227. It gives us a runnable WMB-family benchmark from the partial local corpus while we continue fetching the remaining paper slices.
