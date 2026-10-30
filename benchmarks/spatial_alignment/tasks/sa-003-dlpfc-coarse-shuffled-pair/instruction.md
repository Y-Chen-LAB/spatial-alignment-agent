# DLPFC Coarse Pair Alignment

Align the two shuffled DLPFC slices into a shared coordinate frame. This task emphasizes coarse pre-alignment rather than fine-grained serial reconstruction.

Submit one coordinate CSV per slice with columns `obs_id,x,y`.

The benchmark scores gene-grid similarity, landmark matching by `Region`, region-grid overlap, runtime, and memory.
