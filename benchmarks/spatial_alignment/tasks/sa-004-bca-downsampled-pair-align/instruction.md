# Breast Cancer Downsampled Pair Alignment

Align the two downsampled breast cancer spatial slices. This is a resource-aware alignment task intended to precede full-resolution recovery.

Submit one coordinate CSV per slice with columns `obs_id,x,y`.

The first scoring pass checks output schema, coordinate validity, geometry overlap, and runtime/memory. Label-aware scoring is enabled if compatible cluster labels are available.
