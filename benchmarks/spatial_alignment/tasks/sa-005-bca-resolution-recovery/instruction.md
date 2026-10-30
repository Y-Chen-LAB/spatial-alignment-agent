# Breast Cancer Resolution Recovery

Two serial Xenium sections of the same breast cancer block, `slice_Rep1` and `slice_Rep2`, at full
single-cell resolution. Align `slice_Rep2` onto `slice_Rep1` and submit aligned coordinates for every
cell of both full-resolution slices.

You are also given lower-resolution versions of the same two slices, and an existing alignment of
those lower-resolution versions:

- `slice_Rep1_DR`, `slice_Rep2_DR`: each full-resolution slice aggregated onto a 425 x 425 grid, in
  the same coordinate frame as its full-resolution slice.
- `STAligner_BCA_new_slices_Rep1`, `STAligner_BCA_new_slices_Rep2`: the same grid-level slices, row
  for row, with coordinates after alignment by STAligner (a published method). Rep1 is the reference.

A natural approach is to infer the alignment transform from the lower-resolution data (for example
from the STAligner result, or from your own alignment of the grid-level slices) and then apply it to
the full-resolution coordinates. You may also align the full-resolution slices directly, or refine
the STAligner result. The given alignment is not guaranteed to be good.

Keep `slice_Rep1` in its own coordinate frame. Submit full-resolution coordinate CSVs for
`slice_Rep1` and `slice_Rep2` only, with columns `obs_id,x,y`; the lower-resolution files are inputs,
not outputs.

The score compares the aligned full-resolution slices using held-out tissue-region structure.
