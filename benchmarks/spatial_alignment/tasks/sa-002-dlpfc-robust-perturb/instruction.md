# DLPFC Robust Alignment

Align a DLPFC reference slice and a perturbed moving slice.

Submit coordinate CSVs for the reference and moving slices with columns `obs_id,x,y`.

## Public Pair

The public runnable task uses the official SABench robustness recipe on the DLPFC pair `151673` (reference) and `151674` (moving):

- moving slice rotated by `90` degrees around its center
- then cropped by retaining the left `50%` x-axis region

The perturbed moving slice is supplied in `data/`; you do not need to reproduce it.

## Hidden Robustness Grid

The paper's robustness evaluation does not rely on a single perturbation. It evaluates two perturbation families generated from the same DLPFC pair:

- rotation: `0` to `360` degrees in `15`-degree steps
- crop overlap: `10%` to `100%` in `10%` steps, keeping the left tissue region

Those official conditions are withheld from this workspace.
