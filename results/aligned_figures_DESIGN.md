# Aligned-result figures: what the benchmarks plot, and the layout we use

**SABench** (references/sabench_notebooks_py/Plot, Tutorial). It plots aligned slices as 2D overlays of one pair, one figure per method. BCA_plot.py draws the reference in blue and the moving section in red. AMBC_plot.py lays the sections out as small multiples coloured by total counts. Accuracy goes into separate plots: boxplots per method (MOB_boxplot.py), robustness grids over rotation and crop levels, and radar and bubble charts (robustness.py). It has no 3D views and puts no scores on the overlays.

**GEASO** (geaso/repo/Tutorials, GEASO/plot_utils). It shows `sc.pl.embedding` overlays of the concatenated pair, coloured once by batch (section) and once by cell type or layer. It repeats this at each stage: raw, then coarse, then non-rigid. That makes a before/after strip. For 3D reconstructions (OpenST) it draws an oblique pyvista point cloud coloured by cell type. `Build3D` draws two layers with matching lines between them. Accuracy goes into Sankey diagrams and barplots per method (`catplot`, with GEASO, Spateo, CAST, SLAT and others side by side).

**Ours so far.** albis_figures uses `albis.plot` 3D stacks: unaligned, then model, then truth, coloured by true domain. dlpfc_figures uses 2D overlays coloured by layer, with the reference as faint large discs and the moving section as small dots. For sa-001, which has no coordinate truth, it shows layer agreement: grey where the label matches the nearest neighbour in the previous section, red where it does not.

**Chosen layout** (scripts/plot_aligned_results.py):
- **One grid per task and pair.** It is a 2D pair overlay in the SABench and GEASO convention.
  - Row 1 is the input (before), the ground truth (after, the target), the best published method, the best scripted method, and PASTE2, Spateo, SPACEL and GEASO where they exist.
  - Rows 2 and 3 are the best and the median bare run of each of the 6 models.
- **Panel style.** Every panel has the same encoding, its own score in the title (MAE, else matching accuracy, else the gene mean), and the same colours.
  - With labels, both sections are coloured by region (GEASO style), the reference as faint discs and the moving section as dots.
  - Without labels, the reference is blue and the moving section orange (the SABench BCA style).
- **Label agreement.** Tasks with labels but no coordinate truth also get an agreement grid in the DLPFC/sa-001 style. It is the label counterpart of SABench's layer-matching accuracy.
- **Stacks of 5 or more sections.** Up to 3 representative pairs (first, middle, last), plus one oblique 3D stack grid in the GEASO OpenST style, with the same rows and columns. Sweep ("star") tasks show 3 representative perturbation levels instead.

**Why this layout.** Reviewers in this area expect 2D overlays (both papers) and 3D stacks for reconstruction (GEASO). Putting agents and published baselines in one grid with the same encoding and scores on the panels gives the method-by-method comparison that both papers otherwise split across separate figures.
