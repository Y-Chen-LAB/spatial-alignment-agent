# Albis simulated alignment tasks: unaligned vs aligned, per model

Ten serial-section alignment tasks simulated with Albis (github.com/JanMatthias1/Albis). The true
position of every observation is known. Each task hands an agent sections that were rotated and
shifted independently. The agent must bring every section into section `s00`'s frame.

| Task | Data |
| --- | --- |
| sa-301 | 24k cells, strong domain composition |
| sa-302 | 600k cells, weak (baseline) composition |
| sa-303 | Visium-like spots, 100 µm lattice (lattice orientation leaks the rotation) |
| sa-304 | Visium HD-like 16 µm bins (same leak) |
| sa-305 | Visium HD-like 8 µm bins (same leak) |
| sa-306 | as sa-303, lattice fixed to each slide (no leak, same transforms as sa-303) |
| sa-307 | as sa-304, grid fixed to each slide (no leak, same transforms as sa-304) |
| sa-308 | as sa-301, 9 of 30 thin sections at irregular spacing (gaps of 1–10 sections) |
| sa-309 | as sa-306 with 5x fewer cells, then QC (at least 3 genes) |
| sa-310 | sa-309 with sa-308's irregular spacing |

## Figures

`<task>/<model>.jpg`, for models `claude-opus-5-5`, `gpt-6-astra`, `gpt-5.5`, `gemini-3.5-flash`
and `claude-sonnet-4-6`. Each figure shows the task's sections stacked at their true depth, in
three views drawn with `albis.plot`:

1. **unaligned (as given)**: each section where the agent received it.
2. **aligned by the model**: the model's submitted coordinates, bare condition. Where a model
   has several runs, the one with the median error is shown.
3. **true alignment**: for reference.

Colour is the true spatial domain, which the agents never see. In a correct alignment each domain
forms a continuous column through the stack. A section that is rotated wrongly shows up as a
colour break at its layer. All three views share the same axes.

Regenerate with `~/Albis/.venv/bin/python scripts/plot_albis_alignments.py`.
