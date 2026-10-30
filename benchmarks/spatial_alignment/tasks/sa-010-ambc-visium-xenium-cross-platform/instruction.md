# AMBC Visium ↔ Xenium V1 Cross-Platform Alignment (Data231 ↔ Data234)

A cross-platform pair from the paper's Data231-240 ten-platform Adult Mouse Brain Coronal family (Yan et al. 2026). One slice is 10x Visium 55-µm-spot capture (Data231, ~2.7k spots, 32k genes); the other is 10x Xenium V1 single-cell imaging (Data234 multisection 1, ~162k cells, 541 panel genes). Slices target the same anatomy but at very different resolutions and gene panels.

The paper's Data231-240 task framing: "compiled from publicly available sources, focusing specifically on mouse brain coronal sections... ten slices generated using a variety of ST platforms... all slices exhibit clear and consistent biological structures that were able to be aligned" (PAPER_MANUSCRIPT.md, Methods).

## Task

Align the two slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- Marker gene list is the top-100 highly variable genes intersected across the two slices (intersection ≈ Xenium panel ⊆ Visium full transcriptome).
- Neither input ships cluster annotations from the raw 10x outs, so SABench landmark/region metrics return NaN; gene-grid metrics (PCC/Cos_Sim/SSIM/MI) score the alignment.
- The paper applies edge trimming + right-hemisphere selection for Data231-240 to make the alignment well-posed. The current task uses the full-tissue compiled assets; expect a difficulty premium until the right-hemisphere preprocessing step is added.
