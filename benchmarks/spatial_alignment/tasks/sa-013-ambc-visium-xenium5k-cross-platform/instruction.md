# AMBC Visium ↔ Xenium Prime 5k Cross-Platform Alignment (Data231 ↔ Data235)

A cross-platform pair from the SABench Data231-240 adult mouse brain coronal family. One slice is 10x Visium (`Data231`), and the other is 10x Xenium Prime 5k (`Data235`) compiled from the local public `outs.zip` bundle.

## Task

Align the two slices into a shared coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- This task is intentionally model-free: it specifies only the data contract and the deterministic scorer.
- The paper's Data231-240 family uses right-hemisphere and quality-controlled standardized AnnData slices. The current local task uses the directly compiled public assets available in this workspace; it is aligned to the paper family, but not yet a byte-for-byte reconstruction of the paper's trimmed release.
- The raw Xenium Prime bundle does not ship paper2bench-ready region labels, so this task uses the honest SABench gene-only metric profile.
