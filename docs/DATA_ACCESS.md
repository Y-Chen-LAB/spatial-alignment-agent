# Data access

Associated source record: [Zenodo, doi:10.5281/zenodo.24043338](https://doi.org/10.5281/zenodo.24043338).

Task specifications, instructions, manifests and marker tables are versioned in this
repository. Large compiled `.h5ad` inputs are maintained separately. Dataset accessions,
preprocessing scripts and provenance limitations are documented in [DATA_SOURCES.md](DATA_SOURCES.md).

## Local configuration

The task-input layout is:

```
task_inputs/
  paper2bench-data/
  compiled_data/
```

After obtaining the compiled task inputs, run these commands from the repository root,
substituting the location of your extracted data:

```bash
python scripts/check_portability.py --data-root /path/to/task_inputs/paper2bench-data --apply
ln -sfn /path/to/task_inputs/compiled_data benchmarks/spatial_alignment/compiled_data
```

The first command updates the absolute slice paths in task specifications. The second
connects tasks that use repository-relative compiled-data paths. The original data manifests
may retain the paths recorded when the inputs were prepared.

Archive contents and direct download links will be documented when the record is publicly accessible.
