# Data release and Zenodo upload

## Current status

The repository commits the 121 main task packages: specifications, instructions, manifests
and small marker tables. The 12 Albis smoke variants have been removed. Compiled `.h5ad`
inputs, raw archives and large run outputs remain outside Git.

On JHPCE the data staging directory is:

```
/dcs07/chen/data/alignment-agent/zenodo_release
```

Its README reports about 34 GB of task inputs and 3.1 GB of raw archives. It contains
`task_inputs/`, `raw/`, `SOURCES.tsv` and `MANIFEST.md5`; `results/` is empty.
This is staging, not evidence of a published deposit. No release DOI or download URL is
recorded in this checkout. The staging tree may still contain smoke inputs; those are not
part of the 121-task catalog.

Raw source accessions and known provenance gaps are in [DATA_SOURCES.md](DATA_SOURCES.md).
Dataset licenses in the staged `SOURCES.tsv` are still TODO. Record the redistribution
terms before publishing those files; upstream raw data can remain linked by accession
when redistribution is unsuitable. Deposit the exact compiled inputs needed to run the
cards, along with checksums and provenance. A repository of cards plus upstream raw-data
links alone would still require readers to reproduce all preprocessing.

Before declaring the release complete: record licenses and missing source URLs, fix the
sa-414 manifest provenance noted in DATA_SOURCES.md, choose which results to archive,
verify checksums, publish the data record, then add its version-specific DOI and download
links here and in README.md. This checkout also has no top-level LICENSE or CITATION.cff;
the project owners should choose a code license and citation metadata.

## Commands from this cluster

The following prepares and uploads a draft; it does not publish it. Requires `curl`,
`jq`, `tar` and a Zenodo account. Create a token with `deposit:write` at
https://zenodo.org/account/settings/applications/tokens/new/ .
Use a separate sandbox account/token and `https://sandbox.zenodo.org` to rehearse.

Choose an output directory with at least 40 GB free, outside the repository:

```bash
export ZENODO_STAGE=/dcs07/chen/data/alignment-agent/zenodo_release
export ZENODO_BUNDLE_DIR=/path/to/large/scratch/zenodo-upload
mkdir -p "$ZENODO_BUNDLE_DIR"
(cd "$ZENODO_STAGE" && md5sum -c MANIFEST.md5)
# Include raw only after checking redistribution terms in SOURCES.tsv.
tar -C "$ZENODO_STAGE" -cf "$ZENODO_BUNDLE_DIR/spatial-alignment-data.tar" \
  task_inputs raw SOURCES.tsv MANIFEST.md5 README.md
(cd "$ZENODO_BUNDLE_DIR" && sha256sum spatial-alignment-data.tar > SHA256SUMS)
```

Create one draft and keep its response so an interrupted upload can reuse the same draft:

```bash
read -rsp 'Zenodo token: ' ZENODO_TOKEN; echo
export ZENODO_TOKEN
export ZENODO_URL=https://zenodo.org
curl --fail-with-body --show-error \
  -H "Authorization: Bearer $ZENODO_TOKEN" \
  -H 'User-Agent: spatial-alignment-agent-data-release/1.0' \
  -H 'Content-Type: application/json' -d '{}' \
  "$ZENODO_URL/api/deposit/depositions" > "$ZENODO_BUNDLE_DIR/deposition.json"
jq -er '.id, .links.html' "$ZENODO_BUNDLE_DIR/deposition.json"
BUCKET_URL=$(jq -er '.links.bucket' "$ZENODO_BUNDLE_DIR/deposition.json")
```

Stream each file to the bucket endpoint (the old multipart upload endpoint is unsuitable
for these sizes). A retry restarts the file; these commands do not provide byte resume.

```bash
for NAME in spatial-alignment-data.tar SHA256SUMS; do
  curl --fail-with-body --show-error --retry 5 --retry-delay 10 \
    --upload-file "$ZENODO_BUNDLE_DIR/$NAME" \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    -H 'User-Agent: spatial-alignment-agent-data-release/1.0' \
    "$BUCKET_URL/$NAME" > "$ZENODO_BUNDLE_DIR/$NAME.upload.json" || break
  jq '{key, size, checksum}' "$ZENODO_BUNDLE_DIR/$NAME.upload.json"
done
unset ZENODO_TOKEN
```

Compare Zenodo's returned MD5 with `md5sum` of each local uploaded file. Open the draft URL
printed above, fill title/authors/description/licenses and related GitHub URL, review the
files, and publish when ready. Add the published record DOI to the repository.
Keep baseline outputs or other results in a second record if the combined size exceeds
the standard quota (50 GB total, up to 100 files; ask Zenodo for a quota increase if needed).

API reference: https://developers.zenodo.org/#quickstart-upload
