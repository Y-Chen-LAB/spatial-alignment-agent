# Paper results recovered from SABench notebook outputs

These tables are not in the Nature Computational Science source-data workbooks
(MOESM6-10). They were recovered from cell outputs embedded in the public
SABench notebooks (Zenodo 10.5281/zenodo.18605715, `SABench.zip`, md5
`7a9bc923e8a20bba2eb105dc0685d52e`).

## Fig5D_AMBC_cross_platform_success_*.csv

Source: `Plot/AMBC_results.ipynb`, cell 4 printed output.
Fig. 5d of the paper: for each of 11 methods, a 10x10 platform-pair matrix
(Data231-240 adult mouse brain coronal sections) marking whether the method
ran successfully (1) or failed (0) on that cross-platform pair. Diagonal is
always 0 (not a task). 45 unordered pairs per method.

The notebook derives the matrix from the authors' private
`/SABench/AlignmentResults/*AMBC*/*.pkl` files, which are not shipped.
