"""Minimal AnnData stand-in so the SABench authors' notebook code can run unmodified."""
import h5py, numpy as np, pandas as pd
from scipy import sparse

def _read_str_array(node):
    if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") == "categorical":
        cats = np.asarray(node["categories"][()]).astype(str)
        codes = np.asarray(node["codes"][()])
        out = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], "NA")
        return out.astype(object)
    v = np.asarray(node[()])
    return np.array([x.decode() if isinstance(x, bytes) else x for x in v], dtype=object)

class Ad:
    def __init__(self, X, obs, obsm, var_names):
        self.X, self.obs, self.obsm, self.var_names = X, obs, obsm, var_names
    def copy(self):
        return Ad(self.X, self.obs.copy(), {k: v.copy() for k, v in self.obsm.items()}, self.var_names)
    def __getitem__(self, key):
        mask = key[0] if isinstance(key, tuple) else key
        mask = np.asarray(mask)
        return Ad(self.X[mask], self.obs.iloc[mask].reset_index(drop=True),
                  {k: v[mask] for k, v in self.obsm.items()}, self.var_names)

def read(path, obs_cols=("Region", "Cluster")):
    with h5py.File(path, "r") as h:
        n = h["X"].attrs["shape"][0] if isinstance(h["X"], h5py.Group) else h["X"].shape[0]
        node = h["X"]
        if isinstance(node, h5py.Group):
            enc = node.attrs.get("encoding-type")
            shape = tuple(int(x) for x in node.attrs["shape"])
            M = sparse.csr_matrix if enc == "csr_matrix" else sparse.csc_matrix
            X = M((node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape).tocsr()
        else:
            X = sparse.csr_matrix(np.asarray(node[()]))
        var_names = pd.Index(_read_str_array(h["var"][h["var"].attrs["_index"]]))
        obs = {}
        for c in obs_cols:
            if c in h["obs"]:
                obs[c] = _read_str_array(h["obs"][c])
        obsm = {"spatial": np.asarray(h["obsm"]["spatial"][()], dtype=float)}
    return Ad(X, pd.DataFrame(obs), obsm, var_names)
