"""Multiple-testing corrections for many p-values (e.g. one per grid cell).

All functions take an array of any shape and return adjusted p-values of the
same shape. NaN p-values (empty cells) are ignored and stay NaN; they do not
count toward the number of tests m.
"""
import numpy as np


def _flatten(p):
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    return p.shape, flat, ~np.isnan(flat)


def bonferroni(p):
    """Adjusted p = min(p * m, 1)."""
    shape, flat, valid = _flatten(p)
    out = np.full(flat.shape, np.nan)
    out[valid] = np.minimum(flat[valid] * valid.sum(), 1.0)
    return out.reshape(shape)


def benjamini_hochberg(p):
    """BH adjusted p-values (q-values): reject where adjusted p <= q."""
    shape, flat, valid = _flatten(p)
    out = np.full(flat.shape, np.nan)
    pv = flat[valid]
    m = len(pv)
    if m > 0:
        order = np.argsort(pv)
        scaled = pv[order] * m / np.arange(1, m + 1)
        # running minimum from the largest p downward keeps the order intact
        adj_sorted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
        adj = np.empty(m)
        adj[order] = adj_sorted
        out[valid] = adj
    return out.reshape(shape)


def significant(p_adj, alpha=0.05):
    """True where the adjusted p-value is at most alpha. NaN gives False."""
    return np.asarray(p_adj, dtype=float) <= alpha
