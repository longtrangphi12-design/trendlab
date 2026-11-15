"""Hamed-Rao modified Mann-Kendall, Pettitt change-point test, Benjamini-Yekutieli."""
import numpy as np
from scipy import stats

from trendlab.multitest import benjamini_hochberg
from trendlab.trend import mann_kendall, sen_slope


def hamed_rao(x, alpha=0.05) -> dict:
    """Mann-Kendall with the Hamed-Rao (1998) variance correction.

    1. remove the Sen trend, take the RANKS of the residual;
    2. estimate the autocorrelation rho_k of those ranks;
    3. keep only lags where |rho_k| is itself significant;
    4. Var*(S) = Var(S) * (1 + 2/(n(n-1)(n-2)) * sum (n-k)(n-k-1)(n-k-2) rho_k).

    Returns s, var_s (corrected), z, p, n, factor (= n/n*; >1 means plain MK was
    overconfident), fallback. If factor <= 0 (strong negative autocorrelation) the
    plain MK result is returned with fallback=True, so no cell is silently dropped.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    base = mann_kendall(x)
    if n < 4 or not np.isfinite(base["var_s"]) or base["var_s"] == 0:
        return {**base, "factor": np.nan, "fallback": True}

    t = np.arange(1, n + 1, dtype=float)
    resid = x - sen_slope(x, t) * t
    ranks = stats.rankdata(resid)
    d = ranks - ranks.mean()
    denom = (d * d).sum()
    bound = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)

    total = 0.0
    for k in range(1, n):
        rho = (d[:-k] * d[k:]).sum() / denom
        if abs(rho) > bound:
            total += (n - k) * (n - k - 1) * (n - k - 2) * rho
    factor = 1 + 2 / (n * (n - 1) * (n - 2)) * total
    if factor <= 0:
        return {**base, "factor": float(factor), "fallback": True}

    var_s = base["var_s"] * factor
    s = base["s"]
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    return {"s": s, "var_s": float(var_s), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z))), "n": n, "factor": float(factor),
            "fallback": False}


def pettitt(x) -> dict:
    """Pettitt (1979) test for ONE abrupt shift.

    U_t = sum_{i<=t} sum_{j>t} sign(x_j - x_i); K = max |U_t|;
    p ~= 2 exp(-6 K^2 / (n^3 + n^2)), capped at 1.
    change_index (0-based) = last point BEFORE the shift, in the NaN-free series.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return {"k": np.nan, "change_index": -1, "p": np.nan, "n": n}
    sgn = np.sign(x[None, :] - x[:, None])
    u = np.array([sgn[: t + 1, t + 1:].sum() for t in range(n - 1)])
    k = float(np.abs(u).max())
    idx = int(np.abs(u).argmax())
    p = min(1.0, float(2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))
    return {"k": k, "change_index": idx, "p": p, "n": n}


def benjamini_yekutieli(p):
    """BY adjusted p = BH times c(m) = sum_{i=1..m} 1/i, capped at 1.

    Valid under ANY dependence between tests (BH needs independence or positive
    dependence), at the price of being more conservative. NaN p-values are ignored.
    """
    p = np.asarray(p, dtype=float)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return np.full(p.shape, np.nan)
    c_m = (1.0 / np.arange(1, m + 1)).sum()
    return np.minimum(benjamini_hochberg(p) * c_m, 1.0)