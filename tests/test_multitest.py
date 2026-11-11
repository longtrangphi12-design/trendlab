import numpy as np
import pytest

from trendlab.multitest import benjamini_hochberg, bonferroni, significant


def test_bonferroni_hand_calculated():
    out = bonferroni([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(out, [0.04, 0.16, 0.12, 0.02])


def test_bh_hand_calculated():
    out = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(out, [0.02, 0.04, 0.04, 0.02])


def test_bh_enforces_monotonicity():
    out = benjamini_hochberg([0.03, 0.001, 0.031])
    np.testing.assert_allclose(out, [0.031, 0.003, 0.031])


def test_nan_is_ignored_and_preserved():
    bh = benjamini_hochberg([0.01, np.nan, 0.04])
    bf = bonferroni([0.01, np.nan, 0.04])
    np.testing.assert_allclose(bh, [0.02, np.nan, 0.04])
    np.testing.assert_allclose(bf, [0.02, np.nan, 0.08])


def test_shape_is_preserved():
    p = np.array([[0.01, 0.5], [0.2, 0.03]])
    assert benjamini_hochberg(p).shape == (2, 2)
    assert bonferroni(p).shape == (2, 2)


def test_ordering_properties():
    p = np.random.default_rng(0).uniform(size=1000)
    bh, bf = benjamini_hochberg(p), bonferroni(p)
    assert np.all(bh >= p - 1e-12)          # adjusting never lowers a p-value
    assert np.all(bh <= bf + 1e-12)         # BH is never stricter than Bonferroni
    i, j = np.argsort(p)[:2]
    assert bh[i] <= bh[j]                   # order of p-values is kept


def test_null_data_many_raw_hits_few_adjusted():
    # 1000 cells with NO real trend: p-values are uniform by construction
    p = np.random.default_rng(1).uniform(size=1000)
    raw_hits = significant(p).sum()
    bh_hits = significant(benjamini_hochberg(p)).sum()
    assert raw_hits > 10          # about 50 expected, purely by chance
    assert bh_hits <= raw_hits


def test_significant_helper():
    assert list(significant([0.01, 0.2, np.nan], alpha=0.05)) == [True, False, False]
