from trendlab.autocorr import autocorr_flag, autocorr_map
from trendlab.simulate import autocorr_null_threshold, null_grid


def test_threshold_is_reproducible():
    a = autocorr_null_threshold(24, n_sim=500, seed=1)
    b = autocorr_null_threshold(24, n_sim=500, seed=1)
    assert a == b


def test_higher_quantile_gives_higher_threshold():
    lo = autocorr_null_threshold(24, q=0.90, n_sim=1000, seed=2)
    hi = autocorr_null_threshold(24, q=0.99, n_sim=1000, seed=2)
    assert hi > lo


def test_longer_series_gives_smaller_threshold():
    short = autocorr_null_threshold(24, n_sim=1000, seed=3)
    long_ = autocorr_null_threshold(200, n_sim=1000, seed=3)
    assert long_ < short


def test_false_alarm_rate_matches_design_on_independent_grid():
    thr = autocorr_null_threshold(24, q=0.95, n_sim=3000, seed=4)
    g = null_grid(n_time=24, n_lat=40, n_lon=50, phi=0.0, seed=7)
    frac = float(autocorr_flag(autocorr_map(g), thr).mean())
    assert 0.03 < frac < 0.07
