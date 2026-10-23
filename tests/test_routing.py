import numpy as np

from thesis.routing.calibration import calibrate, ece, fit_temperature
from thesis.routing.conformal import crc_threshold, empirical_threshold, min_calibration_sequences
from thesis.routing.region import binary_entropy, inverse_entropy, region_stats


def test_temperature_recovers_scaled_logits():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z))).astype(float)
    t = fit_temperature(3.0 * z, y)
    assert abs(t - 3.0) < 0.15
    assert ece(calibrate(3.0 * z, t), y) < 0.02


def test_inverse_entropy_matches_published_value():
    eps = inverse_entropy(0.6)
    assert abs(eps - 0.146) < 1e-3
    assert abs(binary_entropy(np.array([eps]))[0] - 0.6) < 1e-9


def test_region_stats_counts():
    p = np.array([0.01, 0.1, 0.3, 0.6, 0.9, 0.99])
    y = np.array([1, 1, 0, 1, 0, 1])
    s = region_stats(p, y, tau_n=0.05, tau_a=0.5)
    assert s["n_routed"] == 2          # 0.1 and 0.3
    assert s["f_edge"] == 1            # the positive at 0.01
    assert s["recall_bound"] == 0.75
    assert s["edge_fp"] == 1           # 0.9 is a negative classified as fall


def test_crc_needs_enough_sequences():
    assert min_calibration_sequences(0.05) == 19
    few = [np.array([0.2, 0.8])] * 5
    assert crc_threshold(few, 0.05, np.linspace(0, 0.5, 51)) is None
    many = [np.array([0.2, 0.8])] * 40
    tau = crc_threshold(many, 0.05, np.linspace(0, 0.5, 51))
    assert tau is not None and tau < 0.2


def test_empirical_threshold():
    probs = np.array([0.02, 0.1, 0.4, 0.7, 0.9])
    y = np.ones(5)
    assert empirical_threshold(probs, y, 0.8, np.array([0.01, 0.05, 0.2])) == 0.05
