import numpy as np
import pandas as pd

from thesis.evaluation import make_windows, split_validation, threshold_at_recall


def test_split_validation_is_disjoint_and_stratified():
    seqs = np.array([f"fall-{i:02d}" for i in range(20)] + [f"adl-{i:02d}" for i in range(30)])
    train, val = split_validation(np.repeat(seqs, 3), 0.2, np.random.default_rng(0))
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(seqs)
    assert sum(s.startswith("fall") for s in val) == 4
    assert sum(s.startswith("adl") for s in val) == 6


def test_threshold_at_recall_meets_target_with_best_precision():
    y = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 0.9])
    thr = threshold_at_recall(y, s, target=0.8)
    pred = s >= thr
    assert (pred & (y == 1)).sum() / (y == 1).sum() >= 0.8
    assert thr == 0.4


def test_make_windows_pads_and_aligns():
    df = pd.DataFrame({
        "seq": ["a"] * 3 + ["b"] * 2,
        "frame": [1, 2, 3, 1, 2],
        "label": [0, 0, 1, 0, 1],
        "f": [1.0, 2.0, 3.0, 10.0, 20.0],
    })
    x, y, meta = make_windows(df, window=3, feature_columns=["f"])
    assert x.shape == (5, 3, 1)
    np.testing.assert_array_equal(x[0, :, 0], [1, 1, 1])
    np.testing.assert_array_equal(x[2, :, 0], [1, 2, 3])
    np.testing.assert_array_equal(x[4, :, 0], [10, 10, 20])
    np.testing.assert_array_equal(y, [0, 0, 1, 0, 1])
    assert meta["seq"].tolist() == ["a", "a", "a", "b", "b"]
