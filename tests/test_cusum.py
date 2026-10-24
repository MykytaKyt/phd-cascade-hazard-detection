import numpy as np

from thesis.changepoint.cusum import (
    alarm_episodes,
    cusum_alarms,
    evaluate_sequence,
    select_h,
    threshold_alarms,
)


def test_cusum_accumulates_and_restarts():
    z = np.array([-1, 1, 1, 1, -5, 2, 2])
    np.testing.assert_array_equal(cusum_alarms(z, h=2.5), [3, 6])
    np.testing.assert_array_equal(cusum_alarms(z, h=2.5, kappa=1.0), [])


def test_threshold_alarms_are_all_frames_above_h():
    z = np.array([0, 3, 3, 0, 3])
    np.testing.assert_array_equal(threshold_alarms(z, h=2), [1, 2, 4])


def test_evaluate_sequence_fall_and_adl():
    frames = np.arange(10, 20)
    o = evaluate_sequence("fall-01", frames, np.array([1, 6]), onset=15)
    assert o.detected and o.false_alarm and o.delay == 1 and o.n_false_alarms == 1
    a = evaluate_sequence("adl-01", frames, np.array([], dtype=int), onset=None)
    assert not a.false_alarm and not a.detected and np.isnan(a.delay)


def test_select_h_prefers_no_false_alarms_with_full_detection():
    frames = np.arange(5)
    seqs = [
        ("fall-01", frames, np.array([0, 0, 2, 2, 2.0]), 2.0),
        ("adl-01", frames, np.array([0, 1.5, 0, 0, 0.0]), None),
    ]
    h = select_h(seqs, threshold_alarms, grid=np.array([1.0, 1.8, 3.0]))
    assert h == 1.8


def test_alarm_episodes_merge_within_refractory():
    np.testing.assert_array_equal(alarm_episodes(np.array([10, 40, 99, 101, 250]), 90), [10, 101, 250])
    frames = np.arange(300)
    a = evaluate_sequence("adl-01", frames, np.array([10, 40, 99, 101, 250]), onset=None, refractory=90)
    assert a.n_false_alarms == 3
