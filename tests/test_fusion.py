import itertools

import numpy as np
import pytest

from thesis.fusion.dst import (
    argmax_decision,
    conjunctive,
    dempster,
    pignistic,
    simple_bba,
    yager,
)


def random_bba(frame, rng, n_focal=4):
    subsets = [frozenset(s) for r in range(1, len(frame) + 1) for s in itertools.combinations(frame, r)]
    picks = rng.choice(len(subsets), size=min(n_focal, len(subsets)), replace=False)
    w = rng.dirichlet(np.ones(len(picks)))
    return {subsets[i]: float(v) for i, v in zip(picks, w)}


def test_binary_matches_closed_form():
    frame = ["A", "N"]
    m1 = {frozenset("A"): 0.6, frozenset("N"): 0.1, frozenset(frame): 0.3}
    m2 = {frozenset("A"): 0.2, frozenset("N"): 0.5, frozenset(frame): 0.3}
    k = 0.6 * 0.5 + 0.1 * 0.2
    md, kd = dempster(m1, m2)
    assert kd == pytest.approx(k)
    assert md[frozenset("A")] == pytest.approx((0.6 * 0.2 + 0.6 * 0.3 + 0.3 * 0.2) / (1 - k))
    my, _ = yager(m1, m2, frame=frame)
    assert my[frozenset(frame)] == pytest.approx(0.3 * 0.3 + k)
    assert sum(my.values()) == pytest.approx(1.0)


def test_simple_bba_and_pignistic():
    frame = ["fall", "fight", "normal"]
    m = simple_bba({"fall": 0.5, "fight": 0.2, "normal": 0.3}, 0.4, frame)
    betp = pignistic(m, frame)
    assert betp["fall"] == pytest.approx(0.6 * 0.5 + 0.4 / 3)
    assert sum(betp.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("k_classes", [2, 3, 4, 5])
def test_argmax_decisions_of_dempster_and_yager_coincide(k_classes):
    """Statement G1: one-step combination, any BBAs, any finite frame."""
    rng = np.random.default_rng(k_classes)
    frame = [f"c{i}" for i in range(k_classes)]
    for _ in range(300):
        bbas = [random_bba(frame, rng) for _ in range(rng.integers(2, 4))]
        _, k = conjunctive(*bbas)
        if k > 0.999:
            continue
        bd = pignistic(dempster(*bbas)[0], frame)
        by = pignistic(yager(*bbas, frame=frame)[0], frame)
        top = max(bd.values())
        if sum(abs(v - top) < 1e-12 for v in bd.values()) > 1:
            continue  # ties
        assert argmax_decision(bd) == argmax_decision(by)


def test_threshold_decisions_differ_unless_beta_is_uniform():
    """Statement G2: BetP >= beta decisions coincide for all inputs only at beta = 1/|Theta|;
    for beta < 1/|Theta| Yager's rule alarms whenever Dempster's does."""
    frame = ["A", "N"]
    rng = np.random.default_rng(0)
    disagreements = {0.5: 0, 0.2: 0}
    yager_more_permissive = True
    for _ in range(2000):
        m1, m2 = random_bba(frame, rng, 3), random_bba(frame, rng, 3)
        _, k = conjunctive(m1, m2)
        if k > 0.999:
            continue
        bd = pignistic(dempster(m1, m2)[0], frame)["A"]
        by = pignistic(yager(m1, m2, frame=frame)[0], frame)["A"]
        for beta in disagreements:
            if (bd >= beta) != (by >= beta):
                disagreements[beta] += 1
                if beta == 0.2 and bd >= beta and by < beta:
                    yager_more_permissive = False
    assert disagreements[0.5] == 0
    assert disagreements[0.2] > 0
    assert yager_more_permissive


def test_hierarchical_threshold_coincides_at_set_share():
    """Statement G2 for a set of hypotheses B: coincidence at beta = |B| / |Theta|."""
    frame = ["a1", "a2", "a3", "n"]
    hazard = ["a1", "a2", "a3"]
    rng = np.random.default_rng(1)
    for _ in range(1000):
        m1, m2 = random_bba(frame, rng, 5), random_bba(frame, rng, 5)
        _, k = conjunctive(m1, m2)
        if k > 0.999:
            continue
        bd = sum(pignistic(dempster(m1, m2)[0], frame)[c] for c in hazard)
        by = sum(pignistic(yager(m1, m2, frame=frame)[0], frame)[c] for c in hazard)
        assert (bd >= 0.75) == (by >= 0.75) or abs(bd - 0.75) < 1e-9 or abs(by - 0.75) < 1e-9
        if bd >= 0.5:
            assert by >= 0.5 - 1e-12
