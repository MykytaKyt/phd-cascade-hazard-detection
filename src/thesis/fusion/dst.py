"""Dempster-Shafer evidence combination on a finite frame of discernment.

A basic belief assignment (BBA) is a dict {frozenset: mass} over non-empty
subsets of the frame, with masses summing to 1. Only focal elements are
stored, so frames with many hypotheses (e.g. 13 UCF-Crime event types plus
"normal") are handled as long as the BBAs have few focal elements.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

BBA = dict[frozenset, float]


def simple_bba(probs: dict[str, float], ignorance: float, frame: Iterable[str]) -> BBA:
    """Bayesian BBA discounted by `ignorance`: m({c}) = (1 - u) p_c, m(Theta) = u."""
    theta = frozenset(frame)
    m: BBA = {frozenset([c]): (1.0 - ignorance) * p for c, p in probs.items() if p > 0}
    if ignorance > 0:
        m[theta] = m.get(theta, 0.0) + ignorance
    return m


def conjunctive(*bbas: BBA) -> tuple[BBA, float]:
    """Unnormalised conjunctive combination of all sources at once.
    Returns (masses on non-empty intersections, conflict K = mass of the empty set)."""
    acc: BBA = dict(bbas[0])
    for m in bbas[1:]:
        nxt: dict = defaultdict(float)
        for a, ma in acc.items():
            for b, mb in m.items():
                c = a & b
                if c:
                    nxt[c] += ma * mb
        acc = dict(nxt)
    # every source sums to 1, so the products sum to 1; the rest fell on the empty set
    return acc, 1.0 - sum(acc.values())


def dempster(*bbas: BBA) -> tuple[BBA, float]:
    """Dempster's rule: conjunctive combination renormalised by 1 - K."""
    m, k = conjunctive(*bbas)
    if k >= 1.0:
        raise ValueError("total conflict (K = 1)")
    return {a: v / (1.0 - k) for a, v in m.items()}, k


def yager(*bbas: BBA, frame: Iterable[str]) -> tuple[BBA, float]:
    """Yager's rule: the conflict mass K is transferred to the whole frame."""
    m, k = conjunctive(*bbas)
    theta = frozenset(frame)
    m = dict(m)
    m[theta] = m.get(theta, 0.0) + k
    return m, k


def pignistic(m: BBA, frame: Sequence[str]) -> dict[str, float]:
    """BetP(c) = sum over focal A containing c of m(A) / |A| (m(empty) = 0)."""
    out = {c: 0.0 for c in frame}
    for a, v in m.items():
        for c in a:
            out[c] += v / len(a)
    return out


def argmax_decision(betp: dict[str, float]) -> str:
    return max(betp, key=betp.get)
