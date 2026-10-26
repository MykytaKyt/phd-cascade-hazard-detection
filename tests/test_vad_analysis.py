import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.vad.analysis import (
    anomaly_logit,
    class_probs,
    frame_auc,
    retrieval_metrics,
    routed_segments,
    segment_clip_frames,
    split_videos,
)
from thesis.vad.ucf import ANOMALY_CLASSES, CLASSES

spec = importlib.util.spec_from_file_location(
    "run_zeroshot", Path(__file__).resolve().parents[1] / "experiments" / "ch5_ucfcrime" / "run_zeroshot.py"
)
run_zeroshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_zeroshot)


def synthetic_scores(n_anom=60, n_norm=40, frames=30, seed=0):
    """Anomalous segments raise the similarity of the video's class prompt."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_anom + n_norm):
        cls = ANOMALY_CLASSES[v % len(ANOMALY_CLASSES)] if v < n_anom else "Normal"
        s, e = (rng.integers(5, 12), rng.integers(18, 26)) if cls != "Normal" else (-1, -1)
        for k in range(frames):
            label = int(s <= k <= e)
            sims = rng.normal(0.20, 0.01, len(CLASSES))
            sims[CLASSES.index("Normal")] += 0.03
            if label:
                sims[CLASSES.index(cls)] += 0.05 + rng.normal(0, 0.01)
            rows.append(dict(video=f"v{v:03d}", cls=cls, frame=16 * k, label=label,
                             **{f"sim_{c}": x for c, x in zip(CLASSES, sims)}))
    return pd.DataFrame(rows)


def test_probabilities_and_auc():
    df = synthetic_scores()
    p = class_probs(df)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert frame_auc(df, anomaly_logit(df)) > 0.8


def test_split_videos_stratified_and_disjoint():
    df = synthetic_scores()
    cal, ev = split_videos(df, 0.2, np.random.default_rng(0))
    assert set(cal).isdisjoint(ev) and len(cal) + len(ev) == 100
    cls = df.groupby("video")["cls"].first()
    assert (cls[cal] != "Normal").sum() == 12 and (cls[cal] == "Normal").sum() == 8


def test_retrieval_finds_class_frames():
    r = retrieval_metrics(synthetic_scores())
    assert len(r) == len(ANOMALY_CLASSES) and r["ap"].mean() > 0.5


def test_routed_segments_and_clip():
    df = pd.DataFrame({"video": ["a"] * 6 + ["b"] * 3, "frame": [0, 16, 32, 48, 64, 80, 0, 16, 32]})
    routed = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], bool)
    seg = routed_segments(df, routed, max_gap=1)
    assert seg[["video", "start", "end"]].values.tolist() == [["a", 0, 64], ["b", 32, 32]]
    seg0 = routed_segments(df, routed, max_gap=0)
    assert len(seg0) == 3
    assert segment_clip_frames(0, 80, 6) == [0, 16, 32, 48, 64, 80]


def test_zeroshot_pipeline_conformal_holds():
    df = synthetic_scores(n_anom=150, n_norm=50)
    res, split0 = run_zeroshot.run(df, [0.1], n_splits=3, cal_fraction=0.2)
    crc = res[res["method"] == "crc"]
    assert crc["feasible"].all() and (crc["n_cal_anomalous"] >= 9).all()
    reqs, n = run_zeroshot.build_requests(*split0, tau_a=0.3, max_requests=50)
    assert len(reqs) <= 50 and all(r["kind"] == "ucf_segment" for r in reqs)


spec_f = importlib.util.spec_from_file_location(
    "analyze_ucf", Path(__file__).resolve().parents[1] / "experiments" / "ch5_ucfcrime" / "analyze_vlm.py"
)
analyze_ucf = importlib.util.module_from_spec(spec_f)
spec_f.loader.exec_module(analyze_ucf)


def test_ucf_fusion_pipeline_and_rule_agreement_at_set_share():
    df = synthetic_scores(n_anom=60, n_norm=40)
    _, split0 = run_zeroshot.run(df, [0.1], n_splits=1, cal_fraction=0.2)
    reqs, _ = run_zeroshot.build_requests(*split0, tau_a=0.3, max_requests=200)
    cls_of = df.groupby("video")["cls"].first()
    responses = {r["id"]: dict(id=r["id"], p_anomaly=0.8 if cls_of[r["video"]] != "Normal" else 0.2,
                               uncertainty=0.2, event=cls_of[r["video"]]) for r in reqs}
    out = analyze_ucf.analyze(df, reqs, responses)
    assert out["n_frames_with_vlm"] > 0
    assert out["rule_disagreements"][f"{13 / 14:.3f}"] == 0
    assert out["vlm_event_accuracy_on_anomalous_segments"] == 1.0
