#!/usr/bin/env python3
"""Chapter 5: K-class evidence fusion on UCF-Crime from cached VLM responses.

Frame of discernment: 13 anomaly classes + Normal (n = 14); B = the anomaly classes.
For every evaluation frame of split 0 (same split and calibration as run_zeroshot.py):
  zero-shot evidence m_C: calibrated class distribution discounted by its normalised
      entropy, m_C({c}) = (1 - u_C) q_c, m_C(Theta) = u_C;
  semantic evidence m_V (frames inside a routed segment with a VLM answer):
      m_V({event}) = p_V (1 - u_V) if the event is an anomaly class, else on B;
      m_V({Normal}) = (1 - p_V)(1 - u_V), m_V(Theta) = u_V;
  decision: alarm iff BetP(B) >= beta (Dempster's or Yager's rule); frames outside the
  routed segments keep the zero-shot decision p >= 1/2.
Because the calibrated p of UCF-Crime frames rarely reaches 1/2 (about 8 % of frames are
anomalous), the zero-shot level is also reported at the threshold t_cal that maximises the
frame F1 on the calibration videos ("tuned"); the tuned cascade keeps p >= t_cal outside
the routed segments.

Reports frame-level P/R/F1 and AUC, video-level detection and false alarms, the
number of decisions where the two rules differ (statement 2.8: they coincide at
beta = 13/14), and the VLM event-class accuracy on anomalous segments.

Writes results/ch5/fusion.json.

Usage:
    python experiments/ch5_ucfcrime/analyze_vlm.py --responses results/vlm/ucf_responses.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from thesis.fusion.dst import dempster, pignistic, yager
from thesis.routing.calibration import calibrate_platt, fit_platt
from thesis.vad.analysis import LOGIT_SCALE, anomaly_logit, split_videos
from thesis.vad.ucf import ANOMALY_CLASSES, CLASSES

ROOT = Path(__file__).resolve().parents[2]
B = frozenset(ANOMALY_CLASSES)
THETA = frozenset(CLASSES)
BETAS = (0.5, 0.3, 0.1, 13 / 14)


def zero_shot_bba(q: np.ndarray) -> dict:
    h = -(q * np.log(np.clip(q, 1e-12, 1))).sum() / np.log(len(q))
    u = float(np.clip(h, 0, 1))
    m = {frozenset([c]): (1 - u) * float(v) for c, v in zip(CLASSES, q) if v > 0}
    m[THETA] = m.get(THETA, 0.0) + u
    return m


def semantic_bba(p_v: float, u_v: float, event: str) -> dict:
    anomaly_set = frozenset([event]) if event in ANOMALY_CLASSES else B
    m = {anomaly_set: p_v * (1 - u_v), frozenset(["Normal"]): (1 - p_v) * (1 - u_v)}
    m[THETA] = m.get(THETA, 0.0) + u_v
    return {k: v for k, v in m.items() if v > 0}


def bet_b(m: dict) -> float:
    betp = pignistic(m, CLASSES)
    return sum(betp[c] for c in ANOMALY_CLASSES)


def prf(y, pred):
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12), fn=fn, fp=fp)


def video_outcomes(ev: pd.DataFrame, pred: np.ndarray) -> dict:
    d = ev.assign(pred=pred)
    anom = d[d["cls"] != "Normal"]
    det = anom[anom["label"] == 1].groupby("video")["pred"].any()
    fa = d[d["cls"] == "Normal"].groupby("video")["pred"].any()
    return dict(videos_detected=float(det.mean()), normal_videos_with_alarm=float(fa.mean()))


def f1_threshold(p: np.ndarray, y: np.ndarray, grid: int = 200) -> float:
    """Threshold maximising the frame F1 (searched over quantiles of p)."""
    cands = np.unique(np.quantile(p, np.linspace(0.5, 0.999, grid)))
    return float(max(cands, key=lambda t: prf(y, p >= t)["f1"]))


def analyze(df: pd.DataFrame, requests: list[dict], responses: dict, cal_fraction: float = 0.2) -> dict:
    z = anomaly_logit(df)
    y = df["label"].to_numpy()
    cal_v, eval_v = split_videos(df, cal_fraction, np.random.default_rng(0))
    cal = df["video"].isin(cal_v).to_numpy()
    a, b = fit_platt(z[cal], y[cal])
    ev = df[~cal].reset_index(drop=True)
    p = calibrate_platt(z[~cal], a, b)
    t_cal = f1_threshold(calibrate_platt(z[cal], a, b), y[cal])
    # class distribution consistent with p: anomaly classes share p by the zero-shot softmax
    sims = ev[[f"sim_{c}" for c in ANOMALY_CLASSES]].to_numpy(dtype=np.float64)
    q = np.column_stack([softmax(LOGIT_SCALE * sims, axis=1) * p[:, None], 1.0 - p])
    y_ev = ev["label"].to_numpy()

    seg_of = np.full(len(ev), -1)
    seg_resp = []
    for k, r in enumerate(requests):
        if r["id"] not in responses:
            continue
        mask = ((ev["video"] == r["video"]) & (ev["frame"] >= r["start"]) & (ev["frame"] <= r["end"])).to_numpy()
        seg_of[mask] = len(seg_resp)
        seg_resp.append(responses[r["id"]])
    inside = seg_of >= 0

    base_pred = p >= 0.5
    tuned_pred = p >= t_cal
    out = dict(n_eval_frames=int(len(ev)), n_frames_with_vlm=int(inside.sum()), n_segments=len(seg_resp),
               anomalous_frames_with_vlm=float(inside[y_ev == 1].mean()), t_cal=t_cal,
               zero_shot=dict(auc=float(roc_auc_score(y_ev, p)), **prf(y_ev, base_pred), **video_outcomes(ev, base_pred)),
               zero_shot_tuned=dict(**prf(y_ev, tuned_pred), **video_outcomes(ev, tuned_pred)))

    fused = {}
    for rule in ("dempster", "yager"):
        score = p.copy()
        for i in np.flatnonzero(inside):
            r = seg_resp[seg_of[i]]
            m_c, m_v = zero_shot_bba(q[i]), semantic_bba(r["p_anomaly"], r["uncertainty"], r.get("event", ""))
            m = dempster(m_c, m_v)[0] if rule == "dempster" else yager(m_c, m_v, frame=CLASSES)[0]
            score[i] = bet_b(m)
        fused[rule] = score
        for beta in BETAS:
            pred = base_pred.copy()
            pred[inside] = score[inside] >= beta
            out[f"{rule}_beta{beta:.3f}"] = dict(auc=float(roc_auc_score(y_ev, np.where(inside, score, p))),
                                                 **prf(y_ev, pred), **video_outcomes(ev, pred))
            pred_t = tuned_pred.copy()
            pred_t[inside] = score[inside] >= beta
            out[f"{rule}_beta{beta:.3f}_tuned"] = dict(**prf(y_ev, pred_t), **video_outcomes(ev, pred_t))
    out["rule_disagreements"] = {
        f"{beta:.3f}": int(((fused["dempster"][inside] >= beta) != (fused["yager"][inside] >= beta)).sum())
        for beta in BETAS
    }
    events = [(r.get("event", ""), ev.loc[seg_of == k, "cls"].iloc[0], ev.loc[seg_of == k, "label"].max())
              for k, r in enumerate(seg_resp)]
    anomalous = [(e, c) for e, c, lab in events if lab == 1]
    out["vlm_event_accuracy_on_anomalous_segments"] = (
        float(np.mean([e == c for e, c in anomalous])) if anomalous else None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default=ROOT / "results" / "ch5" / "clip_scores.parquet")
    parser.add_argument("--requests", default=ROOT / "results" / "vlm" / "ucf_requests.jsonl")
    parser.add_argument("--responses", default=ROOT / "results" / "vlm" / "ucf_responses.jsonl")
    parser.add_argument("--out", default=ROOT / "results" / "ch5" / "fusion.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.scores)
    with open(args.requests) as f:
        requests = [json.loads(x) for x in f if x.strip()]
    responses = {}
    with open(args.responses) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                responses[r["id"]] = r
    out = analyze(df, requests, responses)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
