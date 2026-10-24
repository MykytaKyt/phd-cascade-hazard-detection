"""Sequence-grouped cross-validation of the kinematic classifiers with a
held-out validation split inside every training fold.

Protocol (per outer fold and seed):
  1. outer GroupKFold over sequences -> train / test sequences;
  2. 20 % of the training sequences (stratified by fall/ADL) are held out
     as validation; models are trained on the rest;
  3. the decision threshold is the highest-precision threshold with
     Recall >= target on the validation sequences, and is then applied
     unchanged to the test sequences.

The test labels are never used for model selection or thresholding. The
"oracle" metrics (threshold chosen on the test fold itself, as in the
earlier implementation) are reported alongside for comparison only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

FEATURE_COLUMNS = [
    "com_y", "angle", "mos", "bos_area", "vy", "vx", "ay",
    "aspect", "bbox_h", "mean_conf",
]


def prepare_features(df: pd.DataFrame, feature_columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """Forward-fill missing per-frame features within each sequence; frames
    before the first valid detection get 0."""
    df = df.copy()
    for col in feature_columns:
        if col not in df:
            df[col] = 0.0
    df[feature_columns] = df.groupby("seq")[feature_columns].ffill().fillna(0.0)
    return df


def make_windows(df: pd.DataFrame, window: int = 8, feature_columns=FEATURE_COLUMNS, label_col: str = "label"):
    """Sliding windows of length `window` ending at every frame, left-padded
    by repeating the first frame. Returns (x, y, meta) where meta holds
    seq/frame/label aligned with the windows."""
    xs, ys, seqs, frames = [], [], [], []
    for seq, g in df.groupby("seq", sort=True):
        g = g.sort_values("frame")
        arr = g[feature_columns].to_numpy(dtype=np.float32)
        lab = g[label_col].to_numpy(dtype=np.int64)
        for i in range(len(arr)):
            win = arr[max(0, i - window + 1):i + 1]
            if len(win) < window:
                win = np.vstack([np.repeat(win[:1], window - len(win), axis=0), win])
            xs.append(win)
        ys.append(lab)
        seqs.extend([seq] * len(arr))
        frames.extend(g["frame"].tolist())
    meta = pd.DataFrame({"seq": seqs, "frame": frames, "label": np.concatenate(ys)})
    return np.stack(xs), meta["label"].to_numpy(), meta


def split_validation(train_seqs: np.ndarray, fraction: float, rng: np.random.Generator):
    """Hold out `fraction` of the training sequences, stratified by the
    sequence type (prefix before the first '-', e.g. 'fall' / 'adl')."""
    train_seqs = np.unique(train_seqs)
    kinds = np.array([s.split("-")[0] for s in train_seqs])
    val = []
    for kind in np.unique(kinds):
        group = train_seqs[kinds == kind]
        n_val = max(1, int(round(fraction * len(group))))
        val.extend(rng.choice(group, size=n_val, replace=False))
    val = np.array(sorted(val))
    return np.setdiff1d(train_seqs, val), val


def threshold_at_recall(y_true, scores, target: float = 0.95) -> float:
    """Highest-precision threshold with Recall >= target; if the target is
    unreachable, the threshold giving the maximum recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = recall[:-1] >= target
    if ok.any():
        idx = np.flatnonzero(ok)
        return float(thresholds[idx[np.argmax(precision[:-1][idx])]])
    return float(thresholds[int(np.argmax(recall[:-1]))])


def prf(y_true, scores, threshold: float) -> tuple[float, float, float]:
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return float(p), float(r), float(f1)


def train_nn(model: nn.Module, x, y, epochs: int = 25, batch_size: int = 256, lr: float = 1e-3,
             penalty=None) -> nn.Module:
    """Adam + class-weighted BCE, as in the published experiments. `penalty`
    (callable model -> scalar tensor) is added to the loss if given."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_t = torch.tensor(x)
    y_t = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x_t))
        for i in range(0, len(x_t), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_t[idx]), y_t[idx])
            if penalty is not None:
                loss = loss + penalty(model)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def logits_nn(model: nn.Module, x) -> np.ndarray:
    model.eval()
    return model(torch.tensor(x)).numpy()


def threshold_baseline_scores(x, feature_columns=FEATURE_COLUMNS) -> np.ndarray:
    """Training-free physics rule on the last frame of the window (raw,
    unnormalised features): 0.6 * angle / 90 + 0.4 * com_y."""
    i_angle = feature_columns.index("angle")
    i_comy = feature_columns.index("com_y")
    return 0.6 * x[:, -1, i_angle] / 90.0 + 0.4 * x[:, -1, i_comy]


@dataclass
class ModelSpec:
    name: str
    build: object  # callable(in_dim) -> nn.Module, or None for the threshold rule
    penalty: object = None  # callable(model) -> scalar tensor added to the training loss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_nested_cv(
    df: pd.DataFrame,
    specs: list[ModelSpec],
    seeds=(0,),
    n_splits: int = 5,
    window: int = 8,
    val_fraction: float = 0.2,
    recall_target: float = 0.95,
    label_col: str = "label",
    feature_columns=FEATURE_COLUMNS,
):
    """Returns (records, scores):
    records -- one dict per (model, seed, fold) with validation-threshold
               and oracle test metrics;
    scores  -- DataFrame of per-window scores for validation and test
               splits (model, seed, fold, split, seq, frame, label, score),
               where score is a logit for the networks and the raw rule
               value for the threshold baseline.
    """
    df = prepare_features(df, feature_columns)
    x, y, meta = make_windows(df, window=window, feature_columns=feature_columns, label_col=label_col)
    groups = meta["seq"].to_numpy()
    folds = list(GroupKFold(n_splits=n_splits).split(x, y, groups))

    records, score_frames = [], []
    for seed in seeds:
        for fold, (train_idx, test_idx) in enumerate(folds):
            rng = np.random.default_rng(seed * 1000 + fold)
            tr_seqs, val_seqs = split_validation(groups[train_idx], val_fraction, rng)
            tr_idx = np.flatnonzero(np.isin(groups, tr_seqs))
            val_idx = np.flatnonzero(np.isin(groups, val_seqs))

            flat = x[tr_idx].reshape(-1, x.shape[-1])
            mu, sd = flat.mean(0), flat.std(0) + 1e-6
            norm = lambda a: ((a - mu) / sd).astype(np.float32)  # noqa: E731

            for spec in specs:
                torch.manual_seed(seed * 1000 + fold)
                if spec.build is None:
                    s_val = threshold_baseline_scores(x[val_idx], feature_columns)
                    s_test, n_params = threshold_baseline_scores(x[test_idx], feature_columns), 2
                else:
                    model = train_nn(spec.build(x.shape[-1]), norm(x[tr_idx]), y[tr_idx], penalty=spec.penalty)
                    s_val, s_test, n_params = logits_nn(model, norm(x[val_idx])), logits_nn(model, norm(x[test_idx])), count_params(model)

                thr = threshold_at_recall(y[val_idx], s_val, recall_target)
                oracle_thr = threshold_at_recall(y[test_idx], s_test, recall_target)
                p, r, f1 = prf(y[test_idx], s_test, thr)
                op, orr, of1 = prf(y[test_idx], s_test, oracle_thr)
                records.append(dict(
                    model=spec.name, params=n_params, seed=seed, fold=fold,
                    n_train_seqs=len(tr_seqs), n_val_seqs=len(val_seqs),
                    threshold=thr, precision=p, recall=r, f1=f1,
                    oracle_precision=op, oracle_recall=orr, oracle_f1=of1,
                ))
                for split, idx, s in (("val", val_idx, s_val), ("test", test_idx, s_test)):
                    part = meta.iloc[idx].copy()
                    part["score"] = s
                    part["model"], part["seed"], part["fold"], part["split"] = spec.name, seed, fold, split
                    score_frames.append(part)

    return records, pd.concat(score_frames, ignore_index=True)


def summarize(records: list[dict]) -> pd.DataFrame:
    """Mean and std over all (seed, fold) runs, plus std of per-seed means."""
    r = pd.DataFrame(records)
    cols = ["precision", "recall", "f1", "oracle_precision", "oracle_recall", "oracle_f1"]
    agg = r.groupby("model", sort=False).agg(params=("params", "first"), **{
        f"{c}_{s}": (c, s) for c in cols for s in ("mean", "std")
    })
    seed_means = r.groupby(["model", "seed"], sort=False)["f1"].mean()
    agg["f1_seed_std"] = seed_means.groupby(level=0, sort=False).std(ddof=0)
    return agg
