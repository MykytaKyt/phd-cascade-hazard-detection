#!/usr/bin/env python3
"""Second dataset (GMDCSA-24) and the routing baselines, from the files brought back from the GPU server
(pod_results/: pose features of GMDCSA-24 and VLM answers on a dense grid of 6-frame clips, one clip
every 10 frames, for URFD and GMDCSA-24). No VLM is run here.
  1. GMDCSA-24, leave-one-subject-out: kinematic stage, alarm episodes, verification, bootstrap.
  2. Transfer: kinematic stage trained on URFD, applied unchanged to GMDCSA-24.
  3. Both datasets: VLM-only detection at a fixed stride; random routing with the cascade's number of clips.
An alarm episode is verified on the grid clip nearest to the alarm frame (at most 5 frames away).
Inputs (results/gmdcsa/): gm_meta.json, gm_features.parquet, {gm,urfd}_grid_{q32,e4b}.jsonl.
Usage: PYTHONPATH=src python3 experiments/gmdcsa/gm_and_routing.py   -> results/gmdcsa/gm_and_routing.json"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
OUT = POD = ROOT / "results" / "gmdcsa"   # inputs from the GPU server and outputs side by side
sys.path.insert(0, str(ROOT / "experiments" / "ch4_fusion"))
import thesis.evaluation as ev  # noqa: E402
from analyze_vlm import fused_fall_prob, load_responses  # noqa: E402
from thesis.changepoint.cusum import alarm_episodes  # noqa: E402
from thesis.kinematic.labels import add_labels  # noqa: E402
from thesis.kinematic.models import LTCClassifier  # noqa: E402
from thesis.vlm.requests import alarm_events, calibrate_folds, clip_frames, request_id  # noqa: E402

torch.set_num_threads(6)
SEEDS = (0, 1, 2, 3, 4); STRIDE = 10; VLMS = (("q32", "Qwen3-VL-32B"), ("e4b", "Gemma 4 E4B"))
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)
res = {}


class Folds:
    """Stands in for GroupKFold inside run_nested_cv: one fold per given list of test sequences."""
    def __init__(self, tests): self.tests = tests
    def __call__(self, n_splits): return self
    def split(self, x, y, groups):
        for t in self.tests:
            te = np.isin(groups, t); yield np.flatnonzero(~te), np.flatnonzero(te)


# ---------------- data
meta = json.load(open(POD / "gm_meta.json"))
gm = pd.read_parquet(POD / "gm_features.parquet")
new = {m["seq"]: f"{m['kind']}-gm{m['subject']}-{m['seq'][-2:]}" for m in meta}        # kind first: split_validation stratifies by it
orig = {v: k for k, v in new.items()}
gm["seq"] = gm["seq"].map(new)
onsets_gm = {}
for m in meta:
    if m["kind"] != "fall": continue
    hit = re.search(r"Fall\w*[^\[;]*\[\s*([\d.]+)", m["classes"])                        # start of the Fall/Falling interval, s
    start = float(hit.group(1)) if hit else max(float(x) for x in re.findall(r"to\s*([\d.]+)", m["classes"]))  # 1 video without it
    onsets_gm[new[m["seq"]]] = max(1, int(round(start * m["fps"])))
gm["onset"] = gm["seq"].map(onsets_gm)
gm["fall"] = (gm["onset"].notna() & (gm["frame"] >= gm["onset"])).astype(int)
assert len(onsets_gm) == 79 and gm["seq"].nunique() == 160
subject = {new[m["seq"]]: m["subject"] for m in meta}
urfd = add_labels(pd.read_parquet(ROOT / "data/features.parquet"), ROOT / "data")
onsets_ur = urfd.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
res["gm_data"] = dict(frames=len(gm), fall_frames=int(gm["fall"].sum()), valid_pose=float(gm["valid"].mean()),
                      minutes=float(sum(m["n_frames"] / m["fps"] for m in meta) / 60), sequences=pd.DataFrame(meta).groupby(["subject", "kind"]).size().unstack().to_dict())


def scores_for(name, df, folds, cols=ev.FEATURE_COLUMNS):
    path = CACHE / f"scores_{name}.parquet"
    if path.exists(): return json.load(open(CACHE / f"records_{name}.json")), pd.read_parquet(path)
    ev.GroupKFold = Folds(folds)
    records, scores = ev.run_nested_cv(df, [ev.ModelSpec("ltc24", lambda d: LTCClassifier(d, hidden=24))], seeds=SEEDS, label_col="fall", feature_columns=cols)
    scores.to_parquet(path, index=False); json.dump(records, open(CACHE / f"records_{name}.json", "w"), default=float)
    return records, scores


class Data:
    """One dataset: frames, onsets, grid answers; outcome matrices of a set of alarm episodes."""
    def __init__(self, df, onsets, tag, orig_name=lambda s: s):
        self.onsets, self.orig = onsets, orig_name
        self.frames = {s: g["frame"].to_numpy() for s, g in df.groupby("seq")}
        self.conf = df.set_index(["seq", "frame"])["mean_conf"].fillna(0.0).to_dict()
        self.seqs = sorted(self.frames); self.falls = [s for s in self.seqs if s in onsets]; self.adls = [s for s in self.seqs if s not in onsets]
        self.col = {s: i for i, s in enumerate(self.falls + self.adls)}
        self.resp = {k: load_responses(POD / f"{tag}_grid_{k}.jsonl") for k, _ in VLMS}

    def grid_id(self, seq, frame):
        fr = self.frames[seq]; anchors = fr[::STRIDE]; a = anchors[np.argmin(np.abs(anchors - frame))]
        return request_id(self.orig(seq), clip_frames(int(a), fr), "clip6")

    def answers(self, k, events):
        r = [self.resp[k][self.grid_id(e.seq, e.alarm_frame)] for e in events.itertuples()]
        return np.array([x["p_anomaly"] for x in r], float), np.array([x["uncertainty"] for x in r], float)

    def prepare(self, events):
        events = events.reset_index(drop=True)
        events["true"] = [s in self.onsets and f >= self.onsets[s] for s, f in zip(events["seq"], events["alarm_frame"])]
        events["conf"] = [self.conf.get((s, f), 0.0) for s, f in zip(events["seq"], events["alarm_frame"])]
        return events

    def matrices(self, events, keep):
        """D[seed, fall] = detected; F[seed, seq] = false-alarm episodes (fall sequences first)."""
        D = np.zeros((len(SEEDS), len(self.falls))); F = np.zeros((len(SEEDS), len(self.seqs)))
        for e, k in zip(events.itertuples(), keep):
            if not k: continue
            if e.true: D[SEEDS.index(e.seed), self.col[e.seq]] = 1
            else: F[SEEDS.index(e.seed), self.col[e.seq]] += 1
        return D, F

    def row(self, events, keep):
        D, F = self.matrices(events, keep); nf = len(self.falls)
        return dict(det=float(D.mean()), det_std=float(D.mean(1).std()), fa=float(F.sum(1).mean()), fa_std=float(F.sum(1).std()),
                    adl_seqs=float((F[:, nf:] > 0).sum(1).mean()), calls=len(events) / len(SEEDS))

    def boot(self, events, keep, base, B=10000):
        """95% intervals over sequences (falls and ADL resampled separately), paired with the kinematic stage."""
        rng = np.random.default_rng(0); nf, na = len(self.falls), len(self.adls)
        i_f = rng.integers(0, nf, (B, nf)); i_a = rng.integers(0, na, (B, na))
        def stat(D, F): return D[:, i_f].mean(axis=(0, 2)), (F[:, :nf][:, i_f].sum(2) + F[:, nf:][:, i_a].sum(2)).mean(0)
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        d0, f0 = stat(*self.matrices(events, base)); d1, f1 = stat(*self.matrices(events, keep))
        return dict(det_ci=ci(d1), fa_ci=ci(f1), d_det_ci=ci(d1 - d0), d_fa_ci=ci(f1 - f0))

    def cascade(self, events):
        """Kinematics, fusion (Dempster, alpha = 0, beta = 0.3), VLM alone on the alarm clips."""
        events = self.prepare(events); all_ = np.ones(len(events), bool)
        out = {"Kinematics only": {**self.row(events, all_), **self.boot(events, all_, all_)}}
        uniq = events.drop_duplicates(["seq", "alarm_frame"])
        out["episodes"] = dict(total=len(events), distinct=len(uniq), distinct_true=int(uniq["true"].sum()), per_seed=events.groupby("seed").size().tolist())
        for k, name in VLMS:
            pv, uv = self.answers(k, events)
            bet = np.array([fused_fall_prob(e.p, e.conf, pv[i], uv[i], 0.0, "dempster") for i, e in enumerate(events.itertuples())])
            out[f"Fusion, {name}"] = {**self.row(events, bet >= 0.3), **self.boot(events, bet >= 0.3, all_),
                                      "auc": float(roc_auc_score(uniq["true"], pv[uniq.index])) if uniq["true"].nunique() == 2 else None,
                                      "p_true": float(pv[uniq.index][uniq["true"]].mean()), "p_false": float(pv[uniq.index][~uniq["true"]].mean())}
            out[f"VLM alone on alarm clips, {name}"] = self.row(events, pv >= 0.5)
        return out, events

    def threshold_sweep(self, scores, fa_targets):
        """Kinematic stage with the alarm threshold swept on the test sequences (optimistic reference):
        best detection rate with at most the given number of false-alarm episodes per run."""
        curve = []
        parts = [(seed, pd.concat([t for _, t in calibrate_folds(sg)])) for seed, sg in scores.groupby("seed")]
        seqs = [(seed, s, g.sort_values("frame")) for seed, test in parts for s, g in test.groupby("seq")]
        seqs = [(seed, s, g["frame"].to_numpy(), g["z"].to_numpy()) for seed, s, g in seqs]
        for h in np.linspace(-4, 12, 161):
            det = fa = 0
            for seed, s, fr, z in seqs:
                ep = alarm_episodes(fr[z >= h], 90); on = self.onsets.get(s)
                det += on is not None and bool((fr[z >= h] >= on).any()); fa += len(ep) if on is None else int((ep < on).sum())
            curve.append((float(h), det / len(SEEDS) / len(self.falls), fa / len(SEEDS)))
        return {str(t): max((d for _, d, f in curve if f <= t), default=0.0) for t in fa_targets}, curve

    # ---- baselines without the kinematic stage
    def clips(self, k):
        rows = []
        for s in self.seqs:
            fr = self.frames[s]
            for a in fr[::STRIDE]:
                clip = clip_frames(int(a), fr); rid = request_id(self.orig(s), clip, "clip6")
                rows.append((s, clip[-1], self.resp[k][rid]["p_anomaly"], rid))      # the answer exists once the last frame is captured
        return pd.DataFrame(rows, columns=["seq", "anchor", "pv", "id"])

    def outcome(self, sel):
        """sel: clips answered 'fall'; the alarm time is the last frame of the clip. Episodes merged as for the kinematic alarms."""
        det, fa, adl, delays = 0, 0, 0, []
        for s, g in sel.groupby("seq"):
            a = np.sort(g["anchor"].to_numpy()); on = self.onsets.get(s)
            if on is None: n = len(alarm_episodes(a, 90)); fa += n; adl += n > 0
            else:
                fa += len(alarm_episodes(a[a < on], 90)); late = a[a >= on]
                if len(late): det += 1; delays.append(late[0] - on)
        return det / len(self.falls), fa, adl, delays

    def vlm_only(self, k, every, thr=0.5):
        c = self.clips(k); c = c[c.groupby("seq").cumcount() % every == 0]
        det, fa, adl, delays = self.outcome(c[c["pv"] >= thr])
        return dict(stride=STRIDE * every, calls=c["id"].nunique(), det=det, fa=fa, adl_seqs=adl, delay_median=float(np.median(delays)) if delays else None)

    def random_routing(self, k, n_per_seed, draws=2000, thr=0.5):
        c = self.clips(k).drop_duplicates("id"); rng = np.random.default_rng(0); out = []
        for n in n_per_seed:
            for _ in range(draws):
                pick = c.iloc[rng.choice(len(c), n, replace=False)]
                out.append(self.outcome(pick[pick["pv"] >= thr])[:2])
        d, f = np.array(out).T
        return dict(calls=float(np.mean(n_per_seed)), det=float(d.mean()), det_q=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    fa=float(f.mean()), fa_q=[float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))])


def no_kinematics(data, n_per_seed):
    out = {}
    for k, name in VLMS:
        out[name] = {"vlm_only": [data.vlm_only(k, e, t) | {"thr": t} for e in (1, 3, 9) for t in (0.5,)],
                     "random": data.random_routing(k, n_per_seed)}
    return out


# ---------------- 1. GMDCSA-24, leave-one-subject-out
G = Data(gm, onsets_gm, "gm", lambda s: orig[s])
rec, sc = scores_for("gm_loso", gm, [[s for s in G.seqs if subject[s] == k] for k in (1, 2, 3, 4)])
fr = ev.summarize(rec).iloc[0]
ev_loso = alarm_events(sc, onsets_gm)
res["gm_loso"], ev_loso = G.cascade(ev_loso)
res["gm_loso"]["frame"] = dict(f1=fr["f1_mean"], f1_std=fr["f1_std"], precision=fr["precision_mean"], recall=fr["recall_mean"])
res["gm_loso"]["delay_median_frames"] = float(np.median([e.alarm_frame - onsets_gm[e.seq] for e in ev_loso[ev_loso["true"]].groupby(["seed", "seq"]).head(1).itertuples()]))
res["gm_loso"]["threshold_sweep"], _ = G.threshold_sweep(sc, [round(res["gm_loso"][f"Fusion, {n}"]["fa"], 1) for _, n in VLMS])
res["gm_loso"]["per_subject_det"] = {k: float(np.mean([G.matrices(ev_loso, np.ones(len(ev_loso), bool))[0][:, [i for i, s in enumerate(G.falls) if subject[s] == k]].mean()])) for k in (1, 2, 3, 4)}

# ---------------- 2. transfer URFD -> GMDCSA-24
both = pd.concat([urfd, gm], ignore_index=True)
rec, sc_t = scores_for("urfd_to_gm", both, [G.seqs])
fr = ev.summarize(rec).iloc[0]
res["transfer"], ev_tr = G.cascade(alarm_events(sc_t, {**onsets_ur, **onsets_gm}))
res["transfer"]["frame"] = dict(f1=fr["f1_mean"], f1_std=fr["f1_std"], precision=fr["precision_mean"], recall=fr["recall_mean"])
res["transfer"]["threshold_sweep"], _ = G.threshold_sweep(sc_t, [round(res["transfer"][f"Fusion, {n}"]["fa"], 1) for _, n in VLMS])

# ---------------- 3. VLM-only and random routing
res["gm_no_kinematics"] = no_kinematics(G, res["gm_loso"]["episodes"]["per_seed"])
U = Data(urfd, onsets_ur, "urfd")
ev_ur = U.prepare(pd.read_parquet(ROOT / "results/vlm/urfd_events.parquet"))
k0 = U.row(ev_ur, np.ones(len(ev_ur), bool)); assert abs(k0["det"] - 0.800) < 1e-9 and abs(k0["fa"] - 12.6) < 1e-9, k0
res["urfd_no_kinematics"] = no_kinematics(U, ev_ur.groupby("seed").size().tolist())
# the grid clip nearest to the alarm against the exact alarm clip of the main experiment (cached answers)
old = load_responses(ROOT / "results/vlm/urfd_responses_qwen3-vl-32b.jsonl")
pv_grid, uv_grid = U.answers("q32", ev_ur)
pv_old = np.array([old[request_id(e.seq, clip_frames(e.alarm_frame, U.frames[e.seq]), "clip6")]["p_anomaly"] for e in ev_ur.itertuples()])
bet = np.array([fused_fall_prob(e.p, e.conf, pv_grid[i], uv_grid[i], 0.0, "dempster") for i, e in enumerate(ev_ur.itertuples())])
res["urfd_grid_check"] = dict(same_side=float(((pv_grid >= 0.5) == (pv_old >= 0.5)).mean()), fusion_with_grid_clip=U.row(ev_ur, bet >= 0.3))

# ---------------- 4. raw keypoints against the DDL features, within GMDCSA-24 and in transfer (kinematic stage only)
ukp = add_labels(pd.read_parquet(ROOT / "data/features_kp.parquet"), ROOT / "data")      # second pose-extraction pass of URFD, with keypoints
KP = []
for i in range(17):
    for d in (ukp, gm): d[f"kp{i}_x"] /= d["W"]; d[f"kp{i}_y"] /= d["H"]
    KP += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_c"]
both_kp = pd.concat([ukp, gm], ignore_index=True); loso = [[s for s in G.seqs if subject[s] == k] for k in (1, 2, 3, 4)]
res["representation"] = {}
for name, df_, folds, cols, ons in (("GMDCSA-24, leave-one-subject-out | raw keypoints", gm, loso, KP, onsets_gm),
                                    ("URFD to GMDCSA-24 | 10 DDL features", both_kp, [G.seqs], ev.FEATURE_COLUMNS, {**onsets_ur, **onsets_gm}),
                                    ("URFD to GMDCSA-24 | raw keypoints", both_kp, [G.seqs], KP, {**onsets_ur, **onsets_gm})):
    rec, s_ = scores_for(re.sub(r"\W+", "_", name), df_, folds, cols); fr = ev.summarize(rec).iloc[0]
    e_ = G.prepare(alarm_events(s_, ons))
    res["representation"][name] = dict(f1=fr["f1_mean"], f1_std=fr["f1_std"], **G.row(e_, np.ones(len(e_), bool)))

if __name__ == "__main__":
    json.dump(res, open(OUT / "gm_and_routing.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
