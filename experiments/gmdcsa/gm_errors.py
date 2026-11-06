#!/usr/bin/env python3
"""Where the GMDCSA-24 errors come from (leave-one-subject-out run of gm_and_routing.py). Usage: PYTHONPATH=src python3 experiments/gmdcsa/gm_errors.py"""
import runpy, sys, json, re, collections
import numpy as np, pandas as pd
ns = runpy.run_path(str(__import__("pathlib").Path(__file__).with_name("gm_and_routing.py")), run_name="lib")
G, ev_loso, meta, new, onsets, ffp = ns["G"], ns["ev_loso"], ns["meta"], ns["new"], ns["onsets_gm"], ns["fused_fall_prob"]
info = {new[m["seq"]]: m for m in meta}; e = ev_loso.copy()
pv, uv = G.answers("q32", e); e["pv"] = pv
e["bet"] = [ffp(r.p, r.conf, pv[i], uv[i], 0.0, "dempster") for i, r in enumerate(e.itertuples())]
e["since"] = [r.alarm_frame - onsets[r.seq] if r.seq in onsets else np.nan for r in e.itertuples()]
t = e[e["true"]]
print("true alarm episodes per detected fall and seed: %.2f" % (len(t) / t.groupby(["seed", "seq"]).ngroups))
for lo, hi in ((0, 30), (30, 90), (90, 1e9)):
    g = t[(t["since"] >= lo) & (t["since"] < hi)]; print(f"true alarms {lo}-{hi} frames after onset: n={len(g)}, mean p_V={g['pv'].mean():.2f}, kept={np.mean(g['bet'] >= 0.3):.2f}")
first = t.sort_values("alarm_frame").groupby(["seed", "seq"]).head(1)
print("first true alarm: mean p_V=%.2f, kept=%.2f, median frames after onset=%.0f" % (first["pv"].mean(), (first["bet"] >= 0.3).mean(), first["since"].median()))
kept = t[t["bet"] >= 0.3].groupby(["seed", "seq"]).ngroups; print("falls with a kept alarm per seed", kept / 5, "of", t.groupby(["seed", "seq"]).ngroups / 5)
lost = set(map(tuple, t[["seed", "seq"]].drop_duplicates().to_numpy())) - set(map(tuple, t[t["bet"] >= 0.3][["seed", "seq"]].drop_duplicates().to_numpy()))
c = collections.Counter(s for _, s in lost); print("falls lost by verification:", len(lost) / 5, "per seed;", len(c), "distinct;", sum(v == 5 for v in c.values()), "lost in all 5 seeds")
print("  not fully visible among lost-in-all:", sum("not visible" in info[s]["description"].lower() for s, v in c.items() if v == 5), "| among all falls:", sum("not visible" in m["description"].lower() for m in meta if m["kind"] == "fall"))
print("  examples:", [(s, info[s]["description"][:70], e[(e.seq == s) & e.true]["pv"].round(2).tolist()[:4]) for s, v in list(c.items())[:6] if v == 5])
rs = [G.resp["q32"][G.grid_id(r.seq, r.alarm_frame)]["reason"] for r in first[first["bet"] < 0.3].drop_duplicates("seq").head(8).itertuples()]; print("\n".join("   - " + r for r in rs))
f = e[~e["true"]]; cls = lambda s: sorted({re.sub(r"\s*\(.*", "", c.split("[")[0]).strip() for c in info[s]["classes"].split(";") if c.strip()})
print("false alarms per seed: %.1f in ADL sequences, %.1f before the onset in fall sequences" % (sum(s.startswith("adl") for s in f.seq) / 5, sum(s.startswith("fall") for s in f.seq) / 5))
by = collections.Counter(); n_by = collections.Counter()
for s in G.adls:
    for k in cls(s): n_by[k] += 1
for r in f[f.seq.str.startswith("adl")].drop_duplicates(["seed", "seq"]).itertuples():
    for k in cls(r.seq): by[k] += 1
print("ADL sequences with an alarm per seed, by activity present (of all with it):", {k: (round(by[k] / 5, 1), n_by[k]) for k in n_by})
fa = f[f.seq.str.startswith("adl")]; print("false alarms in ADL kept by Q32 fusion: %.2f; mean p_V %.2f" % ((fa["bet"] >= 0.3).mean(), fa["pv"].mean()))
