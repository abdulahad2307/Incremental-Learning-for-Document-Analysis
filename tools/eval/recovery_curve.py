"""How much of the classifier forgetting can be recovered with m documents per old class?

On the drift features of scripts/eval/run_drift_features.sh (200 training and 300 test documents per class), after
step k the classifier is replaced by a readout fitted on m randomly drawn training documents of each old class and all
200 of the newest class (the newest class is fully available in CIL):
  probe   multinomial logistic regression on standardised features, class-balanced weights
  ncm     nearest class mean (cosine)
for m in --m (default 5 10 20 50 100 200; 20 is the replay memory per class, 200 is the probe of
forgetting_decomposition.py). Each m is drawn --draws times. Reported on the test documents: accuracy over the old
classes, over all known classes and on the newest class, next to the model's own classifier (clf).

Outputs in --out: recovery_runs.csv (per run x seed x step x readout x modality x m, mean over draws),
recovery_summary.csv (mean / std over seeds), recovery_curve.md, recovery_curve.png (last step, joint feature).

Example:
  python tools/eval/recovery_curve.py --root $RUN_ROOT/features/drift --seeds 42 1 2 --out results/recovery
"""
import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from forgetting_decomposition import JOINT, RUNS, load

M_DEFAULT = [5, 10, 20, 50, 100, 200]


def step_path(root, backbone, run, seed, step):
    base = os.path.join(root, backbone, run) if seed == 42 else os.path.join(root, backbone, run, f"seed{seed}")
    return os.path.join(base, f"step{step}.pt")


def accs(pred, labels, old, new):
    ok = pred == labels
    per = {c: ok[labels == c].mean() for c in np.unique(labels)}
    return dict(acc_old=float(np.mean([per[c] for c in old if c in per])),
                acc_all=float(np.mean(list(per.values()))),
                acc_new=float(np.mean([per[c] for c in new if c in per])))


def fit_predict(readout, xtr, ytr, xte):
    if readout == "probe":
        sc = StandardScaler().fit(xtr)
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(sc.transform(xtr), ytr).predict(sc.transform(xte))
    nrm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    cls = np.unique(ytr)
    means = nrm(np.stack([nrm(xtr[ytr == c]).mean(0) for c in cls]))
    return cls[(nrm(xte) @ means.T).argmax(1)]


def one_job(path, backbone, run, seed, step, prev_known, modalities, ms, draws):
    d = load(path)
    known = np.asarray(d["known"], dtype=object)
    new = [c for c in known if c not in prev_known]
    old = [c for c in known if c in prev_known]
    (trf, try_), (tef, tey, logits) = d["train"], d["test"]
    key = dict(backbone=backbone, run=run, seed=seed, step=step)
    rows = [dict(key, readout="clf", modality=JOINT[backbone], m=np.nan, **accs(known[logits.argmax(1)], tey, old, new))]
    mods = [m for m in trf if modalities == "all" or m == JOINT[backbone]]
    for mod in mods:
        for m in ms:
            res = {"probe": [], "ncm": []}
            for r in range(draws if m < 200 else 1):
                rng = np.random.default_rng(1000 * r + m)
                idx = np.concatenate([np.flatnonzero(try_ == c) if c in new else
                                      rng.permutation(np.flatnonzero(try_ == c))[:m] for c in known])
                for ro in res:
                    res[ro].append(accs(fit_predict(ro, trf[mod][idx], try_[idx], tef[mod]), tey, old, new))
            for ro, lst in res.items():
                df = pd.DataFrame(lst)
                rows.append(dict(key, readout=ro, modality=mod, m=m, **df.mean().to_dict(),
                                 **{f"{c}_draw_std": v for c, v in df.std().fillna(0).to_dict().items()}))
    return rows


def summarise(df):
    cols = ["acc_old", "acc_all", "acc_new"]
    g = df.groupby(["backbone", "run", "step", "readout", "modality", "m"], dropna=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.seed.nunique()
    return out.reset_index()


def markdown(summ, step):
    s = summ[summ.step == step]
    lines = [f"# Recovery of the classifier forgetting after step {step}", "",
             "Old-class accuracy (%) of a readout refitted on m training documents per old class (all 200 of the "
             "newest class), mean +- std over seeds; clf = the model's own classifier. Joint feature.", ""]
    for b in sorted(s.backbone.unique()):
        sb = s[(s.backbone == b) & (s.modality == JOINT[b])]
        ms = sorted(sb.m.dropna().unique())
        lines += [f"## {b}", "", "| run | readout | clf | " + " | ".join(f"m={int(m)}" for m in ms) + " |",
                  "|---|---|---|" + "---|" * len(ms)]
        fmt = lambda r: (f"{100 * r.acc_old_mean:.1f}" +
                         (f"±{100 * r.acc_old_std:.1f}" if r.acc_old_std == r.acc_old_std else ""))
        for run in sorted(sb.run.unique()):
            sr = sb[sb.run == run]
            clf = sr[sr.readout == "clf"]
            if clf.empty:
                continue
            for ro in ["probe", "ncm"]:
                x = sr[sr.readout == ro].set_index("m")
                lines.append(f"| {run} | {ro} | {fmt(clf.iloc[0])} | " +
                             " | ".join(fmt(x.loc[m]) if m in x.index else "-" for m in ms) + " |")
        lines.append("")
    return "\n".join(lines)


def figure(summ, step, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ[summ.step == step]
    bbs = sorted(s.backbone.unique())
    fig, axes = plt.subplots(len(bbs), 2, figsize=(10, 3.6 * len(bbs)), squeeze=False)
    for i, b in enumerate(bbs):
        sb = s[(s.backbone == b) & (s.modality == JOINT[b])]
        runs = [r for r in RUNS if r in set(sb.run) and r != "finetune_std"] + \
               (["finetune_std"] if "finetune_std" in set(sb.run) else [])
        for j, ro in enumerate(["probe", "ncm"]):
            ax = axes[i, j]
            for k, run in enumerate(runs):
                col = f"C{k}"
                x = sb[(sb.run == run) & (sb.readout == ro)].sort_values("m")
                ax.errorbar(x.m, 100 * x.acc_old_mean, yerr=100 * x.acc_old_std.fillna(0), color=col, marker="o",
                            ms=3, capsize=2, label=run)
                clf = sb[(sb.run == run) & (sb.readout == "clf")]
                if not clf.empty:
                    ax.axhline(100 * clf.acc_old_mean.iloc[0], color=col, ls="--", lw=0.8)
            ax.axvline(20, color="grey", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(sorted(sb.m.dropna().unique()))
            ax.set_xticklabels([str(int(m)) for m in sorted(sb.m.dropna().unique())])
            ax.set_xlabel("training documents per old class")
            ax.set_ylabel("old-class accuracy (%)")
            ax.set_title(f"{b}, {ro} (dashed: own classifier)")
            if j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--steps", nargs="+", type=int, default=[5])
    ap.add_argument("--m", nargs="+", type=int, default=M_DEFAULT)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--modalities", choices=["joint", "all"], default="joint")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = []
    for b in a.backbones:
        for run in a.runs:
            for seed in a.seeds:
                for k in a.steps:
                    p, prev = step_path(a.root, b, run, seed, k), step_path(a.root, b, run, seed, k - 1)
                    if k - 1 == 0:
                        prev = os.path.join(a.root, b, "step0.pt")
                    if not (os.path.exists(p) and os.path.exists(prev)):
                        continue
                    prev_known = set(load(prev)["known"])
                    jobs.append(delayed(one_job)(p, b, run, seed, k, prev_known, a.modalities, a.m, a.draws))
    print(f"{len(jobs)} run x seed x step combinations", flush=True)
    rows = [r for res in Parallel(n_jobs=a.jobs, verbose=5)(jobs) for r in res]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "recovery_runs.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(a.out, "recovery_summary.csv"), index=False)
    last = max(a.steps)
    with open(os.path.join(a.out, "recovery_curve.md"), "w") as f:
        f.write(markdown(summ, last))
    figure(summ, last, os.path.join(a.out, "recovery_curve.png"))
    print(markdown(summ, last))


if __name__ == "__main__":
    main()
