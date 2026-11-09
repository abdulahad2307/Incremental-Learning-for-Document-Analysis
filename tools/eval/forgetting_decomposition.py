"""Where does the forgetting happen? Split the CIL forgetting of every run into a representation part and a readout
(classifier) part, per modality, on the drift features of scripts/eval/run_drift_features.sh.

Per step k and readout r, per-class accuracy on the 300 test documents of each known class:
  clf     the model's own classifier (argmax of its logits; joint feature only)
  probe   a linear probe (multinomial logistic regression on standardised features) fitted on the step's own features
          of 200 training documents of every known class - the best linear readout of what the representation still
          holds (Davari et al., 2022, "Probing representation forgetting"); fitted per modality
  ncm     nearest class mean (cosine) on the same 200 documents (a readout without training)
Forgetting of readout r after step k (as in utils/metrics/il_metrics.py, per class): mean over the classes learned
before step k of acc_r(class, step it was learned) - acc_r(class, k); base classes are learned at step 0.
Decomposition after step k (joint feature: EAML fused, LayoutLMv3 cls):
  total           = forgetting of clf
  representation  = forgetting of probe  (what no readout trained on the step's features could recover)
  readout         = total - representation (what re-fitting the classifier would recover)
The probe uses 200 training documents of each old class, more than the replay memory holds (20), so it is a
diagnostic, not a CIL method; the recoverable part with the replay memory alone is in classifier_recal (NCM20).
Per-modality representation forgetting (EAML image / text, LayoutLMv3 text / visual) shows which branch loses the
information. A class bootstrap gives 95 % intervals of every forgetting value.

Feature folders: <root>/<backbone>/step0.pt (base model) and <root>/<backbone>/<run>/step<k>.pt (seed 42), or
<root>/<backbone>/<run>/seed<N>/step<k>.pt (other seeds; --seeds 42 1 2).
Outputs in --out: forgetting_classes.csv (per class), forgetting_decomposition.csv (per run x seed x step),
forgetting_summary.csv (mean / std over seeds), forgetting_decomposition.md (per seed: class-bootstrap intervals;
over seeds: mean +- std), figure (last step, mean over seeds).

Example:
  python tools/eval/forgetting_decomposition.py --root $RUN_ROOT/features/drift --out results/forgetting
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

JOINT = {"eaml": "fused", "llmv3": "cls", "llmv3hf": "cls"}
RUNS = ["finetune_std", "noevm_std", "noevm_kd", "evm_std", "evm_kd", "ievm_std", "evmood_std",
        "evmood_std_energygrad"]
N_BOOT = 1000


def load(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    known, S = list(d["known_classes"]), d["splits"]
    tr_y = np.asarray(S["train"]["labels"], dtype=object)
    keep = np.isin(tr_y, known)                       # the train split may hold all 16 classes (--train_classes all)
    te = S["test_seen"]
    return {"known": known,
            "train": ({m: np.asarray(v, dtype=np.float32)[keep] for m, v in S["train"]["features"].items()},
                      tr_y[keep]),
            "test": ({m: np.asarray(v, dtype=np.float32) for m, v in te["features"].items()},
                     np.asarray(te["labels"], dtype=object), np.asarray(te["logits"], dtype=np.float32))}


def per_class(pred, labels, classes):
    return {c: float(np.mean(pred[labels == c] == c)) for c in classes if np.any(labels == c)}


def readouts(step, joint):
    """{(readout, modality): {class: accuracy}} of one step."""
    known = np.asarray(step["known"], dtype=object)
    (trf, try_), (tef, tey, logits) = step["train"], step["test"]
    out = {("clf", joint): per_class(known[logits.argmax(1)], tey, known)}
    for m in trf:
        sc = StandardScaler().fit(trf[m])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(trf[m]), try_)
        out[("probe", m)] = per_class(lr.predict(sc.transform(tef[m])), tey, known)
        nrm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        means = np.stack([nrm(trf[m][try_ == c]).mean(0) for c in known])
        out[("ncm", m)] = per_class(known[(nrm(tef[m]) @ nrm(means).T).argmax(1)], tey, known)
    return out


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    b = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyse(backbone, run, steps):
    joint = JOINT[backbone]
    acc = {k: readouts(s, joint) for k, s in steps.items()}
    learned = {c: 0 for c in steps[0]["known"]}
    for k in sorted(steps)[1:]:
        for c in steps[k]["known"]:
            learned.setdefault(c, k)
    rows = []
    for k in sorted(steps)[1:]:
        for (r, m), a in acc[k].items():
            for c, v in a.items():
                j = learned[c]
                if j < k and j in acc and c in acc[j].get((r, m), {}):
                    rows.append(dict(backbone=backbone, run=run, step=k, readout=r, modality=m, cls=c,
                                     learned_at=j, acc_then=acc[j][(r, m)][c], acc_now=v,
                                     forgetting=acc[j][(r, m)][c] - v))
    return rows


def decompose(classes):
    rng = np.random.default_rng(0)
    rows = []
    for (b, run, seed, k), g in classes.groupby(["backbone", "run", "seed", "step"]):
        joint = JOINT[b]
        f = lambda r, m: g[(g.readout == r) & (g.modality == m)].set_index("cls")["forgetting"]
        tot, rep = f("clf", joint), f("probe", joint)
        common = tot.index.intersection(rep.index)
        row = dict(backbone=b, run=run, seed=seed, step=k, n_old=len(common))
        for name, x in [("total", tot[common]), ("representation", rep[common]),
                        ("readout", tot[common] - rep[common])]:
            row[name] = float(x.mean())
            row[f"{name}_lo"], row[f"{name}_hi"] = boot_ci(x.values, rng)
        row["readout_share"] = row["readout"] / row["total"] if row["total"] > 0.01 else float("nan")
        for m in sorted(g.modality.unique()):
            p, n = f("probe", m), f("ncm", m)
            row[f"repr_{m}"] = float(p.mean())
            row[f"repr_{m}_lo"], row[f"repr_{m}_hi"] = boot_ci(p.values, rng)
            row[f"ncm_{m}"] = float(n.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(dec):
    cols = [c for c in dec.columns if c not in ("backbone", "run", "seed", "step", "n_old")
            and not c.endswith(("_lo", "_hi"))]
    g = dec.groupby(["backbone", "run", "step"])
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.seed.nunique()
    return out.reset_index()


def markdown_seeds(summ):
    pc = lambda m, s: "-" if pd.isna(m) else f"{100 * m:.1f}" + ("" if pd.isna(s) else f" ± {100 * s:.1f}")
    lines = ["", "# Over seeds (mean ± std)", ""]
    for b, g in summ.groupby("backbone"):
        mods = sorted(c[5:-5] for c in g.columns if c.startswith("repr_") and c.endswith("_mean")
                      and g[c].notna().any())
        lines += [f"## {b}", "", "| run | step | seeds | total | representation | readout | "
                  + " | ".join(f"repr {m}" for m in mods) + " |", "|" + "---|" * (6 + len(mods))]
        for _, r in g.sort_values(["run", "step"]).iterrows():
            v = lambda n: pc(r[f"{n}_mean"], r[f"{n}_std"])
            lines.append(f"| {r.run} | {r.step} | {r.n_seeds} | {v('total')} | {v('representation')} | "
                         f"{v('readout')} | " + " | ".join(v(f"repr_{m}") for m in mods) + " |")
        lines.append("")
    return "\n".join(lines)


def markdown(dec):
    pc = lambda v: "-" if pd.isna(v) else f"{100 * v:.1f}"
    lines = ["# Forgetting decomposition", "",
             "Forgetting in points (mean over old classes, 95 % class-bootstrap interval). total = own classifier, "
             "representation = linear probe on the step's features, readout = total - representation; "
             "repr_<modality> = linear-probe forgetting of that modality alone.", ""]
    for b, g in dec.groupby("backbone"):
        mods = sorted(c[5:] for c in g.columns if c.startswith("repr_") and not c.endswith(("_lo", "_hi"))
                      and g[c].notna().any())
        lines += [f"## {b}", "", "| run | step | total | representation | readout | readout share | "
                  + " | ".join(f"repr {m}" for m in mods) + " |",
                  "|" + "---|" * (6 + len(mods))]
        for _, r in g.sort_values(["run", "seed", "step"]).iterrows():
            ci = lambda n: f"{pc(r[n])} [{pc(r[n + '_lo'])}, {pc(r[n + '_hi'])}]"
            lines.append(f"| {r.run} (seed {r.seed}) | {r.step} | {ci('total')} | {ci('representation')} | {ci('readout')} | "
                         f"{pc(r.readout_share)} | " + " | ".join(pc(r[f'repr_{m}']) for m in mods) + " |")
        lines.append("")
    return "\n".join(lines)


def figure(summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    last = summ.loc[summ.groupby(["backbone", "run"]).step.idxmax()].rename(
        columns={"total_mean": "total", "representation_mean": "representation", "readout_mean": "readout"})
    bbs = sorted(last.backbone.unique())
    fig, axes = plt.subplots(1, len(bbs), figsize=(6 * len(bbs), 3.8), squeeze=False)
    for ax, b in zip(axes[0], bbs):
        g = last[last.backbone == b].sort_values("total")
        x = np.arange(len(g))
        ax.bar(x, 100 * g.representation, label="representation", color="#4c72b0")
        ax.bar(x, 100 * g.readout, bottom=100 * g.representation.clip(lower=0), label="readout (classifier)",
               color="#dd8452")
        ax.errorbar(x, 100 * g.total, yerr=100 * g.total_std.fillna(0), fmt="k.", capsize=3,
                    label="total (± std over seeds)")
        ax.set_xticks(x, [f"{r}\n(step {s})" for r, s in zip(g.run, g.step)], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("forgetting (points)")
        ax.set_title(b)
        ax.axhline(0, color="grey", lw=0.5)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "forgetting_decomposition.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="features/drift folder of run_drift_features.sh")
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="results/forgetting")
    args = ap.parse_args()
    rows = []
    for b in args.backbones:
      for seed in args.seeds:
        p0 = os.path.join(args.root, b, "step0.pt")
        if not os.path.exists(p0):
            print(f"[skip] {b}: no {p0}")
            continue
        s0 = load(p0)
        for run in args.runs:
            folder = os.path.join(args.root, b, run, *([] if seed == 42 else [f"seed{seed}"]))
            files = {int(re.search(r"step(\d+)\.pt$", f).group(1)): f for f in glob.glob(os.path.join(folder, "step*.pt"))}
            if not files:
                print(f"[skip] {b}/{run}/seed{seed}: no steps in {folder}")
                continue
            steps = {0: s0, **{k: load(f) for k, f in sorted(files.items())}}
            rows += [dict(r, seed=seed) for r in analyse(b, run, steps)]
            print(f"{b}/{run}/seed{seed}: steps {sorted(files)}", flush=True)
    if not rows:
        sys.exit("no feature files found")
    classes = pd.DataFrame(rows)
    dec = decompose(classes)
    summ = summarise(dec)
    os.makedirs(args.out, exist_ok=True)
    classes.to_csv(os.path.join(args.out, "forgetting_classes.csv"), index=False)
    dec.to_csv(os.path.join(args.out, "forgetting_decomposition.csv"), index=False)
    summ.to_csv(os.path.join(args.out, "forgetting_summary.csv"), index=False)
    with open(os.path.join(args.out, "forgetting_decomposition.md"), "w") as f:
        f.write(markdown(dec) + markdown_seeds(summ))
    figure(summ, args.out)
    print(f"-> {args.out}/forgetting_decomposition.md")


if __name__ == "__main__":
    main()
