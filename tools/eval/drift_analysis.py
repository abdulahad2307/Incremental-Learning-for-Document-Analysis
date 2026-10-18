"""Drift analysis (JOURNAL_EXTENSION_PLAN.md, item 13): which modality's representation of the old classes changes
when a backbone learns new classes, and whether that change explains the forgetting.

Input: the per-modality features of every CIL step of a run (scripts/eval/run_drift_features.sh), on the same
documents at every step: 200 base training documents and 300 test documents of each class.
  <root>/<backbone>/step0.pt            base model (shared by all runs)
  <root>/<backbone>/<run>/step<k>.pt    best model of step k of the run
Modalities: EAML image / text / fused, LayoutLMv3 text / visual / cls (fused and cls feed the classifier).

Measures per backbone x run x step k x modality (old classes = the classes learned before step k):
  cka_base     linear CKA between the step-k and the step-0 features of the base classes' test documents
               (1 = the same representation up to rotation and scale)
  margin_rel   per old class: median half-distance from its training documents to the nearest training document of
               another known class (the EVM margin), at step k relative to step 0 with the same classes; mean over
               the old classes. Features are L2-normalised first, so the scale of a model does not matter.
  ncm_old      accuracy of a nearest-class-mean probe (cosine, means of the training documents) on the old classes'
               test documents, among all known classes: what is left of the old classes in that modality
  evt_scale    median Weibull scale of the EVM extreme vectors (MultivariateEVM, L2-normalised features), and
  evt_theta    the mean Gumbel theta between the modalities (one value per step, stored with modality "all")
Forgetting per old class (from the step models' logits on the same test documents): its accuracy after the step it
was learned in (step 0 for the base classes) minus its accuracy at step k. Its link to each modality is the Spearman
correlation, over the old classes, between forgetting and margin loss (1 - margin_rel), with a bootstrap 95 % CI over
classes, per step and over all steps together.

Outputs in --out: drift_measures.csv (long format), drift_classes.csv (per class), drift_correlation.csv,
drift.md (tables) and drift_<backbone>.png (CKA, margin and NCM probe vs. step; forgetting vs. margin loss).

Example:
  python tools/eval/drift_analysis.py --root $RUN_ROOT/features/drift --backbones eaml llmv3hf \
      --runs noevm_std noevm_kd evm_std evmood_std --out $RUN_ROOT/features/drift/analysis
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.evm.mevm_classifier import MultivariateEVM  # noqa: E402


# ------------------------------------------------------------------ measures
def linear_cka(x, y):
    """Linear CKA (Kornblith et al., 2019) of two representations of the same samples (rows)."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0)
    y = np.asarray(y, dtype=np.float64) - np.mean(y, axis=0)
    xy = np.linalg.norm(y.T @ x) ** 2
    return float(xy / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def normalise(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def class_margins(feats, labels, classes):
    """{class: median over its samples of half the distance to the nearest sample of another class in `classes`}."""
    feats, labels = torch.as_tensor(feats), np.asarray(labels, dtype=object)
    out = {}
    for c in classes:
        own, other = np.flatnonzero(labels == c), np.flatnonzero(np.isin(labels, [o for o in classes if o != c]))
        if len(own) == 0 or len(other) == 0:
            continue
        d = torch.cdist(feats[own], feats[other]).min(dim=1).values / 2.0
        out[c] = float(np.median(d.numpy()))
    return out


def ncm_accuracy(train, train_labels, test, test_labels, classes, eval_classes):
    """Nearest-class-mean (cosine) accuracy on the test samples of `eval_classes`, choosing among `classes`."""
    train_labels, test_labels = np.asarray(train_labels, dtype=object), np.asarray(test_labels, dtype=object)
    means = normalise(np.stack([train[train_labels == c].mean(axis=0) for c in classes]))
    mask = np.isin(test_labels, eval_classes)
    if not mask.any():
        return float("nan")
    pred = np.asarray(classes, dtype=object)[(normalise(test[mask]) @ means.T).argmax(axis=1)]
    return float(np.mean(pred == test_labels[mask]))


def spearman_ci(x, y, n_boot=1000, seed=0):
    """Spearman rho of paired values and a bootstrap 95 % CI over the pairs; NaN with fewer than 4 pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan"), float("nan")
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.all(x[i] == x[i][0]) or np.all(y[i] == y[i][0]):
            continue
        boots.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return float(rho), float(lo), float(hi)


# ------------------------------------------------------------------ data
def load_step(path):
    """Features of one step file as {"known", "modalities", "train": (feats, labels), "test": (feats, labels,
    doc ids, predicted class or None)}; test documents of known and not-yet-learned classes together."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    known, S = d["known_classes"], d["splits"]
    mods = list(S["train"]["features"])
    tests = [S[k] for k in ("test_seen", "test_unseen") if k in S]
    feats = {m: np.concatenate([np.asarray(t["features"][m], dtype=np.float32) for t in tests]) for m in mods}
    labels = np.concatenate([np.asarray(t["labels"], dtype=object) for t in tests])
    docs = [x for t in tests for x in t["doc_id"]]
    pred = np.asarray(known, dtype=object)[np.asarray(S["test_seen"]["logits"], dtype=np.float32).argmax(axis=1)]
    pred = np.concatenate([pred, np.full(len(labels) - len(pred), None, dtype=object)])
    return {"known": list(known), "modalities": mods,
            "train": ({m: np.asarray(S["train"]["features"][m], dtype=np.float32) for m in mods},
                      np.asarray(S["train"]["labels"], dtype=object), list(S["train"]["doc_id"])),
            "test": (feats, labels, docs, pred)}


def aligned(a, b):
    """Indices into a's and b's test documents of the documents both have, in the same order."""
    pos = {d: i for i, d in enumerate(b["test"][2])}
    ia = [i for i, d in enumerate(a["test"][2]) if d in pos]
    return np.array(ia), np.array([pos[a["test"][2][i]] for i in ia])


def aligned_train(a, b):
    pos = {d: i for i, d in enumerate(b["train"][2])}
    ia = [i for i, d in enumerate(a["train"][2]) if d in pos]
    return np.array(ia), np.array([pos[a["train"][2][i]] for i in ia])


def class_accuracy(step, classes):
    _, labels, _, pred = step["test"]
    return {c: float(np.mean(pred[labels == c] == c)) for c in classes if np.any(labels == c)}


# ------------------------------------------------------------------ analysis
def analyse_run(backbone, run, steps, evt=True):
    """steps: {k: loaded step} including 0. Returns (measure rows, per-class rows)."""
    s0 = steps[0]
    base = s0["known"]
    mods = s0["modalities"]
    rows, crows = [], []
    learned_at = {c: 0 for c in base}
    acc_when_learned = class_accuracy(s0, base)
    for k in sorted(steps):
        sk = steps[k]
        known = sk["known"]
        for c in known:
            learned_at.setdefault(c, k)
        if k > 0:
            for c, a in class_accuracy(sk, [c for c in known if learned_at[c] == k]).items():
                acc_when_learned[c] = a
        old = [c for c in known if learned_at[c] < k] if k > 0 else list(base)
        acc_k = class_accuracy(sk, old)
        ia, ib = aligned(s0, sk)
        ta, tb = aligned_train(s0, sk)
        base_docs = np.isin(s0["test"][1][ia], base)
        tr_labels = sk["train"][1][tb]
        for m in mods:
            x0, xk = s0["test"][0][m][ia], sk["test"][0][m][ib]
            rows.append((backbone, run, k, m, "cka_base", linear_cka(x0[base_docs], xk[base_docs])))
            m0 = class_margins(normalise(s0["train"][0][m][ta]), tr_labels, known)
            mk = class_margins(normalise(sk["train"][0][m][tb]), tr_labels, known)
            rel = {c: mk[c] / m0[c] for c in old if c in m0 and c in mk and m0[c] > 0}
            rows.append((backbone, run, k, m, "margin_rel", float(np.mean(list(rel.values()))) if rel else np.nan))
            rows.append((backbone, run, k, m, "ncm_old", ncm_accuracy(sk["train"][0][m], sk["train"][1],
                                                                      sk["test"][0][m], sk["test"][1], known, old)))
            for c in old:
                crows.append((backbone, run, k, m, c, learned_at[c], rel.get(c, np.nan),
                              acc_when_learned.get(c, np.nan) - acc_k.get(c, np.nan)))
        if evt:
            mevm = MultivariateEVM(tailsize=0.3, max_fit_samples=50).fit(
                {c: {m: normalise(sk["train"][0][m][sk["train"][1] == c]) for m in mods} for c in known})
            scales = np.concatenate([[ev[1] for ev in evs] for evs in mevm.weibull_models.values()])   # (n_ev, m)
            for j, m in enumerate(mods):
                rows.append((backbone, run, k, m, "evt_scale", float(np.median(scales[:, j]))))
            rows.append((backbone, run, k, "all", "evt_theta", float(np.mean(list(mevm.theta.values())))))
        rows.append((backbone, run, k, "classifier", "acc_old", float(np.mean(list(acc_k.values()))) if acc_k else np.nan))
        print(f"  {backbone} {run} step {k}: " + "  ".join(
            f"{r[3]} cka {r[5]:.3f}" for r in rows if r[1] == run and r[2] == k and r[4] == "cka_base"), flush=True)
    return rows, crows


def correlations(classes):
    """Spearman(forgetting, margin loss) per backbone x run x modality, per step and over all steps."""
    out = []
    for (b, r, m), g in classes.groupby(["backbone", "run", "modality"]):
        g = g.dropna(subset=["margin_rel", "forgetting"])
        g = g[g.step > 0]
        for k, gk in list(g.groupby("step")) + [("all", g)]:
            rho, lo, hi = spearman_ci(gk.forgetting, 1 - gk.margin_rel)
            out.append({"backbone": b, "run": r, "modality": m, "step": k, "n": len(gk), "rho": rho,
                        "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def figures(measures, classes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for b, mb in measures.groupby("backbone"):
        mods = [m for m in mb.modality.unique() if m not in ("all", "classifier")]
        panels = [("cka_base", "CKA to step 0 (base classes)"), ("margin_rel", "Margin of old classes (rel. to step 0)"),
                  ("ncm_old", "NCM probe accuracy, old classes")]
        fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5 * (len(panels) + 1), 4), constrained_layout=True)
        styles = dict(zip(sorted(mb.run.unique()), ["-", "--", ":", "-."] * 3))
        for ax, (meas, title) in zip(axes, panels):
            for (r, m), g in mb[mb.measure == meas].groupby(["run", "modality"]):
                g = g.sort_values("step")
                ax.plot(g.step, g.value, styles[r], marker="o", color=f"C{mods.index(m)}", label=f"{m}, {r}")
            ax.set_title(title)
            ax.set_xlabel("CIL step")
        axes[0].legend(fontsize=7)
        ax = axes[-1]
        cb = classes[(classes.backbone == b) & (classes.step == classes[classes.backbone == b].step.max())]
        for m, g in cb.groupby("modality"):
            ax.scatter(1 - g.margin_rel, g.forgetting, s=12, color=f"C{mods.index(m)}", label=m)
        ax.set_xlabel("margin loss (1 - margin_rel)")
        ax.set_ylabel("forgetting")
        ax.set_title("Last step: forgetting vs. margin loss")
        ax.legend(fontsize=7)
        fig.savefig(os.path.join(out, f"drift_{b}.png"), dpi=150)
        plt.close(fig)


def markdown(measures, corr):
    lines = ["# Drift analysis", "", "Old classes = classes learned before the step; features of the same documents at "
             "every step.", ""]
    for (b, r), g in measures.groupby(["backbone", "run"]):
        lines += [f"## {b}: {r}", ""]
        piv = g.pivot_table(index=["measure", "modality"], columns="step", values="value")
        lines += ["| Measure | Modality | " + " | ".join(f"step {k}" for k in piv.columns) + " |",
                  "|---|---|" + "---|" * len(piv.columns)]
        for (meas, m), row in piv.iterrows():
            lines.append(f"| {meas} | {m} | " + " | ".join("–" if pd.isna(v) else f"{v:.3f}" for v in row) + " |")
        c = corr[(corr.backbone == b) & (corr.run == r) & (corr.step == "all")]
        if len(c):
            lines += ["", "Spearman (forgetting, margin loss) over old classes x steps: " + "; ".join(
                f"{x.modality} {x.rho:.2f} [{x.ci_low:.2f}, {x.ci_high:.2f}] (n={x.n})" for x in c.itertuples())]
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="features/drift folder of run_drift_features.sh")
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--runs", nargs="+", default=["noevm_std", "noevm_kd", "evm_std", "evmood_std"])
    ap.add_argument("--out", default="results/drift")
    ap.add_argument("--no_evt", action="store_true", help="skip the EVT measures (faster)")
    args = ap.parse_args()
    rows, crows = [], []
    for b in args.backbones:
        step0 = os.path.join(args.root, b, "step0.pt")
        if not os.path.exists(step0):
            print(f"[skip] {b}: no {step0}")
            continue
        s0 = load_step(step0)
        for r in args.runs:
            files = {int(re.search(r"step(\d+)\.pt$", f).group(1)): f
                     for f in glob.glob(os.path.join(args.root, b, r, "step*.pt"))}
            if not files:
                print(f"[skip] {b}/{r}: no steps yet")
                continue
            steps = {0: s0, **{k: load_step(f) for k, f in sorted(files.items())}}
            r_rows, r_crows = analyse_run(b, r, steps, evt=not args.no_evt)
            rows += r_rows
            crows += r_crows
    if not rows:
        sys.exit("no feature files found")
    measures = pd.DataFrame(rows, columns=["backbone", "run", "step", "modality", "measure", "value"])
    classes = pd.DataFrame(crows, columns=["backbone", "run", "step", "modality", "class", "learned_at",
                                           "margin_rel", "forgetting"])
    corr = correlations(classes)
    os.makedirs(args.out, exist_ok=True)
    measures.to_csv(os.path.join(args.out, "drift_measures.csv"), index=False)
    classes.to_csv(os.path.join(args.out, "drift_classes.csv"), index=False)
    corr.to_csv(os.path.join(args.out, "drift_correlation.csv"), index=False)
    with open(os.path.join(args.out, "drift.md"), "w") as f:
        f.write(markdown(measures, corr))
    figures(measures, classes, args.out)
    print(f"-> {args.out}/drift.md")


if __name__ == "__main__":
    main()
