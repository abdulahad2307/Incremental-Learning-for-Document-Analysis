"""Closed-form classifier on the features of the incrementally trained backbones instead of the frozen base model.

Question: does training the backbone in the steps give better features for a closed-form classifier, or is it as well
to freeze it after the base step? On the drift features (scripts/eval/run_drift_features.sh: 200 training and 300 test
documents per class, the same documents at every step), for each run / seed / step k and three feature sources:

  frozen   base model features (step0.pt) for training and test: the frozen-backbone classifier of
           closed_form_classifier.py, on the same documents
  refit    step-k features for training and test, sums of all classes recomputed with the step-k backbone (needs the
           old classes' documents: an upper bound of what the trained features allow)
  stale    the sums of each class are computed when it arrives, with the backbone of that step (base classes: step 0,
           class j: step j), and the classifier is applied to step-k test features. What an incremental system with a
           trained backbone and stored sums would actually have; it suffers from feature drift.

Readout: ridge on [x, 1] or RanPAC (ReLU random projection, --proj_dim), lambda chosen per step on 20 % held-out
training documents of the seen classes (as in closed_form_classifier.py; for stale on the held-out documents' features
of the current backbone). Features are standardised with the base-class training statistics of the backbone that
computed the sums of the base classes (step 0 for frozen / stale, step k for refit).

Outputs in --out: closed_form_trained_runs.csv, closed_form_trained.md

Example:
  python tools/eval/closed_form_trained_features.py --root $RUN_ROOT/features/drift --out $RUN_ROOT/analysis/closed_form_trained
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from closed_form_classifier import BASE_CLASSES, ORDER, JOINT, ClosedForm, accs, best_lambda, holdout, make_phi


def load_step(path, modality):
    """Train (all 16 classes) and test (seen + unseen) features, labels and document IDs of one drift file."""
    S = torch.load(path, map_location="cpu", weights_only=False)["splits"]
    tr = S["train"]
    te = [S[s] for s in ("test_seen", "test_unseen") if s in S]
    order_tr = np.argsort(np.asarray(tr["doc_id"], dtype=str), kind="stable")
    xte = np.concatenate([np.asarray(t["features"][modality], np.float32) for t in te])
    yte = np.concatenate([np.asarray(t["labels"], dtype=object) for t in te])
    ids = np.concatenate([np.asarray(t["doc_id"], dtype=str) for t in te])
    order_te = np.argsort(ids, kind="stable")
    return (np.asarray(tr["features"][modality], np.float32)[order_tr], np.asarray(tr["labels"], dtype=object)[order_tr],
            xte[order_te], yte[order_te])


def step_path(root, backbone, run, seed, step):
    if step == 0:
        return os.path.join(root, backbone, "step0.pt")
    base = os.path.join(root, backbone, run) if seed == 42 else os.path.join(root, backbone, run, f"seed{seed}")
    return os.path.join(base, f"step{step}.pt")


def standardise(x, y):
    keep = np.isin(y, BASE_CLASSES)
    return x[keep].mean(0), x[keep].std(0) + 1e-6


def evaluate_run(root, backbone, run, seed, kind, proj_dim, proj_seed):
    mod = JOINT[backbone]
    paths = [step_path(root, backbone, run, seed, k) for k in range(len(ORDER) + 1)]
    if not all(os.path.exists(p) for p in paths):
        return []
    feats = [load_step(p, mod) for p in paths]                 # same documents, sorted by document ID
    ytr, yte = feats[0][1], feats[0][3]
    assert all((f[1] == ytr).all() and (f[3] == yte).all() for f in feats), "drift files hold different documents"
    val = holdout(ytr)
    d = feats[0][0].shape[1]
    phi = make_phi(kind, d, proj_dim, proj_seed)
    mu0, sd0 = standardise(feats[0][0], ytr)
    z = lambda k, mu, sd: ((feats[k][0] - mu) / sd, (feats[k][2] - mu) / sd)
    rows = []
    key = dict(backbone=backbone, run=run, seed=seed, readout=kind)
    # stale sums, built step by step with the backbone of the step in which a class arrives
    stale_fit, stale_full = ClosedForm(phi(feats[0][0][:1]).shape[1]), ClosedForm(phi(feats[0][0][:1]).shape[1])
    known = []
    for k in range(len(ORDER) + 1):
        new = BASE_CLASSES if k == 0 else [ORDER[k - 1]]
        xk_tr, xk_te = z(k, mu0, sd0)
        hk = phi(xk_tr)
        for c in new:
            stale_fit.add_class(c, hk[(ytr == c) & ~val])
            stale_full.add_class(c, hk[ytr == c])
        known += new
        te, v = np.isin(yte, known), val & np.isin(ytr, known)
        old, newc = (known[:-1], known[-1:]) if k else (known, [])
        x0_tr, x0_te = z(0, mu0, sd0)
        muk, sdk = standardise(feats[k][0], ytr)
        xr_tr, xr_te = z(k, muk, sdk)
        for source, (xtr, xte, fit_full) in {
            "frozen": (x0_tr, x0_te, None),
            "refit": (xr_tr, xr_te, None),
            "stale": (xk_tr, xk_te, (stale_fit, stale_full)),
        }.items():
            if fit_full is None:                                  # sums of all seen classes with this backbone
                h = phi(xtr)
                fit, full = ClosedForm(h.shape[1]), ClosedForm(h.shape[1])
                for c in known:
                    fit.add_class(c, h[(ytr == c) & ~val])
                    full.add_class(c, h[ytr == c])
            else:
                fit, full = fit_full
            lam = best_lambda(fit, xtr[v], ytr[v], phi)
            rows.append(dict(key, source=source, step=k, lam=lam, **accs(full.predict(xte[te], lam, phi), yte[te], old, newc)))
        # the run's own classifier on the same documents
        if k:
            S = torch.load(paths[k], map_location="cpu", weights_only=False)
            kn = np.asarray(S["known_classes"], dtype=object)
            t = S["splits"]["test_seen"]
            rows.append(dict(key, source="own classifier", step=k, lam=np.nan,
                             **accs(kn[np.asarray(t["logits"]).argmax(1)], np.asarray(t["labels"], dtype=object), old, newc)))
    return rows


def markdown(df):
    s = df[df.step == len(ORDER)].groupby(["backbone", "readout", "run", "source"])[["acc_all", "acc_new"]].agg(["mean", "std", "size"])
    lines = ["# Closed-form classifier on trained vs frozen backbones", "",
             "All-class accuracy (%) after step 5 (300 test documents per class; mean +- std over seeds, n seeds), "
             "in brackets the accuracy on the class of step 5. frozen: base-model features; refit: step-5 features, "
             "sums of all classes recomputed (upper bound); stale: sums from the backbone of the step in which each "
             "class arrived; own classifier: the run's trained classifier.", ""]
    for (b, ro), g in s.groupby(level=[0, 1]):
        runs = g.index.get_level_values(2).unique()
        sources = ["frozen", "refit", "stale", "own classifier"]
        lines += [f"## {b}, {ro}", "", "| run | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
        for r in runs:
            cells = []
            for src in sources:
                if (b, ro, r, src) not in g.index:
                    cells.append("")
                    continue
                row = g.loc[(b, ro, r, src)]
                sd = row[("acc_all", "std")]
                cells.append(f"{100 * row[('acc_all', 'mean')]:.1f}" + (f" ± {100 * sd:.1f}" if sd == sd else "")
                             + f" ({100 * row[('acc_new', 'mean')]:.0f})" + f" n={int(row[('acc_all', 'size')])}")
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--runs", nargs="+", default=["noevm_std", "evm_std", "finetune_std", "noevm_std_full", "evm_std_full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--readouts", nargs="+", default=["ridge", "ranpac"], choices=["ridge", "ranpac"])
    ap.add_argument("--proj_dim", type=int, default=2000, help="ranpac projection size (smaller than in "
                                                               "closed_form_classifier.py to keep the many fits fast)")
    ap.add_argument("--proj_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache")                     # one csv per finished run: a restart skips them
    os.makedirs(cache, exist_ok=True)
    frames = []
    for b in a.backbones:
        for run in a.runs:
            for seed in a.seeds:
                for ro in a.readouts:
                    f = os.path.join(cache, f"{b}_{run}_seed{seed}_{ro}_m{a.proj_dim}.csv")
                    if not os.path.exists(f):
                        r = evaluate_run(a.root, b, run, seed, ro, a.proj_dim, a.proj_seed)
                        if not r:
                            continue
                        pd.DataFrame(r).to_csv(f, index=False)
                        print(f"{b} {run} seed {seed} {ro}: done", flush=True)
                    frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(a.out, "closed_form_trained_runs.csv"), index=False)
    md = markdown(df)
    with open(os.path.join(a.out, "closed_form_trained.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
