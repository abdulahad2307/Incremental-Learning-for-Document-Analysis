"""Closed-form incremental classifier on frozen base-model features (analytic / RanPAC-style), against the classifiers
of the incremental learners.

The backbone stays the base model (step 0). Only two sums are stored, extended once when a class arrives:
    G = sum phi(x) phi(x)^T   (D x D, shared)        C[:, c] = sum_{x in c} phi(x)   (D per class)
and after every step the classifier is the ridge solution W = (G + lambda I)^-1 C[:, known]. This equals the ridge
regression on all training data of all classes seen so far, so the classifier itself cannot forget; what limits it is
how well the frozen base features separate the classes. No documents are stored.

Feature maps phi (on standardised features; mean / std from the base-class training data only):
  ridge    phi(x) = [x, 1]
  ranpac   phi(x) = ReLU(x R), R ~ N(0, 1) of size d x M, fixed (McDonnell et al., RanPAC, NeurIPS 2023);
           repeated for --proj_seeds random projections
  ncm      reference: cosine nearest class mean, class means from the same sums (no G needed)

lambda, two protocols (both reported):
  <method>          chosen at every step on held-out documents of all classes seen so far (20 % of the training
                    documents of each class are kept out of the sums for this); the same protocol as the incremental
                    learners, which select their best epoch on the validation set of all seen classes
  <method>-base     chosen once at the base step (base classes only) and then kept fixed
The final sums use all training documents.

Data: features from tools/eval/extract_step_features.py of the base model with --train_classes all, e.g.
  <root>/<backbone>/step0.pt         (drift features: 200 training and 300 test documents per class)
  $RUN_ROOT/features/<backbone>/base_all_cil.pt   (1000 training documents per class, the full test set)
The learners' own classifiers come from the drift features of their steps (--runs, same 300 test documents per class;
only with the drift step0.pt) and/or from the results table (--table: accuracy on the full test set).

Outputs in --out: closed_form_runs.csv, closed_form_summary.csv, closed_form.md

Examples:
  python tools/eval/closed_form_classifier.py --root $RUN_ROOT/features/drift --out $RUN_ROOT/analysis/closed_form_drift
  python tools/eval/closed_form_classifier.py --root $RUN_ROOT/features/drift --runs \\
      --base '$RUN_ROOT/features/{backbone}/base_all_cil.pt' --table $RUN_ROOT/il_runs.csv --out $RUN_ROOT/analysis/closed_form
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

BASE_CLASSES = ["letter", "form", "email", "handwritten", "advertisement", "scientific_report", "invoice",
                "presentation", "questionnaire", "resume", "memo"]
ORDER = ["scientific_publication", "specification", "file_folder", "news_article", "budget"]
JOINT = {"eaml": "fused", "llmv3": "cls", "llmv3hf": "cls"}
LAMBDAS = 10.0 ** np.arange(-2, 9)
TABLE_METHODS = {"No EVM": "noevm_std", "EVM": "evm_std"}


def load_base(path, modality):
    d = torch.load(path, map_location="cpu", weights_only=False)
    S = d["splits"]

    def feats(split):
        f = S[split]["features"]
        x = np.concatenate([np.asarray(f[m], np.float32) for m in sorted(f)], 1) if modality == "concat" \
            else np.asarray(f[modality], np.float32)
        return x, np.asarray(S[split]["labels"], dtype=object)

    xtr, ytr = feats("train")
    parts = [feats(s) for s in ("test_seen", "test_unseen") if s in S]
    xte, yte = (np.concatenate(p) for p in zip(*parts))
    return xtr, ytr, xte, yte


def accs(pred, labels, old, new):
    ok = pred == labels
    per = {c: ok[labels == c].mean() for c in np.unique(labels)}
    o = [per[c] for c in old if c in per]
    n = [per[c] for c in new if c in per]
    return dict(acc_all=float(np.mean(list(per.values()))), acc_old=float(np.mean(o)) if o else np.nan,
                acc_new=float(np.mean(n)) if n else np.nan)


class ClosedForm:
    """Running sums G, C; the ridge readout for any lambda from one eigendecomposition of G."""

    def __init__(self, dim):
        self.G, self.C, self.classes, self._eig = np.zeros((dim, dim)), {}, [], None

    def add_class(self, c, h):
        self.G += h.T.astype(np.float64) @ h.astype(np.float64)
        self.C[c] = h.sum(0, dtype=np.float64)
        self.classes.append(c)
        self._eig = None

    def add_class_x(self, c, x, phi, chunk=8192):
        """add_class on phi(x), computed chunk-wise (large training sets)."""
        C = np.zeros(self.G.shape[0])
        for i in range(0, len(x), chunk):
            h = phi(x[i:i + chunk]).astype(np.float64)
            self.G += h.T @ h
            C += h.sum(0)
        self.C[c] = C
        self.classes.append(c)
        self._eig = None

    def weights(self, lam):
        if self._eig is None:
            e, V = np.linalg.eigh(self.G)
            self._eig = (np.maximum(e, 0.0), V, V.T @ np.stack([self.C[c] for c in self.classes], 1))
        e, V, VtC = self._eig
        return (V @ (VtC / (e + lam)[:, None])).astype(np.float32)

    def predict(self, x, lam, phi, chunk=4096):
        """Labels of the (standardised) features x; phi is applied chunk-wise to bound the memory."""
        W = self.weights(lam)
        cls = np.asarray(self.classes, dtype=object)
        return np.concatenate([cls[(phi(x[i:i + chunk]) @ W).argmax(1)] for i in range(0, len(x), chunk)])


def make_phi(kind, d, m, seed):
    if kind == "ridge":
        return lambda x: np.hstack([x, np.ones((len(x), 1), np.float32)])
    R = np.random.default_rng(seed).standard_normal((d, m)).astype(np.float32)
    return lambda x: np.maximum(x @ R, 0.0)


def subsample(y, n, seed=0):
    """Indices of n randomly chosen training documents of every class; n=None: all."""
    if n is None:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(y == c))[:n] for c in np.unique(y)])


def holdout(y, frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        val[idx[: max(1, int(len(idx) * frac))]] = True
    return val


def best_lambda(cf, xval, yval, phi):
    scores = [np.mean(cf.predict(xval, lam, phi) == yval) for lam in LAMBDAS]
    return float(LAMBDAS[int(np.argmax(scores))])


def run_closed_form(backbone, base_path, modality, proj_dim, proj_seeds, train_per_class=None):
    xtr, ytr, xte, yte = load_base(base_path, modality)
    idx = subsample(ytr, train_per_class)
    xtr, ytr = xtr[idx], ytr[idx]
    n_train = int(min((ytr == c).sum() for c in BASE_CLASSES + ORDER))
    keep = np.isin(ytr, BASE_CLASSES)
    mu, sd = xtr[keep].mean(0), xtr[keep].std(0) + 1e-6          # base-class statistics only
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    val = holdout(ytr)
    key = dict(backbone=backbone, modality=modality, n_train=n_train)
    rows = []
    for kind, seeds in (("ridge", [0]), ("ranpac", proj_seeds)):
        for seed in seeds:
            phi = make_phi(kind, xtr.shape[1], proj_dim, seed)
            dim = phi(xtr[:1]).shape[1]
            fit, full = ClosedForm(dim), ClosedForm(dim)
            known = []
            for step in range(len(ORDER) + 1):
                for c in (BASE_CLASSES if step == 0 else [ORDER[step - 1]]):
                    fit.add_class_x(c, xtr[(ytr == c) & ~val], phi)
                    full.add_class_x(c, xtr[ytr == c], phi)
                    known.append(c)
                v = val & np.isin(ytr, known)
                lam_step = best_lambda(fit, xtr[v], ytr[v], phi)
                if step == 0:
                    lam_base = lam_step
                te = np.isin(yte, known)
                old, new = (known[:-1], known[-1:]) if step else (known, [])
                for name, lam in ((kind, lam_step), (f"{kind}-base", lam_base)):
                    rows.append(dict(key, method=name, seed=seed, step=step, lam=lam,
                                     **accs(full.predict(xte[te], lam, phi), yte[te], old, new)))
    # NCM on the same frozen features (class means = C / n): the closed-form reference without G
    nrm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    known, means = [], {}
    for step in range(len(ORDER) + 1):
        for c in (BASE_CLASSES if step == 0 else [ORDER[step - 1]]):
            means[c] = xtr[ytr == c].mean(0)
            known.append(c)
        te = np.isin(yte, known)
        pred = np.asarray(known, dtype=object)[(nrm(xte[te]) @ nrm(np.stack([means[c] for c in known])).T).argmax(1)]
        old, new = (known[:-1], known[-1:]) if step else (known, [])
        rows.append(dict(key, method="ncm", seed=0, step=step, lam=np.nan, **accs(pred, yte[te], old, new)))
    return rows


def learners_from_features(root, backbone, runs, seeds, base_path):
    """Own classifiers of the incremental learners on the drift test documents (300 per class)."""
    rows = []
    for run in runs:
        for seed in seeds:
            for step in range(len(ORDER) + 1):
                path = base_path if step == 0 else (os.path.join(root, backbone, run, f"step{step}.pt") if seed == 42
                                                    else os.path.join(root, backbone, run, f"seed{seed}", f"step{step}.pt"))
                if not os.path.exists(path):
                    continue
                d = torch.load(path, map_location="cpu", weights_only=False)
                known = np.asarray(d["known_classes"], dtype=object)
                te = d["splits"]["test_seen"]
                pred = known[np.asarray(te["logits"]).argmax(1)]
                old, new = (list(known[:-1]), [known[-1]]) if step else (list(known), [])
                rows.append(dict(backbone=backbone, method=run, seed=seed, step=step, lam=np.nan, modality="own",
                                 n_train=-1, **accs(pred, np.asarray(te["labels"], dtype=object), old, new)))
    return rows


def learners_from_table(table, backbone, seeds):
    """Final test accuracy per step of No EVM / EVM (standard training) from the results table; the last entry
    per step counts (reruns). Only all-class accuracy is in the table."""
    t = pd.read_csv(table)
    t = t[(t.backbone == backbone) & (t.setting == "CIL") & (t.phase == "final") & (t.split == "Test")
          & (t.strategy == "standard") & t.method.isin(list(TABLE_METHODS)) & t.seed.isin(seeds)]
    t = t.sort_values("timestamp").groupby(["method", "seed", "step"]).tail(1)
    return [dict(backbone=backbone, method=TABLE_METHODS[r.method] + " (table)", seed=int(r.seed), step=int(r.step),
                 lam=np.nan, modality="own", n_train=-1, acc_all=float(r.acc), acc_old=np.nan, acc_new=np.nan)
            for r in t.itertuples()]


def markdown(s, lam):
    lines = ["# Closed-form classifier on frozen base features", "",
             "All-class accuracy (%) after each step, mean over seeds (ridge / ncm: deterministic; ranpac: random "
             "projections; learners: training seeds); last column: accuracy on the class of step 5. n = training "
             "documents per class in the sums. <method>: lambda chosen at every step on held-out documents of the seen "
             "classes; <method>-base: chosen once at the base step. (table): the learners' accuracy on the full test "
             "set from the results table; otherwise on the same test documents as the closed-form rows.", ""]
    for b in sorted(s.backbone.unique()):
        sb = s[s.backbone == b]
        steps = sorted(sb.step.unique())
        lines += [f"## {b}", "", "| method | " + " | ".join(f"step {k}" for k in steps) + " | new, step 5 |",
                  "|---|" + "---|" * (len(steps) + 1)]
        for meth in sb.method.unique():
            r = sb[sb.method == meth].set_index("step")
            cells = [f"{100 * r.loc[k, 'acc_all_mean']:.1f}" if k in r.index else "" for k in steps]
            last = r.loc[steps[-1], "acc_new_mean"] if steps[-1] in r.index else np.nan
            lines.append(f"| {meth} | " + " | ".join(cells) + f" | {'' if np.isnan(last) else f'{100 * last:.1f}'} |")
        lines.append("")
    lines += ["lambda per step (first projection seed): "]
    lines += [f"- {b} / {m}: " + ", ".join(f"{v:g}" for v in g.lam) for (b, m), g in lam]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="drift feature root ($RUN_ROOT/features/drift)")
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--base", default=None, help="base-model feature file, {backbone} is replaced "
                                                 "(default: <root>/{backbone}/step0.pt)")
    ap.add_argument("--modality", default="joint", help="joint (fused / cls), concat (all modalities) or a modality name")
    ap.add_argument("--train_per_class", type=int, nargs="+", default=[None],
                    help="training documents per class in the sums (several values: a curve; default: all in --base)")
    ap.add_argument("--proj_dim", type=int, default=5000, help="ranpac: random projection size M")
    ap.add_argument("--proj_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", nargs="*", default=["noevm_std", "evm_std"],
                    help="learners from their drift features (same test documents as the drift step0.pt); none: --runs")
    ap.add_argument("--table", default=None, help="results table (il_runs.csv): learners' full-test accuracy per step")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2], help="training seeds of the learners")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for b in a.backbones:
        base = (a.base or os.path.join(a.root, "{backbone}", "step0.pt")).replace("{backbone}", b)
        mod = JOINT[b] if a.modality == "joint" else a.modality
        for n in a.train_per_class:
            rows += run_closed_form(b, base, mod, a.proj_dim, a.proj_seeds, n)
            print(f"{b}, n={n}: done", flush=True)
        rows += learners_from_features(a.root, b, a.runs, a.seeds, base)
        if a.table:
            rows += learners_from_table(a.table, b, a.seeds)
    df = pd.DataFrame(rows)
    df["method"] = np.where(df.n_train > 0, df.method + " (n=" + df.n_train.astype(str) + ")", df.method)
    df.to_csv(os.path.join(a.out, "closed_form_runs.csv"), index=False)
    keys, metrics = ["backbone", "method", "step"], ["acc_all", "acc_old", "acc_new"]
    s = df.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    s = s.reset_index()
    s["n"] = df.groupby(keys, sort=False).size().values
    s.to_csv(os.path.join(a.out, "closed_form_summary.csv"), index=False)
    lam = df.dropna(subset=["lam"])
    lam = lam[lam.seed == lam.groupby(["backbone", "method"]).seed.transform("min")].groupby(["backbone", "method"], sort=False)
    md = markdown(s, lam)
    with open(os.path.join(a.out, "closed_form.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
