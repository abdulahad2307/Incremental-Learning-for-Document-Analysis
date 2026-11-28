"""Classifier recalibration after the last CIL step with the learner's actual replay memory: nearest class mean (NCM)
on the memory, and weight aligning (WA) of the model's own classifier.

Replay memory. Every CIL step is its own job and starts with an empty memory: before step k, herding selects the
exemplars of every previous class with the checkpoint of step k-1 (EAML: utils/class_IL/cil_utils.ExemplarManager on a
fixed pool of 1000 candidates per class, random.Random(0); LayoutLMv3: utils/llmv3/llmv3_incremental_utils.
ExemplarHandler on the --exemplar_pool base training documents per class, a fixed selection). So the memory a learner holds
after step K is the herding selection with the step K-1 checkpoint, which this script rebuilds with the same classes
and arguments (EAML: the manager itself; LayoutLMv3: the handler's selection in one pass over the pool instead of one
pass per class, with the same single-document feature extraction and the same order). Fine-tuning has no memory; it
gets the selection its step K-1 model would have made, i.e. what it would have stored with a memory.

Readouts on the step-K features (the drift features of tools/eval/extract_step_features.py: 300 test documents per
class; the new class's 200 training documents):
  own           the model's classifier (the bias centring of the runs does not change any prediction)
  wa            weight aligning (Zhao et al., CVPR 2020): the new class's row (weight and bias) of the last linear layer
                rescaled by mean ||w_old|| / mean ||w_new||
  ncm_memory    cosine NCM, old-class means from the rebuilt memory (20 per class) on the step-K model, new class from
                its 200 training documents (as tools/eval/recovery_curve.py)
  ncm_random    the same with 20 randomly drawn training documents per old class (recovery_curve.py, m = 20, one draw)

Outputs in --out: recalibration_runs.csv, recalibration_summary.csv, recalibration.md, memory/<backbone>_<run>_seed<s>.json
(the rebuilt memory: document IDs per class).

Example (scripts/eval/run_recalibration.sh runs it on a GPU node):
  python tools/eval/recalibration.py --backbone eaml --drift $RUN_ROOT/features/drift --data_dir $RVL_DIR \\
      --ocr $EAML_OCR_RVL --out $RUN_ROOT/analysis/recalibration
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_CLASSES = ["letter", "form", "email", "handwritten", "advertisement", "scientific_report", "invoice",
                "presentation", "questionnaire", "resume", "memo"]
ORDER = ["scientific_publication", "specification", "file_folder", "news_article", "budget"]
JOINT = {"eaml": "fused", "llmv3hf": "cls"}
PER_CLASS = 20


def drift_path(root, backbone, run, seed, step):
    base = os.path.join(root, backbone, run) if seed == 42 else os.path.join(root, backbone, run, f"seed{seed}")
    return os.path.join(base, f"step{step}.pt")


def checkpoints(step_ckpt, prev_class):
    """The step K-1 checkpoint next to the step K checkpoint (best model of the step that added prev_class)."""
    d = os.path.dirname(step_ckpt)
    cand = [f for f in glob.glob(os.path.join(d, f"*{prev_class}*")) if "best" in os.path.basename(f)
            and "evm_state" not in os.path.basename(f) and f.endswith((".pt", ".pth"))]
    if len(cand) != 1:
        raise FileNotFoundError(f"step K-1 checkpoint for {prev_class} in {d}: {cand}")
    return cand[0]


# ------------------------------------------------------------------ EAML
def eaml_memory(args, prev_ckpt, previous, new_class, device):
    from utils.class_IL.cil_utils import ExemplarManager, fill_exemplar_memory
    from utils.class_IL.dataloader_utils import EAMLClassILDataset, common_transform
    ds = EAMLClassILDataset(data_dir=os.path.join(args.data_dir, "train"), training_docs=True,
                            current_classes=previous + [new_class], ocr_data_path=args.ocr, transform=common_transform)
    mgr = ExemplarManager(max_exemplars=320, selection_strategy="herding")
    fill_exemplar_memory(mgr, ds, previous, prev_ckpt, len(previous), device)
    return {c: list(mgr.exemplars[c]) for c in previous}, ds


@torch.no_grad()
def eaml_features(ckpt, n_classes, samples, ds, device):
    from torch.utils.data import DataLoader
    from utils.class_IL.dataloader_utils import eaml_collate_fn
    from utils.eaml.eaml_model import EAMLModel
    model = EAMLModel(num_classes=n_classes)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model_state_dict", st))
    model = model.to(device).eval()
    ds.samples = list(samples)
    feats = []
    for b in DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=eaml_collate_fn):
        mf = model.modality_features(b["images"].to(device), texts={k: v.to(device) for k, v in b["texts"].items()})
        feats.append(mf["fused"].float().cpu())
    head = model.fusion_classifier
    return torch.cat(feats).numpy(), head.weight.detach().float().cpu().numpy(), head.bias.detach().float().cpu().numpy()


# ------------------------------------------------------------------ LayoutLMv3 (pre-trained)
@torch.no_grad()
def llmv3_memory(args, prev_ckpt, previous, new_class, seed, device):
    """ExemplarHandler.update_exemplars (herding) in one pass over the pool: same features (one document at a time,
    model.extract_features), same per-class order (dataset index order), same selection rule."""
    from utils.llmv3.llmv3_incremental_dataloader import get_incremental_dataloader
    from utils.llmv3.llmv3_model_loader import load_llmv3_checkpoint
    label_space = previous + [new_class]
    ds = get_incremental_dataloader(dataset_name="rvl_cdip", ocr_tensor_file=args.ocr, classes=previous,
                                    label_classes=label_space, image_dir=args.data_dir, split="train",
                                    batch_size=1, images_per_class=args.exemplar_pool, seed=seed).dataset
    model = load_llmv3_checkpoint(prev_ckpt, len(previous), device)[0].eval()
    feats, idx = {}, {}
    for i in range(len(ds)):
        s = ds[i]
        c = int(s["labels"].item())
        inputs = {k: v.unsqueeze(0).to(device) for k, v in s.items() if k not in ("labels", "image_id")}
        feats.setdefault(c, []).append(model.extract_features(**inputs).squeeze(0).cpu())
        idx.setdefault(c, []).append(i)
    del model
    torch.cuda.empty_cache()
    memory = {}
    for ci, cname in enumerate(previous):
        f, ind = torch.stack(feats[ci]), list(idx[ci])
        mean, sel, acc = f.mean(dim=0), [], f.clone()
        for _ in range(min(PER_CLASS, len(ind))):                       # ExemplarHandler.update_exemplars
            j = torch.argmin(torch.norm(mean - acc, dim=1)).item()
            sel.append(ind[j])
            mean = (mean * len(sel) + acc[j]) / (len(sel) + 1)
            acc = torch.cat([acc[:j], acc[j + 1:]], dim=0)
            ind.pop(j)
        memory[cname] = sel
    return memory, ds


@torch.no_grad()
def llmv3_features(ckpt, n_classes, indices, ds, device):
    from utils.llmv3.llmv3_model_loader import load_llmv3_checkpoint
    model = load_llmv3_checkpoint(ckpt, n_classes, device)[0].eval()
    feats = []
    for i in indices:
        s = ds[i]
        mf = model.modality_features(*(s[k].unsqueeze(0).to(device) for k in ("input_ids", "bbox", "attention_mask",
                                                                              "pixel_values")))
        feats.append(mf["cls"].float().cpu())
    head = model.classifier
    return torch.cat(feats).numpy(), head.weight.detach().float().cpu().numpy(), head.bias.detach().float().cpu().numpy()


def doc_ids_llmv3(ds, indices):
    out = []
    for i in indices:
        s = ds.samples[i] if hasattr(ds, "samples") else None
        out.append(str(s[0]) if s is not None else str(i))
    return out


# ------------------------------------------------------------------ readouts
def accs(pred, labels, old, new):
    ok = pred == labels
    per = {c: ok[labels == c].mean() for c in np.unique(labels)}
    return dict(acc_all=float(np.mean(list(per.values()))), acc_old=float(np.mean([per[c] for c in old])),
                acc_new=float(np.mean([per[c] for c in new])))


def ncm(means_x, means_y, xte):
    nrm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    cls = list(dict.fromkeys(means_y))
    m = nrm(np.stack([nrm(means_x[means_y == c]).mean(0) for c in cls]))
    return np.asarray(cls, dtype=object)[(nrm(xte) @ m.T).argmax(1)]


def run_one(args, run, seed, device):
    joint = JOINT[args.backbone]
    d = torch.load(drift_path(args.drift, args.backbone, run, seed, len(ORDER)), map_location="cpu", weights_only=False)
    known = list(d["known_classes"])
    assert known == BASE_CLASSES + ORDER, known
    previous, new_class = known[:-1], known[-1]
    step_ckpt = d["meta"]["checkpoint"]
    prev_ckpt = checkpoints(step_ckpt, ORDER[-2])
    tr, te = d["splits"]["train"], d["splits"]["test_seen"]
    trx, try_ = np.asarray(tr["features"][joint], np.float32), np.asarray(tr["labels"], dtype=object)
    tex, tey = np.asarray(te["features"][joint], np.float32), np.asarray(te["labels"], dtype=object)
    own_logits = np.asarray(te["logits"], np.float32)
    print(f"{args.backbone}/{run}/seed{seed}: memory from {prev_ckpt}, features from {step_ckpt}", flush=True)

    if args.backbone == "eaml":
        memory, ds = eaml_memory(args, prev_ckpt, previous, new_class, device)
        samples = [s for c in previous for s in memory[c]]
        mx, W, b = eaml_features(step_ckpt, len(known), samples, ds, device)
        my = np.asarray([s[2] for s in samples], dtype=object)
        ids = {c: [os.path.basename(os.path.dirname(s[0])) + "/" + os.path.basename(s[0]) for s in memory[c]]
               for c in previous}
    else:
        memory, ds = llmv3_memory(args, prev_ckpt, previous, new_class, seed, device)
        order = [i for c in previous for i in memory[c]]
        mx, W, b = llmv3_features(step_ckpt, len(known), order, ds, device)
        my = np.asarray([c for c in previous for _ in memory[c]], dtype=object)
        ids = {c: doc_ids_llmv3(ds, memory[c]) for c in previous}

    # own classifier from the step-K head on the same features (check against the stored logits)
    z = tex @ W.T + b
    agree = float(np.mean(z.argmax(1) == own_logits.argmax(1)))
    norms = np.linalg.norm(W, axis=1)
    gamma = float(norms[:-1].mean() / norms[-1])
    Wa, ba = W.copy(), b.copy()
    Wa[-1] *= gamma
    ba[-1] *= gamma
    K = np.asarray(known, dtype=object)
    newx = trx[try_ == new_class]
    rng = np.random.default_rng(1000 * 0 + PER_CLASS)              # recovery_curve.py, m = 20, first draw
    ridx = np.concatenate([rng.permutation(np.flatnonzero(try_ == c))[:PER_CLASS] for c in previous])
    key = dict(backbone=args.backbone, run=run, seed=seed, wa_gamma=gamma, own_agreement=agree,
               memory_size=len(my))
    rows = [dict(key, readout="own", **accs(K[own_logits.argmax(1)], tey, previous, [new_class])),
            dict(key, readout="wa", **accs(K[(tex @ Wa.T + ba).argmax(1)], tey, previous, [new_class])),
            dict(key, readout="ncm_memory", **accs(ncm(np.concatenate([mx, newx]), np.concatenate(
                [my, np.full(len(newx), new_class, dtype=object)]), tex), tey, previous, [new_class])),
            dict(key, readout="ncm_random", **accs(ncm(np.concatenate([trx[ridx], newx]), np.concatenate(
                [try_[ridx], np.full(len(newx), new_class, dtype=object)]), tex), tey, previous, [new_class]))]
    print("  " + ", ".join(f"{r['readout']} {100 * r['acc_all']:.1f}" for r in rows)
          + f" | WA gamma {gamma:.3f}, own head vs stored logits {100 * agree:.1f} %", flush=True)
    return rows, ids


def markdown(s):
    pc = lambda m, sd: f"{100 * m:.1f}" + ("" if pd.isna(sd) else f"±{100 * sd:.1f}")
    lines = ["# Classifier recalibration after step 5 with the actual replay memory", "",
             "Accuracy (%) on 300 test documents per class, mean ± std over seeds (all / old / new class). own: the "
             "model's classifier; wa: weight aligning; ncm_memory: cosine NCM on the rebuilt replay memory (20 per old "
             "class); ncm_random: 20 random training documents per old class (recovery_curve.py, m = 20).", ""]
    for b, g in s.groupby("backbone"):
        lines += [f"## {b}", "", "| run | readout | all | old | new | seeds |", "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.run} | {r.readout} | {pc(r.acc_all_mean, r.acc_all_std)} | "
                         f"{pc(r.acc_old_mean, r.acc_old_std)} | {pc(r.acc_new_mean, r.acc_new_std)} | {r.n} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=["eaml", "llmv3hf"], required=True)
    ap.add_argument("--drift", required=True, help="drift feature root ($RUN_ROOT/features/drift)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--runs", nargs="+", default=["noevm_std", "evm_std", "finetune_std"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--exemplar_pool", type=int, default=1000, help="LayoutLMv3: candidate pool per class (as trained)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(a.out, "memory"), exist_ok=True)
    rows = []
    for run in a.runs:
        for seed in a.seeds:
            if not os.path.exists(drift_path(a.drift, a.backbone, run, seed, len(ORDER))):
                print(f"[skip] {a.backbone}/{run}/seed{seed}: no drift features")
                continue
            r, ids = run_one(a, run, seed, device)
            rows += r
            with open(os.path.join(a.out, "memory", f"{a.backbone}_{run}_seed{seed}.json"), "w") as f:
                json.dump(ids, f, indent=1)
    df = pd.DataFrame(rows)
    path = os.path.join(a.out, "recalibration_runs.csv")
    if os.path.exists(path):                                       # the other backbone's rows stay
        old = pd.read_csv(path)
        df = pd.concat([old[old.backbone != a.backbone], df], ignore_index=True)
    df.to_csv(path, index=False)
    keys = ["backbone", "run", "readout"]
    s = df.groupby(keys, sort=False)[["acc_all", "acc_old", "acc_new"]].agg(["mean", "std"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    s = s.reset_index()
    s["n"] = df.groupby(keys, sort=False).size().values
    s.to_csv(os.path.join(a.out, "recalibration_summary.csv"), index=False)
    md = markdown(s)
    with open(os.path.join(a.out, "recalibration.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
