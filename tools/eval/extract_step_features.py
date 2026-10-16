"""Cache the per-modality features of one checkpoint for the post-hoc EVT comparison (tools/eval/posthoc_evt.py).

For a CIL checkpoint (base model = step 0, or the best model of a step) it saves, with model.modality_features():
  train        --train_per_class (default 200) training documents per known class, from the base models' training
               documents (utils/data_subset.base_training_documents) - the data EVM / MEVM are fitted on;
               --train_classes all: of all classes (the frozen-backbone NCM baseline needs the later classes' means)
  test_seen    all test documents of the classes the checkpoint knows (closed-set accuracy, the knowns)
  test_unseen  all test documents of the classes it has not learned yet (the unknowns of the open-set metrics)
Each split holds {modality: float16 (N, d)}, the classifier logits, the class names and the document IDs.
Modalities: EAML image / text / fused, LayoutLMv3 (Custom or pre-trained) text / visual / cls.

Example (paths from scripts/config.sh; scripts/eval/run_extract_features.sh runs it on a GPU node):
  python tools/eval/extract_step_features.py --backbone eaml --checkpoint $EAML_BASE_11 \
      --known_classes $CIL_BASE_CLASSES --all_classes $ALL_CLASSES --data_dir $RVL_DIR --ocr $EAML_OCR_RVL \
      --out $RUN_ROOT/features/eaml/base_cil.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_subset import select_per_class, DEFAULT_SEED  # noqa: E402

TRAIN_PER_CLASS = 200


def eaml_loaders(args, known, unknown, train_classes):
    from utils.class_IL.dataloader_utils import EAMLClassILDataset, common_transform, eaml_collate_fn

    def make(split, classes, n=None):
        ds = EAMLClassILDataset(os.path.join(args.data_dir, split), classes, ocr_data_path=args.ocr,
                                transform=common_transform, training_docs=(split == "train"))
        if n is not None:   # n of each class's base training documents, chosen reproducibly
            keep = set()
            for c in classes:
                keep.update(select_per_class([p for p, _, k in ds.samples if k == c], n, DEFAULT_SEED, c))
            ds.samples = [s for s in ds.samples if s[0] in keep]
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=eaml_collate_fn), \
            [s[0] for s in ds.samples], list(classes)
    return {"train": make("train", train_classes, args.train_per_class), "test_seen": make("test", known),
            "test_unseen": make("test", unknown) if unknown else None}


def llmv3_loaders(args, known, unknown, train_classes):
    from utils.llmv3.llmv3_incremental_dataloader import get_incremental_dataloader

    def make(split, classes, n=None):
        loader = get_incremental_dataloader(dataset_name="rvl_cdip", ocr_tensor_file=args.ocr, classes=classes,
                                            label_classes=classes, image_dir=args.data_dir, split=split,
                                            batch_size=args.batch_size, images_per_class=n, seed=DEFAULT_SEED)
        ds = loader.dataset
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                          collate_fn=loader.collate_fn), [str(s[0]) for s in ds.samples], list(classes)
    return {"train": make("train", train_classes, args.train_per_class), "test_seen": make("test", known),
            "test_unseen": make("test", unknown) if unknown else None}


def load_model(args, n_classes, device):
    if args.backbone == "eaml":
        from utils.eaml.eaml_model import EAMLModel
        model = EAMLModel(num_classes=n_classes)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        return model.to(device).eval()
    from utils.llmv3.llmv3_model_loader import load_llmv3_checkpoint
    return load_llmv3_checkpoint(args.checkpoint, n_classes, device)[0].eval()


@torch.no_grad()
def extract(model, loader, paths, classes, device, backbone):
    feats, logits, labels = {}, [], []
    for batch in loader:
        if backbone == "eaml":
            mf = model.modality_features(batch["images"].to(device),
                                         texts={k: v.to(device) for k, v in batch["texts"].items()})
            z = model.fusion_classifier(mf["fused"])
        else:
            mf = model.modality_features(*(batch[k].to(device) for k in ("input_ids", "bbox", "attention_mask",
                                                                            "pixel_values")))
            z = model.classifier(mf["cls"])
        for k, v in mf.items():
            feats.setdefault(k, []).append(v.float().cpu())
        logits.append(z.float().cpu())
        labels.append(torch.as_tensor(batch["labels"]).cpu().clone())
    y = torch.cat(labels).numpy()
    return {"features": {k: torch.cat(v).numpy().astype(np.float16) for k, v in feats.items()},
            "logits": torch.cat(logits).numpy().astype(np.float16),
            "labels": np.asarray(classes, dtype=object)[y].tolist(),
            "doc_id": [os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p) for p in paths][:len(y)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=["eaml", "llmv3"], required=True, help="llmv3 = Custom or pre-trained "
                    "(the checkpoint says which; set LLMV3_MODEL=hf for the pre-trained one)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--known_classes", required=True, help="the checkpoint's output classes, in logit order")
    ap.add_argument("--all_classes", required=True, help="all RVL-CDIP classes (the rest are the unknowns)")
    ap.add_argument("--data_dir", required=True, help="RVL-CDIP with train/ val/ test/ folders")
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_per_class", type=int, default=TRAIN_PER_CLASS)
    ap.add_argument("--train_classes", choices=["known", "all"], default="known",
                    help="classes of the train split: the checkpoint's classes, or all classes (NCM baseline)")
    args = ap.parse_args()

    known = [c.strip() for c in args.known_classes.split(",")]
    unknown = [c.strip() for c in args.all_classes.split(",") if c.strip() not in known]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, len(known), device)
    train_classes = known + unknown if args.train_classes == "all" else known
    loaders = (eaml_loaders if args.backbone == "eaml" else llmv3_loaders)(args, known, unknown, train_classes)
    out = {"meta": {"checkpoint": os.path.abspath(args.checkpoint), "backbone": args.backbone,
                    "llmv3_model": os.environ.get("LLMV3_MODEL", "")},
           "known_classes": known, "unknown_classes": unknown, "splits": {}}
    for name, item in loaders.items():
        if item is None:
            continue
        out["splits"][name] = extract(model, *item, device, args.backbone)
        print(f"{name}: {len(out['splits'][name]['labels'])} documents, modalities "
              f"{list(out['splits'][name]['features'])}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
