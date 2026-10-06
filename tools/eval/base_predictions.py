"""Step-0 predictions: evaluate a base model once on the test sets of its incremental setting.

The incremental scripts save one predictions file per step (utils/eval/predictions.py); this tool adds the row before
the first step, which the accuracy matrix (forgetting, BWT) needs. It is shared by all methods of a backbone.

  CIL (11-class base model): "seen" = test documents of the base classes, "unseen" = those of the classes added later
  DIL (16-class base model): "rvl" = RVL-CDIP test, "tobacco" = Tobacco-3482 test (built as the DIL scripts build it;
                             its Note / Report documents are kept but cannot be predicted by the base model)

The test loaders are the ones the incremental scripts use, so the documents match their predictions files.
Output: <dir of the checkpoint>/predictions/base_<cil|dil>_seed<seed>.pt (or --out_dir).

Examples (scripts/eval/run_base_predictions.sh runs all four of a backbone):
  python tools/eval/base_predictions.py --backbone eaml --setting CIL --checkpoint $EAML_BASE_11 \
      --base_classes $CIL_BASE_CLASSES --all_classes $ALL_CLASSES --data_root $DATA_ROOT --ocr $EAML_OCR_RVL
  python tools/eval/base_predictions.py --backbone llmv3 --setting DIL --checkpoint $LLMV3_BASE_16 \
      --all_classes $DIL_CLASSES --base_classes $ALL_CLASSES --data_root $DATA_ROOT --ocr $LLMV3_OCR_RVL $LLMV3_OCR_TOB
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.eval.predictions import save_predictions  # noqa: E402
from utils.run_log import _git_commit  # noqa: E402
from utils.seed import set_seed  # noqa: E402

RVL_DOMAIN, TOB_DOMAIN = "all_prepdataset", "Tobacco3482-jpg"   # folder names under --data_root (scripts/config.sh)


def eaml_loaders(args, base_classes, all_classes):
    if args.setting == "CIL":
        from utils.class_IL.dataloader_utils import get_class_il_loader
        test_dir = os.path.join(args.data_root, RVL_DOMAIN, "test")
        future = [c for c in all_classes if c not in base_classes]
        return {"seen": (get_class_il_loader("eaml", test_dir, base_classes, args.batch_size, ocr_data=args.ocr[0]),
                         base_classes),
                "unseen": (get_class_il_loader("eaml", test_dir, future, args.batch_size, ocr_data=args.ocr[0]),
                           future)}
    from utils.domain_IL.dil_dataloader import DILDataLoader
    loader = DILDataLoader(data_root=args.data_root, domain_list=[RVL_DOMAIN, TOB_DOMAIN],
                           batch_size=args.batch_size, img_size=(229, 229), num_workers=4,
                           ocr_tensor_dirs={RVL_DOMAIN: args.ocr[0], TOB_DOMAIN: args.ocr[1]},
                           class_to_idx={c: i for i, c in enumerate(all_classes)})
    test = loader.get_domain_loaders("test")
    return {"rvl": (test.get(RVL_DOMAIN), all_classes), "tobacco": (test.get(TOB_DOMAIN), all_classes)}


def llmv3_loaders(args, base_classes, all_classes):
    from utils.llmv3.llmv3_incremental_dataloader import get_incremental_dataloader

    def make(dataset, ocr, domain, classes, label_classes):
        return get_incremental_dataloader(dataset_name=dataset, ocr_tensor_file=ocr, classes=classes,
                                          label_classes=label_classes, image_dir=os.path.join(args.data_root, domain),
                                          split="test", batch_size=args.batch_size, images_per_class=None,
                                          seed=args.seed)
    if args.setting == "CIL":
        future = [c for c in all_classes if c not in base_classes]
        return {"seen": (make("rvl_cdip", args.ocr[0], RVL_DOMAIN, base_classes, base_classes), base_classes),
                "unseen": (make("rvl_cdip", args.ocr[0], RVL_DOMAIN, future, future), future)}
    return {"rvl": (make("rvl_cdip", args.ocr[0], RVL_DOMAIN, all_classes, all_classes), all_classes),
            "tobacco": (make("tobacco3482", args.ocr[1], TOB_DOMAIN, all_classes, all_classes), all_classes)}


def load_model(args, num_outputs, device):
    if args.backbone == "eaml":
        from utils.eaml.eaml_model import EAMLModel
        model = EAMLModel(num_classes=num_outputs)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        return model.to(device)
    from utils.llmv3.llmv3_model_loader import load_llmv3_checkpoint
    return load_llmv3_checkpoint(args.checkpoint, num_outputs, device)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", choices=["eaml", "llmv3"], required=True)
    ap.add_argument("--setting", choices=["CIL", "DIL"], required=True)
    ap.add_argument("--checkpoint", required=True, help="base model checkpoint")
    ap.add_argument("--base_classes", required=True,
                    help="comma-separated output classes of the base model, in its logit order")
    ap.add_argument("--all_classes", required=True,
                    help="CIL: all 16 classes; DIL: the global class list of the DIL scripts (DIL_CLASSES)")
    ap.add_argument("--data_root", required=True, help="DATA_ROOT of scripts/config.sh")
    ap.add_argument("--ocr", nargs="+", required=True, help="OCR data: RVL-CDIP (and Tobacco-3482 for DIL)")
    ap.add_argument("--out_dir", default=None, help="default: the checkpoint's directory")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42, help="seed recorded in the file name (the test splits are fixed)")
    args = ap.parse_args()
    if args.setting == "DIL" and len(args.ocr) != 2:
        ap.error("DIL needs --ocr <RVL-CDIP OCR> <Tobacco-3482 OCR>")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_classes = [c.strip() for c in args.base_classes.split(",")]
    all_classes = [c.strip() for c in args.all_classes.split(",")]
    backbone = args.backbone
    if backbone == "llmv3" and os.environ.get("LLMV3_MODEL") == "hf":
        backbone = "llmv3hf"

    model = load_model(args, len(base_classes), device).eval()
    sets = (eaml_loaders if args.backbone == "eaml" else llmv3_loaders)(args, base_classes, all_classes)
    save_predictions(model, device, args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint)),
                     f"base_{args.setting.lower()}", base_classes, sets,
                     backbone=backbone, setting=args.setting, method="base", strategy="", bias_correction="",
                     seed=args.seed, step=0, new_class=None, checkpoint=os.path.abspath(args.checkpoint),
                     script=os.path.basename(__file__), git_commit=_git_commit())


if __name__ == "__main__":
    main()
