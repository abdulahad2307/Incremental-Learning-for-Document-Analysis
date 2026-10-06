import argparse
import os
import torch
from utils.seed import add_seed_arg, set_seed
from utils.llmv3.llmv3_data_loader import get_dataloaders, get_test_loader
from utils.llmv3.llmv3_model_loader import build_llmv3_model, checkpoint_meta, HF_LAYOUTLMV3, MODEL_TYPES
from utils.llmv3.llmv3_train_utils import train_epoch, val_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Document Classification")
    parser.add_argument("--dataset", type=str, choices=["rvl_cdip", "tobacco3482", "docbank", "publaynet"], required=True)
    parser.add_argument("--ocr_tensor_file", type=str, required=True)
    parser.add_argument("--base_classes", type=str, default=None,
                        help="Comma-separated limited class list to train on, e.g. letter,form,email")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Batches per optimizer step; effective batch = batch_size x grad_accum_steps "
                             "(LayoutLMv3 paper: 64, e.g. 8 x 8)")
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="Optimizer steps to train (LayoutLMv3 paper: 20,000; no early stopping); --epochs is an "
                             "upper bound. The checkpoint with the best validation accuracy is kept.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="Fixed learning rate (2e-5 in the LayoutLMv3 paper)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bbox_style", type=str, choices=["rect", "poly"], default="poly")
    parser.add_argument("--model_type", choices=MODEL_TYPES, default="custom",
                        help="custom: Custom LayoutLMv3 (BERT + ViT + fusion, thesis model); hf: pre-trained LayoutLMv3 "
                             "(needs OCR tensors from tools/ocr/ocr_extraction_bbox_layoutlmv3.py and --bbox_style rect)")
    parser.add_argument("--hf_model_name", type=str, default=HF_LAYOUTLMV3, help="Pre-trained model for --model_type hf")
    parser.add_argument("--text_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume from")
    parser.add_argument("--images_per_class", type=int, default=None,
                        help="Max number of images to sample per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducible sampling")
    add_seed_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_image_dir = os.path.join(args.image_dir, "train")
    val_image_dir = os.path.join(args.image_dir, "val")

    if args.base_classes:
        base_classes = [c.strip() for c in args.base_classes.split(',')]
    else:
        base_classes = None

    train_loader, val_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        ocr_tensor_file=args.ocr_tensor_file,
        base_classes=args.base_classes,
        image_dir=train_image_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        bbox_style=args.bbox_style,
        images_per_class=args.images_per_class,
        seed=args.seed,
        val_image_dir=val_image_dir if os.path.isdir(val_image_dir) else None,  # official val split
    )

    device = torch.device(args.device)

    if args.model_type == "hf" and args.bbox_style != "rect":
        raise ValueError("--model_type hf needs one [x0, y0, x1, y1] box per token: use --bbox_style rect")
    model = build_llmv3_model(
        args.model_type,
        num_labels=num_classes,
        text_model_name=args.text_encoder,
        vision_model_name=args.vision_encoder,
        hf_model_name=args.hf_model_name,
    )
    print(f"Model: {type(model).__name__} ({args.model_type})")
    model.to(device)

    # LayoutLMv3 paper (Huang et al., 2022), RVL-CDIP fine-tuning: Adam with a fixed learning rate of 2e-5 (no weight
    # decay: torch's AdamW would add its default of 0.01, which the paper does not use)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Optimizer: Adam, fixed lr={args.lr}; effective batch {args.batch_size * args.grad_accum_steps} "
          f"({args.batch_size} x {args.grad_accum_steps})" + (f", at most {args.max_steps} steps" if args.max_steps else ""))
    total_steps = 0

    best_val_acc = 0.0

    # Resume functionality
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint['best_val_acc']
        total_steps = checkpoint.get('total_steps', 0)
        print(f"Resumed training from {args.resume} at epoch {args.resume_epoch}")

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        remaining = None if args.max_steps is None else args.max_steps - total_steps
        train_loss, train_acc, steps = train_epoch(model, train_loader, optimizer, device,
                                                   grad_accum_steps=args.grad_accum_steps,
                                                   max_optimizer_steps=remaining)
        total_steps += steps
        val_loss, val_acc = val_epoch(model, val_loader, device)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f} / Val loss={val_loss:.4f}, acc={val_acc:.4f} "
              f"(optimizer steps: {total_steps})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, f"layoutlmv3_{args.dataset}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'total_steps': total_steps,
                **checkpoint_meta(model),
            }, save_path)
            print(f"Saved best model to {save_path}")
        # Latest state for resuming a run that hits the job's time limit (--resume <this> --resume_epoch <epoch>)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc, 'total_steps': total_steps, **checkpoint_meta(model)},
                   os.path.join(args.save_dir, f"layoutlmv3_{args.dataset}_last.pt"))

        if args.max_steps is not None and total_steps >= args.max_steps:
            print(f"Reached --max_steps ({total_steps} optimizer steps).")
            break

    print("Training completed.")

    # Test accuracy of the best-validation checkpoint (the value for LLMV3[HF]_BASE_*_ACC in scripts/config.sh)
    test_image_dir = os.path.join(args.image_dir, "test")
    if os.path.isdir(test_image_dir):
        best = torch.load(os.path.join(args.save_dir, f"layoutlmv3_{args.dataset}_best.pt"), map_location=device)
        model.load_state_dict(best["model_state_dict"])
        test_loader = get_test_loader(args.dataset, args.ocr_tensor_file, args.base_classes, test_image_dir,
                                      batch_size=args.batch_size, max_length=args.max_length,
                                      bbox_style=args.bbox_style, seed=args.seed)
        test_loss, test_acc = val_epoch(model, test_loader, device)
        print(f"Best checkpoint (epoch {best['epoch']}, val acc {best['best_val_acc']:.4f}): "
              f"Test loss={test_loss:.4f}, Test accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
