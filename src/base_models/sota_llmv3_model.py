import argparse
import os
import torch
from utils.llmv3.llmv3_data_loader import get_dataloaders
from utils.llmv3.llmv3_model_loader import LayoutLMv3
from utils.llmv3.llmv3_train_utils import train_epoch, val_epoch
from utils.llmv3.llmv3_eval_utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Document Classification")
    parser.add_argument("--dataset", type=str, choices=["rvl_cdip", "tobacco3482", "docbank", "publaynet"], required=True)
    parser.add_argument("--ocr_tensor_file", type=str, required=True)
    parser.add_argument("--base_classes", type=str, default=None,
                        help="Comma-separated limited class list to train on, e.g. letter,form,email")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bbox_style", type=str, choices=["rect", "poly"], default="poly")
    parser.add_argument("--text_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume from")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--images_per_class", type=int, default=None,
                        help="Max number of images to sample per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducible sampling")
    return parser.parse_args()


def main():
    args = parse_args()
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
        seed=args.seed
    )

    device = torch.device(args.device)

    model = LayoutLMv3(
        text_model_name=args.text_encoder,
        vision_model_name=args.vision_encoder,
        num_labels=num_classes
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches_per_epoch = len(train_loader)
    total_steps = batches_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    best_val_acc = 0.0
    patience_counter = 0

    # Resume functionality
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_acc = checkpoint['best_val_acc']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed training from {args.resume} at epoch {args.resume_epoch}")

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, device)

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.4f} / Val loss={val_loss:.4f}, acc={val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.save_dir, f"layoutlmv3_{args.dataset}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print("Training completed.")


if __name__ == "__main__":
    main()
