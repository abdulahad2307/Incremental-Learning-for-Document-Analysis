import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils.docformer_tessrect.datastet import RVLCDIPDataset, collate_fn
from utils.docformer_tessrect.model import DocFormer
from utils.docformer_tessrect.trainer import DocFormerTrainer

def main():
    parser = argparse.ArgumentParser(description="DocFormer for RVL-CDIP Document Classification")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory (should contain train/val/test subdirectories)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for models and logs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2.5e-5, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length for text')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint to resume training or for evaluation')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize datasets
    train_dataset = RVLCDIPDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='train'
    )
    
    val_dataset = RVLCDIPDataset(
        data_dir=args.data_dir,
        max_seq_length=args.max_seq_length,
        split='val'
    )

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Initialize model
    num_classes = len(train_dataset.class_to_idx)
    model = DocFormer(num_classes=num_classes)

    # Load checkpoint if resuming or evaluating
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.resume}")

    # Evaluation only mode
    if args.eval_only:
        if not args.resume:
            raise ValueError("Must provide --resume checkpoint for evaluation")
        
        trainer = DocFormerTrainer(
            model=model,
            train_loader=train_loader,  # Not used in eval
            val_loader=val_loader,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        val_loss, val_acc = trainer.validate()
        print(f"Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        return

    # Training mode
    trainer = DocFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

    # Start training
    trainer.train()

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }, os.path.join(args.output_dir, 'final_model.pt'))

if __name__ == '__main__':
    main()