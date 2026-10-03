import os
import time
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from utils.docformer.dataloader import OCRDataset, collate_fn
from utils.docformer.model import DocFormer 
from utils.docformer.config import DocFormerConfig
from utils.docformer.trainer import DocFormerTrainer
from utils.docformer.evaluator import DocFormerEvaluator

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='DocFormer Training/Evaluation')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ocr_token_dir', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated list of classes to train/evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--finetune_lr', type=float, default=2.5e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--training_stage', choices=['text_pretrain', 'multimodal_pretrain', 'finetune'], default='finetune')
    parser.add_argument('--evaluate_only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)

    # Config
    config = DocFormerConfig()
    config.max_position_embeddings = args.max_seq_len
    config.device = args.device
    config.learning_rate = args.learning_rate
    config.finetune_lr = args.finetune_lr
    config.num_train_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.num_workers = args.num_workers
    config.use_amp = args.use_amp
    config.save_interval = args.save_interval
    config.output_dir = args.output_dir
    config.weight_decay = args.weight_decay
    config.early_stop_patience = args.early_stop_patience
    config.training_stage = args.training_stage

    # Data
    logger.info("Loading datasets...")
    train_dataset = OCRDataset(args.data_dir, 'train', args.ocr_token_dir, config)
    val_dataset = OCRDataset(args.data_dir, 'val', args.ocr_token_dir, config)
    test_dataset = OCRDataset(args.data_dir, 'test', args.ocr_token_dir, config)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    num_classes = len(train_dataset.class_to_idx)
    logger.info(f"Detected {num_classes} classes")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True,
        persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=config.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=config.num_workers > 0
    )

    # Model
    logger.info("Initializing model...")
    model = DocFormer(config, num_classes=num_classes).to(config.device)

    # Trainer
    trainer = DocFormerTrainer(model, config, train_loader, val_loader)

    logger.info("Starting training...")
    best_model_path = trainer.train(config.output_dir)

    logger.info("Evaluating on test set using best model...")
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    evaluator = DocFormerEvaluator(model, config, test_loader, train_dataset.idx_to_class)
    test_metrics = evaluator.evaluate_model()
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%, Test Loss: {test_metrics['loss']:.4f}")

if __name__ == '__main__':
    main()
