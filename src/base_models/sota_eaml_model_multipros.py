import argparse
import os
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from torch.utils.data import DataLoader
from utils.eaml.dataloader_old import EAML_DataLoader, load_class_list
from utils.eaml.eaml_model import EAMLModel
from utils.eaml.mutual_learning import MutualLearningLoss
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import json
import logging
from datetime import datetime

def setup_logging(output_dir, rank=0):
    """Setup logging for training"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)

class EarlyStoppingHandler:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                logging.info("Early stopping triggered")

class EAMLTrainer:
    def __init__(self, model, device=None, learning_rate=1e-4, weight_decay=0.01,
                 cls_weight=1.0, kld_weight=0.3, kld_threshold=0.1, rank=0, world_size=1):
        
        self.rank = rank
        self.world_size = world_size
        
        if device is None:
            device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        
        if rank == 0:
            logging.info(f"Using device: {self.device}")
            logging.info(f"World size: {world_size}")
        
        self.model = model.to(self.device)
        
        # Wrap model with DDP for multi-GPU training
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
        
        self.criterion = MutualLearningLoss(
            cls_weight=cls_weight,
            kld_weight=kld_weight,
            threshold=kld_threshold
        )
        
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )

    def train_epoch(self, dataloader, epoch):
        start_time = time.time()
        self.model.train()
        
        total_loss = 0
        cls_loss_sum = 0
        kld_loss_sum = 0
        correct = 0
        total = 0
        
        # Use tqdm only on rank 0
        if self.rank == 0:
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        else:
            progress = dataloader
        
        for batch_idx, batch in enumerate(progress):
            images = batch['images'].to(self.device, non_blocking=True)
            texts = {
                'input_ids': batch['texts']['input_ids'].to(self.device, non_blocking=True),
                'attention_mask': batch['texts']['attention_mask'].to(self.device, non_blocking=True)
            }
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, texts, return_features=True)
            loss_dict = self.criterion(outputs, labels)
            loss = loss_dict['total_loss']
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            cls_loss_sum += loss_dict['cls_loss'].item()
            kld_loss_sum += loss_dict['kld_loss'].item()
            
            _, predicted = torch.max(outputs['fusion_logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar only on rank 0
            if self.rank == 0 and hasattr(progress, 'set_postfix'):
                progress.set_postfix({
                    'loss': total_loss/(batch_idx+1),
                    'cls_loss': cls_loss_sum/(batch_idx+1),
                    'kld_loss': kld_loss_sum/(batch_idx+1),
                    'acc': 100*correct/total
                })
        
        self.scheduler.step()
        
        epoch_time = time.time() - start_time
        
        if self.rank == 0:
            logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        return total_loss / len(dataloader), epoch_time

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        cls_loss_sum = 0
        kld_loss_sum = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device, non_blocking=True)
                texts = {
                    'input_ids': batch['texts']['input_ids'].to(self.device, non_blocking=True),
                    'attention_mask': batch['texts']['attention_mask'].to(self.device, non_blocking=True)
                }
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(images, texts, return_features=True)
                loss_dict = self.criterion(outputs, labels)
                
                total_loss += loss_dict['total_loss'].item()
                cls_loss_sum += loss_dict['cls_loss'].item()
                kld_loss_sum += loss_dict['kld_loss'].item()
                
                _, predicted = torch.max(outputs['fusion_logits'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = cls_loss_sum / len(dataloader)
        avg_kld_loss = kld_loss_sum / len(dataloader)
        accuracy = 100 * correct / total
        
        if self.rank == 0:
            logging.info(f"Evaluation - Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, "
                        f"KLD Loss: {avg_kld_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy

    def save_checkpoint(self, output_dir, epoch, val_loss, is_best=False):
        if self.rank != 0:  # Only save on rank 0
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model state dict (handle DDP wrapper)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint_path = os.path.join(output_dir, f"eaml_checkpoint_ep{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        
        if is_best:
            best_model_path = os.path.join(output_dir, "eaml_best_model.pt")
            shutil.copyfile(checkpoint_path, best_model_path)
            logging.info(f"Saved best model with validation loss: {val_loss:.4f}")

    def cleanup_checkpoints(self, output_dir, keep_last_n=2):
        if self.rank != 0:
            return
        
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("eaml_checkpoint_ep")]
        checkpoints.sort(key=lambda x: int(x.split("ep")[1].split(".")[0]))
        
        checkpoints_to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
        
        for checkpoint in checkpoints_to_delete:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logging.info(f"Deleted old checkpoint: {checkpoint}")

def setup_distributed(rank, world_size, port="12355"):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """Worker function for distributed training"""
    
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Setup logging
    setup_logging(args.output_dir, rank)
    
    try:
        # Load class list
        class_list = args.classes
        if not class_list:
            raise ValueError("Class list not found. Please provide via --classes")
        
        # Create data loaders
        eaml_loader = EAML_DataLoader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            class_list=class_list,
            ocr_data_path=args.ocr_data_path,
            handle_empty_text=args.handle_empty_text,
            fallback_text=args.fallback_text,
            num_workers=args.num_workers
        )
        
        # Create distributed samplers for multi-GPU training
        train_loader = eaml_loader.get_loader('train', shuffle=(world_size == 1))
        val_loader = eaml_loader.get_loader('val', shuffle=False)
        
        if world_size > 1:
            train_sampler = DistributedSampler(train_loader.dataset, rank=rank, num_replicas=world_size)
            train_loader = DataLoader(
                train_loader.dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=train_loader.collate_fn,
                pin_memory=True
            )
        
        model = EAMLModel(
            num_classes=len(class_list),
            embed_dim=args.embed_dim,
            freeze_image_encoder=args.freeze_image_encoder
        )
        
        trainer = EAMLTrainer(
            model=model,
            device=f'cuda:{rank}' if torch.cuda.is_available() else 'cpu',
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            cls_weight=args.cls_weight,
            kld_weight=args.kld_weight,
            kld_threshold=args.kld_threshold,
            rank=rank,
            world_size=world_size
        )
        
        start_epoch = 0
        
        # Resume from checkpoint if given
        if args.resume and rank == 0:
            if os.path.isfile(args.resume):
                logging.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
                start_epoch = checkpoint['epoch'] + 1
                
                # Load model state (handle DDP wrapper)
                if hasattr(trainer.model, 'module'):
                    trainer.model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    trainer.model.load_state_dict(checkpoint['model_state_dict'])
                
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                logging.warning(f"No checkpoint found at '{args.resume}'")
        
        if world_size > 1:
            start_epoch_tensor = torch.tensor(start_epoch, device=f'cuda:{rank}')
            dist.broadcast(start_epoch_tensor, src=0)
            start_epoch = start_epoch_tensor.item()
        
        if not args.eval_only:
            # Training loop
            early_stopping = EarlyStoppingHandler(patience=args.patience) if rank == 0 else None
            best_val_loss = float('inf')
            
            for epoch in range(start_epoch, args.num_epochs):
                # Set epoch for distributed sampler
                if world_size > 1 and hasattr(train_loader, 'sampler'):
                    train_loader.sampler.set_epoch(epoch)
                
                train_loss, epoch_time = trainer.train_epoch(train_loader, epoch)
                val_loss, val_acc = trainer.evaluate(val_loader)
                
                if rank == 0:
                    gpu_memory = torch.cuda.memory_allocated(rank) / 1024**2 if torch.cuda.is_available() else 0
                    logging.info(f"Epoch {epoch+1}/{args.num_epochs}: train_loss={train_loss:.4f}, "
                               f"val_loss={val_loss:.4f}, time={epoch_time:.2f}s, "
                               f"GPU Memory: {gpu_memory:.1f}MB")
                    
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    
                    trainer.save_checkpoint(args.output_dir, epoch, val_loss, is_best)
                    trainer.cleanup_checkpoints(args.output_dir, keep_last_n=args.keep_checkpoints)
                    
                    if early_stopping:
                        early_stopping(val_loss)
                        if early_stopping.early_stop:
                            logging.info("Early stopping triggered!")
                            break
                
                # Synchronize all processes
                if world_size > 1:
                    dist.barrier()
            
            if rank == 0:
                logging.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
                
                # Save class list
                with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
                    json.dump(class_list, f)
        
        else:
            # Evaluation only
            val_loss, val_acc = trainer.evaluate(val_loader)
            if rank == 0:
                logging.info(f"Evaluation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    finally:
        # Cleanup distributed training
        if world_size > 1:
            cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Enhanced EAML for Document Classification")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--ocr_data_path', type=str, required=True, help='Path to precomputed OCR data (JSON/tensor)')
    parser.add_argument('--classes', nargs='+', default=[], help='Space-separated list of class names')
    parser.add_argument('--handle_empty_text', type=str, choices=['exclude', 'fallback', 'include'], 
                       default='exclude', help='How to handle empty OCR text')
    parser.add_argument('--fallback_text', type=str, default='[EMPTY]', 
                       help='Fallback text for empty OCR when handle_empty_text=fallback')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--freeze_image_encoder', action='store_true', help='Freeze image encoder weights')
    
    # Loss arguments
    parser.add_argument('--cls_weight', type=float, default=1.0, help='Weight for classification loss')
    parser.add_argument('--kld_weight', type=float, default=0.3, help='Weight for KL divergence loss')
    parser.add_argument('--kld_threshold', type=float, default=0.1, help='Threshold for truncated KL divergence')
    
    # Training control
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--keep_checkpoints', type=int, default=2, help='Number of recent checkpoints to keep')
    
    # Multi-GPU arguments
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate arguments
    if not args.classes:
        raise ValueError("Class list must be provided via --classes")
    
    if not os.path.exists(args.ocr_data_path):
        raise ValueError(f"OCR data file not found: {args.ocr_data_path}")
    
    # Check GPU availability
    if args.world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU training requested but CUDA not available")
        if args.world_size > torch.cuda.device_count():
            raise RuntimeError(f"Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available")
    
    # Launch training
    if args.world_size > 1:
        print(f"Launching distributed training on {args.world_size} GPUs")
        mp.spawn(train_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        print("Launching single-GPU training")
        train_worker(0, 1, args)

if __name__ == "__main__":
    main()
