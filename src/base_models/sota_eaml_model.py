import argparse
import os
import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from utils.eaml.dataloader import EAML_DataLoader, load_class_list
from utils.eaml.eaml_model import EAMLModel
from utils.eaml.mutual_learning import MutualLearningLoss
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import json
import numpy as np

class EarlyStoppingHandler:
    def __init__(self, patience=7, min_delta=0, verbose=True):
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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print("Early stopping triggered")

class EAMLTrainer:
    def __init__(self, model,class_list, device=None, learning_rate=1e-4, weight_decay=0.01,
                 cls_weight=1.0, kld_weight=0.3, kld_threshold=0.1):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU requested but not available. Check your GPU configuration.")
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.criterion = MutualLearningLoss(
            cls_weight=cls_weight,
            kld_weight=kld_weight,
            threshold=kld_threshold
        )
        self.class_list = class_list
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        #self.optimizer = torch.optim.SGD(
        #    model.parameters(),
        #    lr=learning_rate,
        #    momentum=0.9,
        #    nesterov=True,
        #    weight_decay=weight_decay
        #)
        #self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        #    self.optimizer,
        #    T_0=10,
        #    T_mult=2,
        #    eta_min=1e-6
        #)
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
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            images = batch['images'].to(self.device)
            texts = {
                'input_ids': batch['texts']['input_ids'].to(self.device),
                'attention_mask': batch['texts']['attention_mask'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            # ---- Label Integrity Checking ----
            assert labels.min() >= 0 and labels.max() < len(self.class_list), \
                f"Batch label out of bounds! Got min={labels.min()}, max={labels.max()}, num_classes={len(self.class_list)}"
            # printing some samples for manual inspection (once!)
            if epoch == 0 and progress.n == 0:
                print("First batch labels:", labels.cpu().numpy())
                print("First images batch, mean:", images.mean().item())
                print("First tokenized[0] input_ids:", texts['input_ids'][0])
            self.optimizer.zero_grad()
            outputs = self.model(images, texts, return_features=True)
            loss_dict = self.criterion(outputs, labels)
            loss = loss_dict['total_loss']
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            #gradient norms for monitoring
            #grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
            #print(f"Grad norms: {grad_norms}")

            self.optimizer.step()
            total_loss += loss.item()
            cls_loss_sum += loss_dict['cls_loss'].item()
            kld_loss_sum += loss_dict['kld_loss'].item()
            _, predicted = torch.max(outputs['fusion_logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress.set_postfix({
                'loss': total_loss/(progress.n+1),
                'cls_loss': cls_loss_sum/(progress.n+1),
                'kld_loss': kld_loss_sum/(progress.n+1),
                'acc': 100*correct/total
            })
            #self.scheduler.step()
        self.scheduler.step()
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

        label_counts = np.zeros(len(self.class_list), dtype=int)
        for batch in dataloader:
            batch_labels = batch['labels'].cpu().numpy()
            for l in batch_labels:
                label_counts[l] += 1

        print("Label distribution in loader:", dict(enumerate(label_counts)))
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
                images = batch['images'].to(self.device)
                texts = {
                    'input_ids': batch['texts']['input_ids'].to(self.device),
                    'attention_mask': batch['texts']['attention_mask'].to(self.device)
                }
                labels = batch['labels'].to(self.device)
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
        print(f"Evaluation - Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, KLD Loss: {avg_kld_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def save_checkpoint(self, output_dir, epoch, val_loss, val_acc, is_best=False):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, f"eaml_checkpoint_ep{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_path)
        if is_best:
            best_model_path = os.path.join(output_dir, "eaml_best_model.pt")
            shutil.copyfile(checkpoint_path, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")

    def cleanup_checkpoints(self, output_dir, keep_last_n=2, keep_best=True):
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("eaml_checkpoint_ep")]
        checkpoints.sort(key=lambda x: int(x.split("ep")[1].split(".")[0]))
        checkpoints_to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
        for checkpoint in checkpoints_to_delete:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Deleted old checkpoint: {checkpoint}")

    def log_gpu_stats(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            return f"GPU:{i} Memory: {memory_allocated:.1f}MB (allocated) / {memory_reserved:.1f}MB (reserved)"
        return "GPU not available"

def main():
    parser = argparse.ArgumentParser(description="EAML for Document Classification with Precomputed OCR")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--ocr_data_path', type=str, required=True, help='Path to precomputed OCR file (.json or .pt)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for L2 regularization')
    parser.add_argument('--class_mapping_path', type=str, required=True, help='Path to JSON file with full class mapping')
    parser.add_argument('--classes', type=str, default=None, help='Comma-separated list of class names to subset')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--resume', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help="Force device selection")
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--keep_checkpoints', type=int, default=2, help='Number of recent checkpoints to keep')
    parser.add_argument('--cls_weight', type=float, default=1.0, help='Weight for classification loss')
    parser.add_argument('--kld_weight', type=float, default=0.3, help='Weight for KL divergence loss')
    parser.add_argument('--kld_threshold', type=float, default=0.1, help='Threshold for truncated KL divergence')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--freeze_image_encoder', type=bool, default=False, help='False allows weights to update, True for feature extraction only')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.class_mapping_path, 'r') as f:
        full_class_mapping = json.load(f)

    #class_list = args.classes


    # mapping to int key => class string, sorted by index:
    ordered_classes = [full_class_mapping[str(i)] for i in range(len(full_class_mapping))]

    if args.classes:
        requested_subset = args.classes.split(",")
        for cls in requested_subset:
            if cls not in ordered_classes:
                raise ValueError(f"Subset class {cls} not in full class mapping")
        # Maintain order according to full mapping
        class_list = [cls for cls in ordered_classes if cls in requested_subset]
    else:
        class_list = ordered_classes
    print("Using class list:", class_list)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA/GPU requested but not available. Check your GPU configuration.")
    device = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
    if class_list is None:
        raise ValueError("Class list not found. Please provide via --classes")
    
    print("Class to Index Mapping:")
    for idx, cname in enumerate(class_list):
        print(f"{idx}: {cname}")


    eaml_loader = EAML_DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        class_list=class_list,
        ocr_data_path=args.ocr_data_path
    )

    train_loader = eaml_loader.get_loader('train')
    val_loader = eaml_loader.get_loader('val', shuffle=False)
    
    model = EAMLModel(
        num_classes=len(class_list),
        embed_dim=args.embed_dim,
        dropout_rate=args.dropout_rate,
        freeze_image_encoder=args.freeze_image_encoder
    )

    start_epoch = 0
    trainer = EAMLTrainer(
        model=model,
        class_list = class_list,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cls_weight=args.cls_weight,
        kld_weight=args.kld_weight,
        kld_threshold=args.kld_threshold
    )

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    if not args.eval_only:
        early_stopping = EarlyStoppingHandler(patience=args.patience)
        best_val_loss = float('inf')
        for epoch in range(start_epoch, args.num_epochs):
            train_loss, epoch_time = trainer.train_epoch(train_loader, epoch)
            val_loss, val_acc = trainer.evaluate(val_loader)
            gpu_stats = trainer.log_gpu_stats()
            print(f"Epoch {epoch+1}/{args.num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={epoch_time:.2f}s, {gpu_stats}")
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            trainer.save_checkpoint(args.output_dir, epoch, val_loss,val_acc, is_best)
            trainer.cleanup_checkpoints(args.output_dir, keep_last_n=args.keep_checkpoints)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
            json.dump(class_list, f)
    else:
        if args.resume and os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}' for evaluation...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("No checkpoint found for evaluation (--resume required).")
        
        print("Evaluating on validation set...")
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        test_loader = eaml_loader.get_loader('test', shuffle=False)
        print("Evaluating on test set...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} GPU(s)")
        print(f"First GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available - falling back to CPU")
    main()
