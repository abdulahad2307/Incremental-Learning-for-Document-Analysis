import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class DocFormerTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        
        # Loss functions
        self.mm_mlm_loss_fn = torch.nn.CrossEntropyLoss()
        self.ltr_loss_fn = torch.nn.SmoothL1Loss()
        self.tdi_loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def train_epoch(self, train_loader, epoch, scheduler=None):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                bboxes=batch["bboxes"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                task="pretrain"
            )
            
            # Calculate losses
            mm_mlm_loss = self.mm_mlm_loss_fn(
                outputs["mm_mlm_logits"].view(-1, self.model.text_embeddings.config.vocab_size),
                batch["input_ids"].view(-1)
            )
            
            ltr_loss = self.ltr_loss_fn(
                outputs["ltr_output"],
                batch["pixel_values"]
            )
            
            # For TDI, create random negative samples
            batch_size = batch["input_ids"].size(0)
            tdi_labels = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            
            # 20% of the time use negative samples
            neg_indices = torch.randperm(batch_size)[:batch_size // 5]
            tdi_labels[neg_indices] = 0.0
            
            tdi_loss = self.tdi_loss_fn(
                outputs["tdi_logits"].squeeze(),
                tdi_labels
            )
            
            # Combine losses with weights
            loss = (self.config.mm_mlm_weight * mm_mlm_loss + 
                    self.config.ltr_weight * ltr_loss + 
                    self.config.tdi_weight * tdi_loss)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    bboxes=batch["bboxes"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    task="pretrain"
                )
                
                # Calculate losses
                mm_mlm_loss = self.mm_mlm_loss_fn(
                    outputs["mm_mlm_logits"].view(-1, self.model.text_embeddings.config.vocab_size),
                    batch["input_ids"].view(-1)
                )
                
                ltr_loss = self.ltr_loss_fn(
                    outputs["ltr_output"],
                    batch["pixel_values"]
                )
                
                tdi_labels = torch.ones(batch["input_ids"].size(0), dtype=torch.float32, device=self.device)
                tdi_loss = self.tdi_loss_fn(
                    outputs["tdi_logits"].squeeze(),
                    tdi_labels
                )
                
                loss = (self.config.mm_mlm_weight * mm_mlm_loss + 
                        self.config.ltr_weight * ltr_loss + 
                        self.config.tdi_weight * tdi_loss)
                
                total_loss += loss.item()
        
        return total_loss / len(eval_loader)
    
    def save_checkpoint(self, path, epoch, best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        
        if best:
            torch.save(checkpoint, os.path.join(path, "best_model.pt"))
        else:
            torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pt"))
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"]