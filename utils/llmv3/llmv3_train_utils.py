import torch
from torchmetrics import Accuracy
import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device, grad_accum_steps=1, max_optimizer_steps=None):
    """One epoch; with grad_accum_steps > 1 the gradients of that many batches are summed before each optimizer
    step (effective batch = batch_size x grad_accum_steps). Stops early after max_optimizer_steps steps.
    Returns (mean loss, accuracy, optimizer steps taken)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    steps = 0
    accur = Accuracy(task="multiclass", num_classes=model.classifier.out_features).to(device)
    loop = tqdm(dataloader, desc="Training", leave=False)
    optimizer.zero_grad()
    for i, batch in enumerate(loop):
        inputs = {k: v.to(device) for k,v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        logits = model(**inputs)
        loss = F.cross_entropy(logits, labels)
        (loss / grad_accum_steps).backward()
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            steps += 1
        n_batches += 1

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        accur.update(preds, labels)

        loop.set_postfix(loss=loss.item(), accuracy=accur.compute().item())
        if max_optimizer_steps is not None and steps >= max_optimizer_steps:
            break
    avg_loss = total_loss / max(n_batches, 1)
    acc = accur.compute().item()
    return avg_loss, acc, steps


def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    accur = Accuracy(task="multiclass", num_classes=model.classifier.out_features).to(device)
    loop = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in loop:
            inputs = {k: v.to(device) for k,v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            accur.update(preds, labels)

            loop.set_postfix(loss=loss.item(), accuracy=accur.compute().item())
    avg_loss = total_loss / len(dataloader)
    acc = accur.compute().item()
    return avg_loss, acc
