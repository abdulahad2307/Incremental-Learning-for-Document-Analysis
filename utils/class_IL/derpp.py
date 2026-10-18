"""DER++ (Dark Experience Replay++, Buzzega et al., NeurIPS 2020) for class-incremental learning.

A replay memory of exemplars, each stored with the logits of the model that added it to the memory. Every training
step adds two replay terms, each on its own batch drawn from the memory:
    loss = CE(current batch) + alpha * MSE(logits(x'), stored logits of x') + beta * CE(logits(x''), y'')
The classifier grows by one class per CIL step, so a stored logit vector covers the classes known when it was stored;
the MSE compares the current logits on exactly those classes.

The memory outlives a step's job: it is saved next to the step's best checkpoint (memory_path) and loaded by the
next step. Step 1 builds it from the base model: the exemplars it selected for the base classes and its logits.
Items are whatever the backbone's dataset returns (EAML: dataset samples, LayoutLMv3: item dicts); `collate` builds
a batch from them and `forward` returns the classifier logits of a batch.
"""
import os

import torch
import torch.nn.functional as F


def memory_path(checkpoint_path):
    """The DER++ memory saved with a step's best checkpoint."""
    return os.path.splitext(checkpoint_path)[0] + "_derpp_memory.pt"


class DerppMemory:
    def __init__(self):
        self.items, self.labels, self.logits = [], [], []    # logits: 1-D float tensors (classes known when stored)

    def __len__(self):
        return len(self.items)

    def add(self, items, labels, logits):
        """Append exemplars with their labels (class indices) and the logits (n, K) of the model adding them."""
        logits = logits.detach().float().cpu()
        assert len(items) == len(labels) == logits.shape[0], "items, labels and logits must have the same length"
        self.items += list(items)
        self.labels += [int(l) for l in labels]
        self.logits += [row.clone() for row in logits]

    def save(self, path):
        torch.save({"items": self.items, "labels": self.labels, "logits": self.logits}, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path):
        m = cls()
        state = torch.load(path, map_location="cpu", weights_only=False)
        m.items, m.labels, m.logits = state["items"], state["labels"], state["logits"]
        return m

    def batches(self, batch_size, collate, seed=0):
        """Endless replay batches: (collated batch, labels (B,), stored logits (B, K_max) padded with 0, mask)."""
        g = torch.Generator().manual_seed(seed)
        while True:
            order = torch.randperm(len(self), generator=g).tolist()
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                k = max(self.logits[i].numel() for i in idx)
                stored = torch.zeros(len(idx), k)
                mask = torch.zeros(len(idx), k)
                for r, i in enumerate(idx):
                    n = self.logits[i].numel()
                    stored[r, :n], mask[r, :n] = self.logits[i], 1.0
                yield collate([self.items[i] for i in idx]), torch.tensor([self.labels[i] for i in idx]), stored, mask


def logit_matching(current, stored, mask):
    """MSE between the current logits and the stored ones, on the classes each stored vector covers."""
    k = stored.shape[1]
    diff = (current[:, :k] - stored.to(current.device)) * mask.to(current.device)
    return (diff ** 2).sum() / mask.sum().clamp_min(1.0)


def derpp_replay_loss(forward, replay, device, alpha=0.1, beta=0.5):
    """alpha * logit matching on one replay batch + beta * CE on another (DER++ draws them independently)."""
    batch1, _, stored, mask = next(replay)
    loss = alpha * logit_matching(forward(batch1), stored, mask)
    batch2, labels2, _, _ = next(replay)
    return loss + beta * F.cross_entropy(forward(batch2), labels2.to(device))


@torch.no_grad()
def compute_logits(forward, items, collate, batch_size, model):
    """Logits (n, K) of `model` (in eval mode) for a list of memory items."""
    was_training = model.training
    model.eval()
    out = [forward(collate(items[s:s + batch_size])).float().cpu() for s in range(0, len(items), batch_size)]
    model.train(was_training)
    return torch.cat(out) if out else torch.zeros(0, 0)


def train_epoch(model, loader, optimizer, main_loss, forward, replay, device, alpha=0.1, beta=0.5):
    """One DER++ epoch: for every batch of the current data, main_loss(batch) -> (loss, preds, labels) plus the two
    replay terms. Returns (mean loss, accuracy on the current data)."""
    model.train()
    total, correct, loss_sum, n_batches = 0, 0, 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        loss, preds, labels = main_loss(batch)
        loss = loss + derpp_replay_loss(forward, replay, device, alpha, beta)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        n_batches += 1
        correct += (preds.detach() == labels.to(preds.device)).sum().item()
        total += labels.numel()
    return loss_sum / max(n_batches, 1), correct / max(total, 1)
