import torch
import torch.nn.functional as F

class TruncatedKLDLoss(torch.nn.Module):
    def __init__(self, threshold=0.1):
        """
        Implemetation to truncate KL Divergence loss for mutual learning.
        
        Parameters:
            threshold (float): Small value to truncate KL loss.
        """
        super(TruncatedKLDLoss, self).__init__()
        self.threshold = threshold

    def forward(self, p_logits, q_logits):
        p_probs = F.softmax(p_logits, dim=1)
        q_probs = F.softmax(q_logits, dim=1)
        kl_div = F.kl_div(p_probs.log(), q_probs, reduction="batchmean")
        return torch.clamp(kl_div, min=self.threshold)
