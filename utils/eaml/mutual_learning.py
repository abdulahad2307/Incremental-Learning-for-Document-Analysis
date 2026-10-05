"""EAML training loss (Bakkali et al., EAML, IJDAR 2021, Eqs. 2-9).

    L_1 = CE(image) + beta * TrKL(P_text  || P_image)
    L_2 = CE(text)  + beta * TrKL(P_image || P_text)
    L_3 = CE(fusion)
    L   = w_1 L_1 + w_2 L_2 + w_3 L_3,      w_i in [0, 1], sum w_i = 1,  beta = 0.5

    TrKL(P_2 || P_1) = sum_k P_2 * max{0, log(P_2 / P_1)}     (truncated KL: negative knowledge is ignored)

The paper does not give w_i; they default to 1/3 each. As in deep mutual learning, the other modality's distribution
P_2 is the target of each KL term (no gradient flows through it).
"""
import torch
import torch.nn.functional as F


class TruncatedKLDLoss(torch.nn.Module):
    def __init__(self, threshold=0.0):
        """Truncated KL divergence of the EAML paper. `threshold` is kept for backward compatibility and ignored:
        the paper truncates every class term at 0 (max{0, log(P_2 / P_1)})."""
        super(TruncatedKLDLoss, self).__init__()
        self.threshold = threshold

    def forward(self, target_logits, logits):
        """TrKL(P_target || P), averaged over the batch.

        Args:
            target_logits: logits of the other modality (P_2, the target; detached)
            logits: logits of the modality being trained (P_1)
        """
        log_p2 = F.log_softmax(target_logits.detach(), dim=1)
        log_p1 = F.log_softmax(logits, dim=1)
        per_class = log_p2.exp() * torch.clamp(log_p2 - log_p1, min=0.0)
        return per_class.sum(dim=1).mean()


class MutualLearningLoss(torch.nn.Module):
    def __init__(self, cls_weight=1.0, kld_weight=0.5, threshold=0.0, modality_weights=(1 / 3, 1 / 3, 1 / 3)):
        """
        Parameters:
            cls_weight (float): scale of the three cross-entropy terms (1.0 in the paper)
            kld_weight (float): beta, weight of the truncated-KL terms (0.5 in the paper)
            threshold (float): unused, kept for backward compatibility (see TruncatedKLDLoss)
            modality_weights: (w_1, w_2, w_3) for the image, text and fusion losses; must sum to 1
        """
        super(MutualLearningLoss, self).__init__()
        if abs(sum(modality_weights) - 1.0) > 1e-6:
            raise ValueError(f"modality_weights must sum to 1, got {modality_weights}")
        self.cls_weight = cls_weight
        self.kld_weight = kld_weight
        self.w_img, self.w_txt, self.w_fus = modality_weights
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.kld_criterion = TruncatedKLDLoss(threshold=threshold)

    def forward(self, outputs, labels):
        """
        Args:
            outputs: dict with 'image_logits', 'text_logits' and 'fusion_logits' (EAMLModel(..., return_features=True))
            labels: ground-truth labels
        """
        image_logits = outputs['image_logits']
        text_logits = outputs['text_logits']
        fusion_logits = outputs['fusion_logits']

        img_cls_loss = self.cls_criterion(image_logits, labels)
        txt_cls_loss = self.cls_criterion(text_logits, labels)
        fusion_cls_loss = self.cls_criterion(fusion_logits, labels)

        kld_img = self.kld_criterion(text_logits, image_logits)   # TrKL(P_text || P_image), trains the image branch
        kld_txt = self.kld_criterion(image_logits, text_logits)   # TrKL(P_image || P_text), trains the text branch

        loss_img = self.cls_weight * img_cls_loss + self.kld_weight * kld_img      # L_1
        loss_txt = self.cls_weight * txt_cls_loss + self.kld_weight * kld_txt      # L_2
        loss_fus = self.cls_weight * fusion_cls_loss                               # L_3
        total_loss = self.w_img * loss_img + self.w_txt * loss_txt + self.w_fus * loss_fus

        return {
            'total_loss': total_loss,
            'cls_loss': self.w_img * img_cls_loss + self.w_txt * txt_cls_loss + self.w_fus * fusion_cls_loss,
            'kld_loss': self.w_img * kld_img + self.w_txt * kld_txt,
            'img_cls_loss': img_cls_loss,
            'txt_cls_loss': txt_cls_loss,
            'fusion_cls_loss': fusion_cls_loss
        }
