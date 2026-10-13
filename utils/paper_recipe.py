"""Training recipe of the backbones, shared by the base models and every incremental step (CIL, DIL).

  EAML (Bakkali et al., 2021):      SGD with Nesterov momentum 0.9, lr 1e-3 halved every 10 epochs, batch 16,
                                    no weight decay; augmentation in utils/image_transforms.eaml_train_transform
  LayoutLMv3 (Huang et al., 2022):  Adam, fixed lr 2e-5, batch 64, no weight decay, no augmentation
Both: early stopping on the validation set (patience 10) within an incremental step, fp32, no gradient clipping.
The incremental-learning methods (ER, EWC, KD, EVM, OOD, bias correction) add their own terms on top.
"""
import torch

EAML_LR = 1e-3
LLMV3_LR = 2e-5


def eaml_optimizer(params, lr=EAML_LR):
    return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)


def llmv3_optimizer(params, lr=LLMV3_LR):
    return torch.optim.Adam(params, lr=lr)


class EAMLStepLR:
    """lr = base_lr * 0.5 ** (epochs_done // 10) (EAML paper). Call step() once per epoch (its arguments are ignored,
    so it can replace the earlier validation-driven schedulers); epochs_done > 0 continues a resumed run."""

    def __init__(self, optimizer, base_lr, epochs_done=0):
        self.optimizer, self.base_lr, self.epochs_done = optimizer, base_lr, max(0, int(epochs_done))
        self._apply()

    def _apply(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * 0.5 ** (self.epochs_done // 10)

    def step(self, *_, **__):
        self.epochs_done += 1
        self._apply()


class FixedLR:
    """Constant learning rate (LayoutLMv3 paper); step() does nothing."""

    def __init__(self, *_, **__):
        pass

    def step(self, *_, **__):
        pass
