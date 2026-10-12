"""Configurable classification losses used by the shared trainer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


def build_loss(config, class_weights=None):
    name = config.get("loss", "cross_entropy")
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=config.get("label_smoothing", 0.0))
    if name == "focal":
        return FocalLoss(gamma=config.get("focal_gamma", 2.0), weight=class_weights)
    raise ValueError("Unsupported loss: " + str(name))
