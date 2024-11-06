from typing import Literal

import optuna
import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric

from src.tree_loss import generate_label_hierarchy


class PatchedOptunaCallback(
    optuna.integration.PyTorchLightningPruningCallback, pytorch_lightning.Callback
):
    pass

class LabelRemapper(nn.Module):

    def __init__(self, remapping_dict, from_level, to_level) -> None:

        self.remapping_dict = remapping_dict
        self.from_level = from_level
        self.to_level = to_level

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        out_remappings = generate_label_hierarchy(x, self.remapping_dict, start_level=self.from_level)
        return out_remappings[self.to_level]

# Loss functions


def dice_score_hard(pred: torch.Tensor, target: torch.Tensor, average="micro", zero_division=0., invert=False):
    """ Soft implementation of the DICE score."""

    # target is either of shape [C] or of shape [C,H,W]. Add batch dimension.
    if target.dim() in [1, 3]:
        target = target.unsqueeze(1)

    # Integer encoding of classes. Expand to one-hot.
    if target.size(1) == 1:

        target_new = torch.zeros_like(pred)
        target_new.scatter_(1, target, 1)
        target = target_new

    assert (pred.shape == target.shape)

    dims = {i for i in range(pred.dim())}
    dims = {"micro": dims, "macro": dims-{1}, "samples": dims-{0}, None: dims-{0, 1}}
    dims = tuple(dims[average])

    intersection = torch.sum(pred * target, dims)
    cardinality = torch.sum(pred + target, dims)

    dice_score = (2. * intersection + zero_division) / (cardinality + zero_division)

    if invert:
        dice_score = 1 - dice_score

    if average is None:
        return dice_score
    else:
        return torch.mean(dice_score)


def dice_score_soft(pred: torch.Tensor, target: torch.Tensor, average="micro", zero_division=0., invert=False):
    """ Soft implementation of the DICE score."""

    # target is either of shape [C] or of shape [C,H,W]. Add batch dimension.
    if target.dim() in [1, 3]:
        target = target.unsqueeze(1)

    # Integer encoding of classes. Expand to one-hot.
    if target.size(1) == 1:

        target_new = torch.zeros_like(pred)
        target_new.scatter_(1, target, 1)
        target = target_new

    assert (pred.shape == target.shape)

    dims = {i for i in range(pred.dim())}
    dims = {"micro": dims, "macro": dims-{1}, "samples": dims-{0}, None: dims-{0, 1}}
    dims = tuple(dims[average])

    intersection = torch.sum(pred * target, dims)
    cardinality = torch.sum(pred + target, dims)

    dice_score = (2. * intersection + zero_division) / (cardinality + zero_division)

    if invert:
        dice_score = 1 - dice_score

    if average is None:
        return dice_score
    else:
        return torch.mean(dice_score)

def dice_loss_soft(pred: torch.Tensor, target: torch.Tensor, average: Literal["micro", "macro", "samples"] = "micro", zero_division: float = 0.):
    """ Soft implementation of the DICE loss."""

    assert (pred.shape == target.shape)


    pred = F.softmax(pred, dim=1)

    assert average in ["micro", "macro", "samples"]

    dice_loss = dice_score_soft(pred, target, average, zero_division, invert=True)
    return dice_loss




class SoftDiceLoss(nn.Module):

    def __init__(self, average: Literal["micro", "macro", "samples"] = "micro", epsilon: float = 0.) -> None:
        super().__init__()

        self.average = average
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        return dice_loss_soft(x, y, self.average, self.epsilon)


class MultiLabelLoss(nn.Module):

    def __init__(self, hard_targets=True) -> None:
        super().__init__()
        self.hard_targets = hard_targets

    def forward(self, pred, mask):

        if self.hard_targets:
            mask = (mask > 0).to(float)

        return nn.functional.binary_cross_entropy_with_logits(pred, mask, reduction="mean")


class SoftDICEMetric(Metric):

    def __init__(self, average: Literal["micro", "macro"] = "micro", zero_division=0.,  **kwargs):
        super().__init__(**kwargs)

        self.zero_div = zero_division
        self.average = average
        assert self.average in ["micro", "macro"]

        self.add_state("agg_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:

        dice_score = dice_score_soft(pred=preds, target=target, average=self.average, zero_division=self.zero_div, invert=False)

        self.total += preds.size(0)
        self.agg_score += dice_score

    def compute(self) -> Tensor:
        return self.agg_score / self.total


class L1CalibrationMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("agg_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:

        self.agg_score += ((preds - target).abs().sum(dim=1)/2).sum()
        self.total += torch.numel(preds)//preds.size(1)  # preds.reshape(-1, preds.size(1)).size(0)

    def compute(self) -> Tensor:
        return self.agg_score / self.total
