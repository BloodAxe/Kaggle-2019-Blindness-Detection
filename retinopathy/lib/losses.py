import math
import pytest
import torch
from pytorch_toolbelt.losses import WingLoss
from torch import nn
from torch.nn.modules.loss import _Loss, MSELoss
import torch.nn.functional as F
from catalyst.contrib import registry


@registry.Criterion
class MagnetLoss(nn.Module):
    def __init__(self, margin=2):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        import torch.nn.functional as F
        bs = int(features.size(0))
        loss = 0

        index = torch.arange(bs).to(features.device)

        for i in range(bs):
            same_label = labels == labels[i]
            skip_index = index != i

            dist = F.pairwise_distance(features[i].unsqueeze(0).expand_as(features), features)

            same_class_dist = dist[skip_index & same_label]
            if len(same_class_dist):
                same_class_loss = torch.pow(same_class_dist, 2)
                loss = loss + torch.sum(same_class_loss)

            diff_class_dist = dist[skip_index & ~same_label]
            if len(diff_class_dist):
                diff_class_loss = torch.pow(F.relu(self.margin - diff_class_dist), 2)
                loss = loss + torch.sum(diff_class_loss)

        return loss / (bs * bs - bs)


def clip_regression(input, target, min=0, max=4):
    min_mask = (target == min) & (input <= min)
    max_mask = (target == max) & (input >= max)
    input = input.masked_fill(min_mask, min)
    input = input.masked_fill(max_mask, max)
    return input, target


@registry.Criterion
class ClippedWingLoss(WingLoss):
    def __init__(self, width=5, curvature=0.5, reduction='mean', min=0, max=4):
        super(ClippedWingLoss, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input, target):
        input, target = clip_regression(input, target, self.min, self.max)
        return super().forward(input, target)


@registry.Criterion
class ClippedMSELoss(MSELoss):
    def __init__(self, min=0, max=4, size_average=None, reduce=None, reduction='mean'):
        super(ClippedMSELoss, self).__init__(size_average, reduce, reduction)
        self.min = min
        self.max = max

    def forward(self, input, target):
        input, target = clip_regression(input, target, self.min, self.max)
        return super().forward(input, target)


def test_magnet_loss():
    margin = 2
    eps = 1e-4
    loss = MagnetLoss(margin=margin)
    x = torch.tensor([[0, 0, 0, 0],
                      [0, 0, 0, 0]], dtype=torch.float32)

    y = torch.tensor([0, 0], dtype=torch.long)

    l = loss(x, y)
    print(l)
    assert pytest.approx(0, abs=eps) == float(l)

    x = torch.tensor([[0, 0, 0, 0],
                      [0, 0, 0, 0]], dtype=torch.float32)

    y = torch.tensor([0, 1], dtype=torch.long)

    l = loss(x, y)
    print(l)
    assert pytest.approx(margin * 2, abs=eps) == float(l)

    x = torch.tensor([[1, 2, 0, 5],
                      [1, 1, 0, 5],
                      [3, -1, 2, -2],
                      [1, 2, 1, 5],
                      [0, 3, 7, 4],
                      [0, -3, -2, 4],
                      [-4, -2, 4, 0]], dtype=torch.float32)

    y = torch.tensor([0, 1, 1, 0, 0, 2, 3],
                     dtype=torch.long)

    l = loss(x, y)
    print(l)
