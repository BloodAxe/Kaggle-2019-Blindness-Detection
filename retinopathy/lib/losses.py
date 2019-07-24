from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from catalyst.contrib import registry
from pytorch_toolbelt.losses import WingLoss
from torch import nn
from torch.nn.modules.loss import MSELoss, SmoothL1Loss


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
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smooth_factor=0.05):
        super().__init__()
        self.smooth_factor = smooth_factor

    def _smooth_labels(self, num_classes, target):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        target_one_hot = F.one_hot(target, num_classes).float()
        target_one_hot[target_one_hot == 1] = 1 - self.smooth_factor
        target_one_hot[target_one_hot == 0] = self.smooth_factor
        return target_one_hot

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=1)
        target_one_hot = self._smooth_labels(input.size(1), target)
        return F.kl_div(logp, target_one_hot, reduction='sum')


@registry.Criterion
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smooth_factor=0.05):
        super().__init__()
        self.smooth_factor = smooth_factor

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smooth_factor) + (1 - one_hot) * self.smooth_factor / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return loss.sum()


@registry.Criterion
class ClippedWingLoss(WingLoss):
    def __init__(self, width=5, curvature=0.5, reduction='mean', min=0, max=4, ignore_index=None):
        super(ClippedWingLoss, self).__init__()
        self.min = min
        self.max = max
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        input, target = clip_regression(input, target, self.min, self.max)
        return super().forward(input, target.float())


@registry.Criterion
class ClippedMSELoss(MSELoss):
    def __init__(self, min=0, max=4, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(ClippedMSELoss, self).__init__(size_average, reduce, reduction)
        self.min = min
        self.max = max
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        input, target = clip_regression(input, target.float(), self.min, self.max)
        return super().forward(input, target.float())


@registry.Criterion
class ClippedHuber(SmoothL1Loss):
    def __init__(self, min=0, max=4, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(ClippedHuber, self).__init__(size_average, reduce, reduction)
        self.min = min
        self.max = max
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        input, target = clip_regression(input, target.float(), self.min, self.max)
        return super().forward(input, target.float())


def quad_kappa_loss(input, targets, y_pow=1, eps=1e-15):
    """
    https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/losses.py#L22

    :param input:
    :param targets:
    :param y_pow:
    :param eps:
    :return:
    """
    batch_size = input.size(0)
    num_ratings = 5
    assert input.size(1) == num_ratings
    tmp = torch.arange(0, num_ratings).view((num_ratings, 1)).expand((-1, num_ratings)).float()
    weights = (tmp - torch.transpose(tmp, 0, 1)) ** 2 / (num_ratings - 1) ** 2
    weights = weights.type(targets.dtype).to(targets.device)

    y_ = input ** y_pow
    y_norm = y_ / (eps + y_.sum(dim=1).reshape((batch_size, 1)))

    hist_rater_a = y_norm.sum(dim=0)
    hist_rater_b = targets.sum(dim=0)

    conf_mat = torch.mm(torch.transpose(y_norm, 0, 1), targets)

    nom = torch.sum(weights * conf_mat)
    denom = torch.sum(weights * torch.mm(hist_rater_a.reshape((num_ratings, 1)),
                                         hist_rater_b.reshape((1, num_ratings))) / batch_size)

    return - (1.0 - nom / denom)


class CappaLoss(nn.Module):
    # TODO: Test
    def __init__(self, y_pow=1, eps=1e-15):
        super().__init__()
        self.y_pow = y_pow
        self.eps = eps

    def forward(self, input, target):
        t = F.log_softmax(input).exp()
        loss = quad_kappa_loss(target, t, self.y_pow, self.eps)
        return loss


class HybridCappaLoss(nn.Module):
    # TODO: Test
    # https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/losses.py#L51
    def __init__(self, y_pow=1, log_scale=0.5, eps=1e-15, log_cutoff=0.9):
        super().__init__()
        self.y_pow = y_pow
        self.log_scale = log_scale
        self.log_cutoff = log_cutoff
        self.eps = eps

    def forward(self, input, target):
        #     cross entropy loss, summed over classes, mean over batches
        log_loss_res = F.cross_entropy(input, target, reduction='none').sum(dim=1).mean()

        # second term
        y = F.log_softmax(input).exp()
        kappa_loss_res = quad_kappa_loss(y, target, y_pow=self.y_pow)

        return kappa_loss_res + self.log_scale * torch.clamp(log_loss_res, self.log_cutoff, 10 ** 3)


def test_quad_kappa_loss():
    targets = torch.tensor(([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]])).float()
    predicted = torch.tensor([[1, 5, 1, 1, 1],
                              [1, 10, 0, 0, 0],
                              [0, 0, 9, 0, 0],
                              [4, 3, 2, 1, 0],
                              [1, 2, 3, 4, 5]]).float().softmax(dim=1)
    loss = quad_kappa_loss(predicted, targets)
    print(loss)


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


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss
    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    Returns
    -------
    loss : torch.Tensor
        Reduced loss.
    """
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')


def cumulative_link_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                         reduction: str = 'elementwise_mean',
                         class_weights: Optional[np.ndarray] = None
                         ) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.
    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.
    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    Returns
    -------
    loss: torch.Tensor
    """
    eps = 1e-15
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true.unsqueeze(1)), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function
    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.
    """

    def __init__(self, reduction: str = 'elementwise_mean',
                 class_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)
