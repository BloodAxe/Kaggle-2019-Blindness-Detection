from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from catalyst.contrib import registry
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss, wing_loss
from torch import nn
from torch.nn.modules.loss import MSELoss, SmoothL1Loss, _Loss


class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls]

            loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
        return loss


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


def soft_crossentropy(input: torch.Tensor,
                      target: torch.Tensor,
                      ignore_index=None,
                      smooth_factor=0.01,
                      reduction='mean'):
    if ignore_index is not None:
        mask = target != ignore_index
        target = target[mask]
        input = input[mask]

    if not len(target):
        return torch.tensor(0.).type_as(input).to(input.device)

    n_class = input.size(1)
    one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - smooth_factor) + (1 - one_hot) * smooth_factor / (n_class - 1)
    log_prb = F.log_softmax(input, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    return loss


@registry.Criterion
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, smooth_factor=0.01, ignore_index=None, reduction='mean'):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return soft_crossentropy(input, target,
                                 ignore_index=self.ignore_index,
                                 smooth_factor=self.smooth_factor,
                                 reduction=self.reduction)


@registry.Criterion
class WingLoss(_Loss):
    def __init__(self, width=5, curvature=0.5, reduction='mean', ignore_index=None):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        return wing_loss(input, target.float(), self.width, self.curvature, self.reduction)


def cauchy_loss(y_pred, y_true, c=1.0, reduction='mean'):
    x = y_pred - y_true
    loss = torch.log(0.5 * (x / c) ** 2 + 1)
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        loss = loss.mean()
    return loss


@registry.Criterion
class CauchyLoss(_Loss):
    def __init__(self, c=1.0, reduction='mean', ignore_index=None):
        super(CauchyLoss, self).__init__(reduction=reduction)
        self.c = c
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        return cauchy_loss(input, target.float(), self.c, self.reduction)


@registry.Criterion
class ClippedWingLoss(WingLoss):
    def __init__(self, width=5, curvature=0.5, reduction='mean', min=0, max=4, ignore_index=None):
        super(ClippedWingLoss, self).__init__(width=width, curvature=curvature, reduction=reduction)
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
class CustomMSE(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(CustomMSE, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0., device=input.device, dtype=input.dtype)

        return super().forward(input, target.float())


@registry.Criterion
class RMSE(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(RMSE, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.eps = 1e-7

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0., device=input.device, dtype=input.dtype)

        loss = (F.mse_loss(input, target.float(), reduction=self.reduction) + self.eps).sqrt()
        return loss


@registry.Criterion
class Huber(SmoothL1Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(Huber, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0., device=input.device, dtype=input.dtype)

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
            return torch.tensor(0., device=input.device, dtype=input.dtype)

        input, target = clip_regression(input, target.float(), self.min, self.max)
        return super().forward(input, target.float())


def quad_kappa_loss_v2(predictions, labels, y_pow=2, eps=1e-9):
    # with tf.name_scope(name):
    #     labels = tf.to_float(labels)
    #     repeat_op = tf.to_float(
    #         tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
    #     repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
    #     weights = repeat_op_sq / tf.to_float((num_ratings - 1) ** 2)

    batch_size = predictions.size(0)
    num_ratings = predictions.size(1)
    assert predictions.size(1) == num_ratings

    tmp = torch.arange(0, num_ratings).view((num_ratings, 1)).expand((-1, num_ratings)).float()
    weights = (tmp - torch.transpose(tmp, 0, 1)) ** 2 / (num_ratings - 1) ** 2
    weights = weights.type(labels.dtype).to(labels.device)

    pred_ = predictions ** y_pow
    pred_norm = pred_ / (eps + torch.sum(pred_, 1).view(-1, 1))

    hist_rater_a = torch.sum(pred_norm, 0)
    hist_rater_b = torch.sum(labels, 0)

    conf_mat = torch.matmul(pred_norm.t(), labels)

    nom = torch.sum(weights * conf_mat)
    denom = torch.sum(
        weights * torch.matmul(hist_rater_a.view(num_ratings, 1), hist_rater_b.view(1, num_ratings)) / batch_size)
    return -(1.0 - nom / (denom + eps))


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

    # y_ = input ** y_pow
    # y_norm = y_ / (eps + y_.sum(dim=1).reshape((batch_size, 1)))

    hist_rater_b = input.sum(dim=0)
    # hist_rater_b = y_norm.sum(dim=0)
    hist_rater_a = targets.sum(dim=0)

    O = torch.mm(input.t(), targets)
    O = O / O.sum()
    E = torch.mm(hist_rater_a.reshape((num_ratings, 1)),
                 hist_rater_b.reshape((1, num_ratings)))
    E = E / E.sum()
    nom = torch.sum(weights * O)
    denom = torch.sum(weights * E)

    return - (1.0 - nom / denom)


@registry.Criterion
class RegKappa(_Loss):
    def __init__(self, ignore_index=None):
        super(RegKappa, self).__init__()
        self.min = min
        self.max = max
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        target = target.float()
        num = 2 * torch.sum(input * target)
        denom = input.norm(2) + target.norm(2)
        eps = 1e-7
        kappa = num / (denom + eps)
        return 1. - kappa


class CappaLoss(nn.Module):
    # TODO: Test
    def __init__(self, y_pow=1, eps=1e-15):
        super().__init__()
        self.y_pow = y_pow
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        t = F.log_softmax(input, dim=1).exp()
        loss = quad_kappa_loss(t, target, self.y_pow, self.eps)
        return loss


class HybridCappaLoss(nn.Module):
    # TODO: Test
    # https://github.com/JeffreyDF/kaggle_diabetic_retinopathy/blob/master/losses.py#L51
    def __init__(self, y_pow=2, log_scale=1.0, eps=1e-15, log_cutoff=0.9, ignore_index=None, gamma=2.):
        super().__init__()
        self.y_pow = y_pow
        self.log_scale = log_scale
        self.log_cutoff = log_cutoff
        self.eps = eps
        self.ignore_index = ignore_index
        self.gamma = 2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        focal_loss = 0
        num_classes = input.size(1)
        for cls in range(num_classes):
            cls_label_target = (target == cls).long()
            cls_label_input = input[:, cls]
            focal_loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=None)

        # Second term
        y = F.log_softmax(input, dim=1).exp()
        target_one_hot = F.one_hot(target, input.size(1)).float()
        # +1 to make loss be [0;2], instead [-1;1]
        kappa_loss = 1 + quad_kappa_loss_v2(y, target_one_hot, y_pow=self.y_pow, eps=self.eps)

        return kappa_loss + self.log_scale * focal_loss


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


@registry.Criterion
class MagnetLoss(nn.Module):
    def __init__(self, margin=2):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        import torch.nn.functional as F
        bs = int(features.size(0))
        loss = torch.tensor(0, dtype=torch.float32, device=features.device)

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

        return loss / max(1, bs * bs - bs)
