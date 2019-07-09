import math
import pytest
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class MagnetLoss(nn.Module):
    def __init__(self, margin = 2):
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


class ClippedMSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    __constants__ = ['reduction']

    def __init__(self, min=0, max=4, size_average=None, reduce=None, reduction='mean'):
        super(ClippedMSELoss, self).__init__(size_average, reduce, reduction)
        self.min = min
        self.max = max

    def forward(self, input, target):
        input = torch.clamp(input, self.min, self.max)
        return F.mse_loss(input, target, reduction=self.reduction)


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
