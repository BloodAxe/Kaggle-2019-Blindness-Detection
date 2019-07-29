import torch
import torch.nn.functional as F
import pytest
from pytorch_toolbelt.utils.torch_utils import to_numpy

from retinopathy.losses import ClippedWingLoss, ClippedMSELoss, ClippedHuber


def test_kl():
    logits = torch.tensor([
        [10, 1, 2, 3, 4]
    ]).float()

    target = logits.log_softmax(dim=1).exp()
    input = logits.log_softmax(dim=1)
    l = F.kl_div(input, target, reduction='batchmean')
    print(l)

    logits2 = torch.tensor([
        [1, 10, 2, 3, 4]
    ]).float()

    input2 = logits2.log_softmax(dim=1)
    l = F.kl_div(input2, target, reduction='batchmean')
    print(l)


@torch.no_grad()
def test_huber_loss():
    loss_fn = ClippedHuber(min=0, max=4, reduction='none')

    x = torch.arange(-1, 5, 0.1)
    y0 = torch.tensor(0.0).expand_as(x)
    y1 = torch.tensor(1.0).expand_as(x)
    y2 = torch.tensor(2.0).expand_as(x)
    y3 = torch.tensor(3.0).expand_as(x)
    y4 = torch.tensor(4.0).expand_as(x)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y0)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y1)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y2)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y3)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y4)))
    plt.title(f'ClippedHuber')
    plt.show()


@torch.no_grad()
def test_mse_loss():
    loss_fn = ClippedMSELoss(reduction='none', min=0, max=4)

    x = torch.arange(-1, 5, 0.1)
    y0 = torch.tensor(0.0).expand_as(x)
    y1 = torch.tensor(1.0).expand_as(x)
    y2 = torch.tensor(2.0).expand_as(x)
    y3 = torch.tensor(3.0).expand_as(x)
    y4 = torch.tensor(4.0).expand_as(x)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y0)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y1)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y2)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y3)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y4)))
    plt.title(f'ClippedMSELoss')
    plt.show()
