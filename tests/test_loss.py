import pytest
import torch
import torch.nn.functional as F
from pytorch_toolbelt.utils.torch_utils import to_numpy

from retinopathy.losses import ClippedMSELoss, ClippedHuber, MagnetLoss, HybridCappaLoss, ClippedWingLoss


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


@torch.no_grad()
@pytest.mark.parametrize(['width', 'curvature'], [
    # [5, 0.5],
    # [5, 0.1],
    # [1, 0.1],
    # [2, 0.5],
    # [2, 0.01],
    # [5, 0.9],
    # [10, 0.8],
    # [1, 0.1],
    # [2, 0.1],
    [4, 0.9],
    [4, 0.8],
    [4, 0.7],
    [4, 0.6],
    [4, 0.5],
    [4, 0.4],
    [4, 0.3],
    [4, 0.2],
    [4, 0.1]
])
def test_wing_loss(width, curvature):
    loss_fn = ClippedWingLoss(width, curvature, min=0, max=4, reduction='none')

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
    plt.title(f'ClippedWingLoss({width},{curvature})')
    plt.show()


def test_quad_kappa_loss():
    criterion = HybridCappaLoss()
    target = torch.tensor([0, 1]).long()
    loss_worst = criterion(torch.tensor([[0, 0, 0, 0, 10], [0, 10, 0, 0, 0]]).float(), target)
    loss_bad = criterion(torch.tensor([[10, 10, 0, 0, 0], [0, 10, 0, 0, 0]]).float(), target)
    loss_ideal = criterion(torch.tensor([[10, 0, 0, 0, 0], [0, 10, 0, 0, 0]]).float(), target)
    assert loss_ideal < loss_bad < loss_worst


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
