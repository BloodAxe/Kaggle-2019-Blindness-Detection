import torch

from retinopathy.models.common import regression_to_class


def test_round():
    x = torch.tensor([-0.9, -0.2, 0.2, 0.5, 0.7, 1.1, 1.4, 1.5, 1.6, 2.4, 2.5, 2.6, 3.3, 3.5, 3.9, 4, 4.5, 5])
    y = regression_to_class(x)
    print(x)
    print(y)
