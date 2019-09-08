import torch
from pytorch_toolbelt.utils.torch_utils import count_parameters
from torch import nn


class GlobalRankPooling(nn.Module):
    def __init__(self, num_features, spatial_size):
        super().__init__()
        self.conv = nn.Conv1d(num_features, num_features, spatial_size, groups=num_features)

    def forward(self, x: torch.Tensor):
        spatial_size = x.size(2) * x.size(3)
        assert spatial_size == self.conv.kernel_size[0], f'Expected spatial size {self.conv.kernel_size[0]}, ' \
                                                         f'got {x.size(2)}x{x.size(3)}'

        x = x.view(x.size(0), x.size(1), -1)  # Flatten spatial dimensions
        x_sorted, index = x.topk(spatial_size, dim=2)

        x = self.conv(x_sorted)  # [B, C, 1]
        return x.squeeze(2)


def test_rank_pooling():
    x = torch.randn((4, 2048, 32, 32), dtype=torch.float)
    net = GlobalRankPooling(2048, 32 * 32)
    y = net(x)
    print(count_parameters(net))
    print(y.size())
