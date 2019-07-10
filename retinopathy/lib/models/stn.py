import torch
from pytorch_toolbelt.modules.pooling import GlobalMaxPool2d
from torch import nn
from pytorch_toolbelt.modules.encoders import Resnet18Encoder
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class STN(nn.Module):
    def __init__(self, pretrained=True):
        super(STN, self).__init__()
        encoder = Resnet18Encoder(pretrained=pretrained, layers=[0, 1])

        self.features = encoder.output_filters[1]

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            encoder.layer0,
            encoder.layer1,
            GlobalMaxPool2d(),
            Flatten()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.features, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                     0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.features)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
