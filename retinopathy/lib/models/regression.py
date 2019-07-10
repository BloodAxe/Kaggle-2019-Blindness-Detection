import torch
from pytorch_toolbelt.modules.activations import swish
from pytorch_toolbelt.modules.encoders import *
from pytorch_toolbelt.modules.pooling import *
from pytorch_toolbelt.modules.scse import *
from torch import nn
import torch.nn.functional as F

from retinopathy.lib.models.stn import STN


class RMSPool2d(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return avg_pool.sqrt()


class RegressionModule(nn.Module):
    def __init__(self, input_features, output_classes, reduction=4, dropout=0.25):
        super().__init__()
        self.rms_pool = RMSPool2d()
        self.bn = nn.BatchNorm1d(input_features)
        self.drop = nn.Dropout(dropout)
        self.output_classes = output_classes

        bottleneck = input_features // reduction

        self.fc1 = nn.Linear(input_features, bottleneck)
        self.fc2 = nn.Linear(bottleneck, bottleneck)
        self.fc3 = nn.Linear(bottleneck, bottleneck)
        self.fc4 = nn.Linear(bottleneck, output_classes)

    def forward(self, input):
        x = self.rms_pool(input)
        features = x.view(x.size(0), -1)

        x = self.bn(features)
        x = self.drop(x)
        x = self.fc1(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        x = F.leaky_relu(x)

        x = self.fc3(x)
        x = F.leaky_relu(x)

        logits = self.fc4(x)
        if self.output_classes == 1:
            logits = logits.squeeze(1)

        return logits, features


class BaselineRegressionModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_dimensions=1, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.regressor = RegressionModule(encoder.output_filters[-1], num_dimensions, dropout=dropout)

    def forward(self, input):
        features = self.encoder(input)[-1]
        logits, features = self.regressor(features)
        return {'logits': logits, 'features': features}


class STNRegressionModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_dimensions=1, dropout=0.2, pretrained=True):
        super().__init__()
        features = encoder.output_filters[-1]
        self.stn = STN(features)
        self.encoder = encoder
        self.regressor = RegressionModule(features, num_dimensions, dropout=dropout)

    def forward(self, input):
        features = self.encoder(input)[-1]
        input_transformed = self.stn(input, features)
        features = self.encoder(input_transformed)[-1]
        logits, features = self.regressor(features)
        return {'logits': logits, 'features': features, 'stn': input_transformed}


def regression_to_class(value: torch.Tensor, min=0, max=4):
    value = torch.round(value)
    value = torch.clamp(value, min, max)
    return value.long()


def test_round():
    x = torch.tensor([-0.9, -0.2, 0.2, 0.5, 0.7, 1.1, 1.4, 1.5, 1.6, 2.4, 2.5, 2.6, 3.3, 3.5, 3.9, 4, 4.5, 5])
    y = regression_to_class(x)
    print(x)
    print(y)
