import numpy as np
import pytorch_toolbelt.inference.functional as FF
import torch
from pytorch_toolbelt.modules.encoders import EncoderModule
from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d, GlobalMaxPool2d, GWAP
from torch import nn
from torch.nn import functional as F

from retinopathy.models.oc import ASP_OC_Module


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class RMSPool2d(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return (avg_pool + self.eps).sqrt()


class FourReluBlock(nn.Module):
    """
    Block used for making final regression predictions
    """

    def __init__(self, features, bottleneck, num_classes, dropout=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

        self.fc1 = nn.Linear(features, bottleneck)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(bottleneck, bottleneck)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(bottleneck, bottleneck)
        self.act3 = nn.LeakyReLU(inplace=True)

        self.fc4 = nn.Linear(bottleneck, num_classes)
        self.act4 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)

        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.act3(x)

        x = self.fc4(x)
        x = self.act4(x)

        return x


def regression_to_class(value: torch.Tensor, min=0, max=4):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if isinstance(value, (int, float)):
        value = torch.tensor(value)
    value = torch.round(value)
    value = torch.clamp(value, min, max)
    return value.long()


class GlobalAvgPoolHead(nn.Module):
    """
    1) Squeeze last feature map in num_classes
    2) Compute global average
    """

    def __init__(self, feature_maps, num_classes:int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Conv2d(self.features_size, num_classes, kernel_size=1)

        # Regression to grade using SSD-like module
        self.regression = nn.Sequential(
            nn.Conv2d(self.features_size, 16, kernel_size=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            GlobalAvgPool2d(),
            Flatten()
        )

    def forward(self, feature_maps):
        # Take last feature map
        features = feature_maps[-1]

        features = self.dropout(features)

        # Squeeze to num_classes
        logits = self.bottleneck(features)
        # Compute average
        logits = F.adaptive_avg_pool2d(logits, output_size=1)
        # Flatten
        logits = logits.view(logits.size(0), logits.size(1))

        regression = self.regression(features)
        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression
        }


class GlobalWeightedAvgPoolHead(nn.Module):
    """
    1) Squeeze last feature map in num_classes
    2) Compute global average
    """

    def __init__(self, feature_maps, num_classes:int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.gwap = GWAP(self.features_size)
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)

        # Regression to grade using SSD-like module
        self.regression = nn.Sequential(
            nn.Conv2d(self.features_size, 16, kernel_size=1, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            GlobalAvgPool2d(),
            Flatten()
        )

    def forward(self, feature_maps):
        # Take last feature map
        features = feature_maps[-1]
        features = self.gwap(features)
        features = features.view(features.size(0), features.size(1))
        features = self.dropout(features)

        logits = self.logits(features)

        regression = self.regression(features)
        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression
        }


class EncoderHeadModel(nn.Module):
    def __init__(self, encoder: EncoderModule, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def features_size(self):
        return self.head.features_size

    def forward(self, image):
        feature_maps = self.encoder(image)
        result = self.head(feature_maps)
        return result
