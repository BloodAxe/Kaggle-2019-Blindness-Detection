from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d
from torch import nn
from torch.nn import functional as F

from retinopathy.models.common import Flatten


class GlobalAvgPoolHead(nn.Module):
    """
    1) Squeeze last feature map in num_classes
    2) Compute global average
    """

    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Conv2d(self.features_size, num_classes, kernel_size=1)

        # Regression to grade using SSD-like module
        self.regression = nn.Sequential(
            GlobalAvgPool2d(),
            Flatten(),
            nn.Linear(self.features_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU6()
        )

        self.ordinal = nn.Sequential(
            GlobalAvgPool2d(),
            Flatten(),
            nn.Linear(self.features_size, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, num_classes - 1))

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

        ordinal = self.ordinal(features).sigmoid().sum(dim=1)

        return {
            'features': features.mean(dim=(2, 3)),
            'logits': logits,
            'regression': regression,
            'ordinal': ordinal
        }


class GlobalAvgPoolHeadV2(nn.Module):
    """
    """

    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.avgpool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)
        self.regression = nn.Linear(self.features_size, 1)
        self.ordinal = nn.Linear(self.features_size, num_classes - 1)

    def forward(self, feature_maps):
        # Take last feature map
        features = self.avgpool(feature_maps[-1])
        features = features.view(features.size(0), features.size(1))
        features = self.dropout(features)

        # Squeeze to num_classes
        logits = self.logits(features)
        regression = (self.regression(features) + 2.).log()
        ordinal = self.ordinal(features).sigmoid().sum(dim=1)

        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            'ordinal': ordinal
        }
