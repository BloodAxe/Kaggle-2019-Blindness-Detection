import torch
from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d
from torch import nn


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


class RMSPoolHead(nn.Module):
    """
    """

    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.rms_pooling = RMSPool2d(self.features_size)
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)

        self.regression = nn.Sequential(
            nn.Linear(self.features_size, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1),
        )

        self.ordinal = nn.Sequential(
            nn.Linear(self.features_size, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, num_classes - 1),
        )

    def forward(self, feature_maps):
        # Take last feature map
        features = feature_maps[-1]
        features = self.rms_pooling(features)
        features = features.view(features.size(0), features.size(1))
        features = self.dropout(features)

        logits = self.logits(features)
        regression = self.regression(features)
        ordinal = self.ordinal(features).sigmoid().sum(dim=1)

        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            'ordinal': ordinal
        }
