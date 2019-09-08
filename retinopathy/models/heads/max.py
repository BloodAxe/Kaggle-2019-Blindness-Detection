from pytorch_toolbelt.modules.pooling import GlobalMaxPool2d
from torch import nn


class GlobalMaxPoolHeadV2(nn.Module):
    """
    """

    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.maxpool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)
        self.regression = nn.Sequential(nn.Linear(self.features_size, self.features_size // 4),
                                        nn.BatchNorm1d(self.features_size // 4),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.features_size // 4, self.features_size // 8),
                                        nn.BatchNorm1d(self.features_size // 8),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.features_size // 8, 1))

        self.ordinal = nn.Sequential(nn.Linear(self.features_size, self.features_size // 4),
                                     nn.BatchNorm1d(self.features_size // 4),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.features_size // 4, self.features_size // 8),
                                     nn.BatchNorm1d(self.features_size // 8),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.features_size // 8, num_classes - 1))

    def forward(self, feature_maps):
        # Take last feature map
        features = self.maxpool(feature_maps[-1])
        features = features.view(features.size(0), features.size(1))
        features = self.dropout(features)

        # Squeeze to num_classes
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


class GlobalMaxPoolHead(nn.Module):
    """
    1) Squeeze last feature map in num_classes
    2) Compute global average
    """

    def __init__(self, feature_maps, num_classes: int, dropout=0., reduction=8):
        super().__init__()
        self.features_size = feature_maps[-1] // reduction
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_maps[-1], self.features_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.features_size),
            nn.ReLU(inplace=True))

        self.maxpool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)

        # Regression to grade using SSD-like module
        self.regression = nn.Sequential(
            nn.Linear(self.features_size, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1),
            nn.ELU(inplace=True),
        )

        self.ordinal = nn.Sequential(
            nn.Linear(self.features_size, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, num_classes - 1),
        )

    def forward(self, feature_maps):
        # Take last feature map
        features = feature_maps[-1]
        features = self.bottleneck(features)
        features = self.maxpool(features)
        features = features.view(features.size(0), features.size(1))
        features = self.dropout(features)

        logits = self.logits(features)

        regression = self.regression(features)
        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        ordinal = self.ordinal(features).sigmoid().sum(dim=1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            'ordinal': ordinal
        }
