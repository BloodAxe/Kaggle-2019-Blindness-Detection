from pytorch_toolbelt.modules.pooling import GWAP
from torch import nn


class GlobalWeightedAvgPoolHead(nn.Module):
    """
    1) Squeeze last feature map in num_classes
    2) Compute global average
    """

    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.gwap = GWAP(self.features_size)
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
        features = self.gwap(features)
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

