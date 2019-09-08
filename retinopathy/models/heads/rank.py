from torch import nn

from retinopathy.rank_pooling import GlobalRankPooling


class RankPoolingHeadModel(nn.Module):
    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = feature_maps[-1]
        self.rank_pool = GlobalRankPooling(self.features_size, 16 * 16)
        self.dropout = nn.AlphaDropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)

        # Regression to grade using SSD-like module
        self.regression = nn.Sequential(
            nn.Linear(self.features_size, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1)
        )

        self.ordinal = nn.Linear(self.features_size, num_classes - 1)

    def forward(self, features):
        features = self.rank_pool(features[-1])
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


class RankPoolingHeadModelV2(nn.Module):
    def __init__(self, feature_maps, num_classes: int, dropout=0.):
        super().__init__()
        self.features_size = 512
        self.bottleneck = nn.Conv2d(feature_maps[-1], self.features_size, kernel_size=1)
        self.rank_pool = GlobalRankPooling(self.features_size, 16 * 16)
        self.dropout = nn.AlphaDropout(dropout)
        self.logits = nn.Linear(self.features_size, num_classes)

        # Regression to grade using SSD-like module
        self.regression = nn.Sequential(
            nn.Linear(self.features_size, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 16),
            nn.ELU(inplace=True),
            nn.Linear(16, 1)
        )

        self.ordinal = nn.Linear(self.features_size, num_classes - 1)

    def forward(self, features):
        features = self.bottleneck(self.dropout(features[-1]))
        features = self.rank_pool(features)

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
