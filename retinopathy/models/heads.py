import torch
from pytorch_toolbelt.modules.encoders import EncoderModule
from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d, GlobalMaxPool2d
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

    def __init__(self, features, num_classes, reduction=4, dropout=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

        bottleneck = features // reduction
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


class CLSBlock(nn.Module):
    """
    Block used for making final classification predictions
    """

    def __init__(self, features, num_classes, reduction=8):
        super().__init__()
        bottleneck = features // reduction

        self.bn1 = nn.BatchNorm1d(features)
        self.fc1 = nn.Linear(features, bottleneck, bias=False)

        self.bn2 = nn.BatchNorm1d(bottleneck)
        self.fc2 = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fc1(x)

        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x


class GlobalAvgPool2dHead(nn.Module):
    """Global average pooling classifier module"""

    def __init__(self, features):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class GlobalMaxPool2dHead(nn.Module):
    """Global max pooling classifier module"""

    def __init__(self, features):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.max_pool = GlobalMaxPool2d()

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        return x


class ObjectContextPoolHead(nn.Module):
    def __init__(self, features, oc_features, dropout=0.0):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = oc_features
        self.oc = ASP_OC_Module(features, oc_features, dropout=dropout, dilations=(3, 5, 7))
        self.max_pool = GlobalMaxPool2d()

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.oc(x)
        x = self.max_pool(x)
        return x


class GlobalMaxAvgPool2dHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features * 2
        self.avg_pool = GlobalAvgPool2d()
        self.max_pool = GlobalMaxPool2d()

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = x.view(x.size(0), -1)
        return x


class RMSPoolHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.rms_pool = RMSPool2d()

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.rms_pool(x)
        x = x.view(x.size(0), -1)
        return x


def regression_to_class(value: torch.Tensor, min=0, max=4):
    value = torch.round(value)
    value = torch.clamp(value, min, max)
    return value.long()


class EncoderHeadModel(nn.Module):
    def __init__(self, encoder: EncoderModule, head: nn.Module, num_classes=5,
                 num_regression_dims=1,
                 dropout=0.0,
                 reduction=8):
        super().__init__()
        self.encoder = encoder
        self.head = head
        bottleneck_features = head.features_size // reduction
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(head.features_size, bottleneck_features)

        self.regressor = FourReluBlock(bottleneck_features, num_regression_dims, reduction=1)
        self.logits = nn.Linear(bottleneck_features, num_classes)
        # self.ordinal = nn.Sequential(nn.Linear(bottleneck_features, num_classes),
        #                              LogisticCumulativeLink(num_classes, init_cutpoints='ordered'))

    @property
    def features_size(self):
        return self.head.features_size

    def forward(self, input):
        feature_maps = self.encoder(input)
        features = self.head(feature_maps)
        features = self.dropout(features)
        features = self.bottleneck(features)

        logits = self.logits(features)
        regression = self.regressor(features)
        # ordinal = self.ordinal(features)

        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            # 'ordinal': ordinal
        }


class PoolAndSqueeze(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.0):
        super().__init__()
        self.pool = RMSPool2d()
        self.dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(input_features, output_features)
        self.output_features = output_features

    def forward(self, input):
        features = self.pool(input)
        features = features.view(features.size(0), features.size(1))
        features = self.dropout(features)
        features = self.bottleneck(features)
        return features


class MultistageModel(nn.Module):
    def __init__(self, encoder: EncoderModule, pooling_module: nn.Module,
                 num_classes=5,
                 num_regression_dims=1,
                 dropout=0.0,
                 reduction=8):
        super().__init__()
        self.encoder = encoder
        self.pool1 = PoolAndSqueeze(encoder.output_filters[-1], encoder.output_filters[-1] // reduction, dropout)
        self.pool2 = PoolAndSqueeze(encoder.output_filters[-2], encoder.output_filters[-2] // reduction, dropout)
        self.pool3 = PoolAndSqueeze(encoder.output_filters[-3], encoder.output_filters[-3] // reduction, dropout)

        bottleneck_features = self.pool1.output_features + self.pool2.output_features + self.pool3.output_features
        self.regressor = FourReluBlock(bottleneck_features, num_regression_dims, reduction=1)
        self.logits = nn.Linear(bottleneck_features, num_classes)
        # self.ordinal = nn.Sequential(nn.Linear(bottleneck_features, num_classes),
        #                              LogisticCumulativeLink(num_classes, init_cutpoints='ordered'))

    @property
    def features_size(self):
        return self.head.features_size

    def forward(self, input):
        feature_maps = self.encoder(input)

        pool1 = self.pool1(feature_maps[-1])
        pool2 = self.pool2(feature_maps[-2])
        pool3 = self.pool3(feature_maps[-3])

        features = torch.cat([pool1, pool2, pool3], dim=1)

        logits = self.logits(features)
        regression = self.regressor(features)
        # ordinal = self.ordinal(features)

        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            # 'ordinal': ordinal
        }
