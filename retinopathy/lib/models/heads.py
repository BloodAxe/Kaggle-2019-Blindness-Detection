import torch
from pytorch_toolbelt.modules.encoders import EncoderModule
from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d, GlobalMaxPool2d
from torch import nn
from torch.nn import functional as F

from retinopathy.lib.models.oc import ASP_OC_Module


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


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


class FourReluBlock(nn.Module):
    """
    Block used for making final regression predictions
    """

    def __init__(self, features, num_classes, reduction=4):
        super().__init__()
        self.bn = nn.BatchNorm1d(features)

        bottleneck = features // reduction
        self.fc1 = nn.Linear(features, bottleneck)
        self.fc2 = nn.Linear(bottleneck, bottleneck)
        self.fc3 = nn.Linear(bottleneck, bottleneck)
        self.fc4 = nn.Linear(bottleneck, num_classes)

    def forward(self, x):
        x = self.bn(x)

        x = self.fc1(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.fc2(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.fc3(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.fc4(x)
        x = F.leaky_relu(x, inplace=True)
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

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.avg_pool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(dropout)
        self.last_linear = head_block(features, num_classes)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.dropout(x)
        logits = self.last_linear(x)
        return features, logits


class GlobalMaxPool2dHead(nn.Module):
    """Global max pooling classifier module"""

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.max_pool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout)
        self.logits = head_block(features, num_classes)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.dropout(x)
        logits = self.logits(x)
        return features, logits


class GlobalWeightedAvgPool2dHead(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0, **kwargs):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.logits = head_block(features, num_classes)

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, feature_maps):
        x = feature_maps[-1]

        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = x.sum(dim=[2, 3])
        features = x
        x = self.dropout(x)
        logits = self.logits(x)
        return features, logits


class GlobalWeightedMaxPool2dHead(nn.Module):
    """
    Global Weighted Max Pooling from paper "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0, **kwargs):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.max_pool = GlobalMaxPool2d()
        self.logits = head_block(features, num_classes)

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, feature_maps):
        x = feature_maps[-1]

        input_x = x
        x = self.fscore(x)
        x = self.norm(x)
        x = x * input_x
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.dropout(x)
        logits = self.logits(x)
        return features, logits


class ObjectContextPoolHead(nn.Module):
    """
    """

    def __init__(self, features, num_classes, oc_features, head_block=nn.Linear, dropout=0.0, **kwargs):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = oc_features
        self.oc = ASP_OC_Module(features, oc_features, dropout=dropout, dilations=(3, 5, 7))
        self.max_pool = GlobalMaxPool2d()
        self.logits = head_block(oc_features, num_classes)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.oc(x)
        x = self.max_pool(x)
        features = x.view(x.size(0), -1)
        logits = self.logits(features)
        return features, logits


class GlobalMaxAvgPool2dHead(nn.Module):
    """Global average pooling classifier module"""

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.avg_pool = GlobalAvgPool2d()
        self.max_pool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout)
        self.last_linear = head_block(features, num_classes)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.avg_pool(x) + self.max_pool(x)
        x = x.view(x.size(0), -1)
        features = x
        x = self.dropout(x)
        logits = self.last_linear(x)
        return features, logits


class HyperPoolHead(nn.Module):
    """Global average pooling classifier module"""

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0):
        super().__init__()
        self.features_size = sum(features)
        self.max_pool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout)
        self.last_linear = head_block(self.features_size, num_classes)

    def forward(self, feature_maps):
        features = []
        for feature_map in feature_maps:
            x = self.max_pool(feature_map)
            x = x.view(x.size(0), -1)
            features.append(x)

        features = torch.cat(features, dim=1)
        x = self.dropout(features)
        logits = self.last_linear(x)
        return features, logits


class RMSPoolRegressionHead(nn.Module):
    def __init__(self, features, output_classes, reduction=4, dropout=0.25):
        super().__init__()
        if isinstance(features, list):
            features = features[-1]

        self.features_size = features
        self.rms_pool = RMSPool2d()
        self.bn = nn.BatchNorm1d(features)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.output_classes = output_classes

        bottleneck = features // reduction

        self.fc1 = nn.Linear(features, bottleneck)
        self.fc2 = nn.Linear(bottleneck, bottleneck)
        self.fc3 = nn.Linear(bottleneck, bottleneck)
        self.fc4 = nn.Linear(bottleneck, output_classes)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        x = self.rms_pool(x)
        features = x.view(x.size(0), -1)

        x = self.bn(features)
        x = self.drop(x)
        x = self.fc1(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.fc2(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.fc3(x)
        x = F.leaky_relu(x, inplace=True)

        logits = self.fc4(x)
        return features, logits


def regression_to_class(value: torch.Tensor, min=0, max=4):
    value = torch.round(value)
    value = torch.clamp(value, min, max)
    return value.long()


class EncoderHeadModel(nn.Module):
    def __init__(self, encoder: EncoderModule, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def features_size(self):
        return self.head.features_size

    def forward(self, input):
        feature_maps = self.encoder(input)
        features, logits = self.head(feature_maps)

        if logits.size(1) == 1:
            logits = logits.squeeze(1)

        return {'features': features, 'logits': logits}
