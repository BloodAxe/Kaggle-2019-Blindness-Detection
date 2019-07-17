import torch
from pytorch_toolbelt.modules.activations import swish
from pytorch_toolbelt.modules.encoders import EncoderModule
from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d, GlobalMaxPool2d
from torch import nn
from torch.nn import functional as F

from retinopathy.lib.models.oc import ASP_OC_Module


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

    def __init__(self, features, num_classes, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(features, features // 2)
        self.fc2 = nn.Linear(features // 2, features // 4)
        self.fc3 = nn.Linear(features // 4, features // 8)
        self.fc4 = nn.Linear(features // 8, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.drop(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.drop(x)

        x = self.fc3(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.drop(x)

        x = self.fc4(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.drop(x)
        return x


class GlobalAvgPool2dHead(nn.Module):
    """Global average pooling classifier module"""

    def __init__(self, features, num_classes, head_block=nn.Linear, dropout=0.0):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()
        self.dropout = nn.Dropout(dropout)
        self.last_linear = head_block(features, num_classes)

    def forward(self, x):
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
        self.max_pool = GlobalMaxPool2d()
        self.dropout = nn.Dropout(dropout)
        self.logits = head_block(features, num_classes)

    def forward(self, x):
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
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.logits = head_block(features, num_classes)

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        return x / x.sum(dim=[2, 3], keepdim=True)

    def forward(self, x):
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

    def forward(self, x):
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
        self.oc = ASP_OC_Module(features, oc_features, dropout=dropout, dilations=(3, 5, 7))
        self.max_pool = GlobalMaxPool2d()
        self.logits = head_block(oc_features, num_classes)

    def forward(self, x):
        x = self.oc(x)
        x = self.max_pool(x)
        features = x.view(x.size(0), -1)
        logits = self.logits(features)
        return features, logits


class GlobalMaxAvgPool2dHead(nn.Module):
    def __init__(self, input_features, output_classes, reduction=4, dropout=0.25):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()
        self.max_pool = GlobalMaxPool2d()
        self.bn = nn.BatchNorm1d(input_features * 2)
        self.drop = nn.Dropout(dropout)

        self.bottleneck = nn.Linear(input_features * 2, input_features // reduction)
        self.logits = nn.Linear(input_features // reduction, output_classes)

    def forward(self, input):
        x1 = self.avg_pool(input)
        x2 = self.max_pool(input)
        x = torch.cat([x1, x2], dim=1)
        features = x.view(x.size(0), -1)

        x = self.bn(features)
        x = swish(x)
        x = self.drop(x)
        x = self.bottleneck(x)
        x = swish(x)
        logits = self.logits(x)

        return features, logits


class RMSPoolRegressionHead(nn.Module):
    def __init__(self, input_features, output_classes, reduction=4, dropout=0.25):
        super().__init__()
        self.rms_pool = RMSPool2d()
        self.bn = nn.BatchNorm1d(input_features)
        self.drop = nn.Dropout(dropout, inplace=True)
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
        x = F.leaky_relu(x, inplace=True)

        x = self.fc2(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.fc3(x)
        x = F.leaky_relu(x, inplace=True)

        logits = self.fc4(x)
        if self.output_classes == 1:
            logits = logits.squeeze(1)

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

    def forward(self, input):
        features = self.encoder(input)[-1]
        features, logits = self.head(features)
        return {'features': features, 'logits': logits}
