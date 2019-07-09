import torch
from pytorch_toolbelt.modules.activations import swish
from pytorch_toolbelt.modules.encoders import *
from pytorch_toolbelt.modules.pooling import *
from pytorch_toolbelt.modules.scse import *
from torch import nn
import torch.nn.functional as F


class ClassifierModule(nn.Module):
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

        return logits, features


class BaselineClassificationModel(nn.Module):
    def __init__(self, encoder: EncoderModule, num_classes, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.classifier = ClassifierModule(encoder.output_filters[-1], num_classes, dropout=dropout)

    def forward(self, input):
        features = self.encoder(input)[-1]
        logits, features = self.classifier(features)
        return {'logits': logits, 'features': features}
