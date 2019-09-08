from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules.coord_conv import AddCoords
from pytorch_toolbelt.modules.decoders import FPNDecoder
from pytorch_toolbelt.modules.fpn import FPNBottleneckBlockBN
from pytorch_toolbelt.modules.hypercolumn import HyperColumn
from pytorch_toolbelt.modules.pooling import GlobalAvgPool2d, GlobalMaxPool2d
from torch import nn


class CoordDoubleConvBNRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int, abn_block=ABN):
        super().__init__()
        self.add_coords = AddCoords(with_r=True)

        self.conv1 = nn.Conv2d(in_dec_filters + 3, out_filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn1 = abn_block(out_filters)

        self.conv2 = nn.Conv2d(out_filters + 3, out_filters, kernel_size=3, padding=1, stride=1, bias=False)
        self.abn2 = abn_block(out_filters)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv1(x)
        x = self.abn1(x)

        x = self.add_coords(x)
        x = self.conv2(x)
        x = self.abn2(x)
        return x


class FPNHeadModel(nn.Module):
    def __init__(self, feature_maps, num_classes: int, dropout=0., reduction=8):
        super().__init__()
        self.decoder = FPNDecoder(features=feature_maps[1:],
                                  bottleneck=FPNBottleneckBlockBN,
                                  prediction_block=CoordDoubleConvBNRelu,
                                  fpn_features=128,
                                  prediction_features=128)

        self.hypercolumn = HyperColumn(mode='nearest',align_corners=None)
        self.maxpool = GlobalMaxPool2d()
        self.features_size = sum(self.decoder.output_filters)
        self.logits = nn.Linear(self.features_size, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.logits = nn.Linear(self.features_size, num_classes)
        self.regression = nn.Sequential(nn.Linear(self.features_size, self.features_size // 4),
                                        nn.BatchNorm1d(self.features_size // 4),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.features_size // 4, self.features_size // 8),
                                        nn.BatchNorm1d(self.features_size // 8),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.features_size // 8, 1))

        self.ordinal = nn.Sequential(nn.Linear(self.features_size, self.features_size),
                                     nn.BatchNorm1d(self.features_size),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(self.features_size, self.features_size),
                                     nn.BatchNorm1d(self.features_size),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(self.features_size, num_classes - 1),
                                     nn.Sigmoid())

    def forward(self, features):
        features = self.decoder(features[1:])
        features = self.hypercolumn(*features)
        features = self.maxpool(features)
        features = features.view(features.size(0), features.size(1))

        features = self.dropout(features)

        logits = self.logits(features)

        regression = self.regression(features)
        if regression.size(1) == 1:
            regression = regression.squeeze(1)

        ordinal = self.ordinal(features).sum(dim=1)

        return {
            'features': features,
            'logits': logits,
            'regression': regression,
            'ordinal': ordinal
        }
