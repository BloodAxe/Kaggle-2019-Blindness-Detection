from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules.encoders import EfficientNetEncoder
from retinopathy.models.efficientnet_backbone import efficient_net_b0, efficient_net_b1, efficient_net_b2, \
    efficient_net_b3, efficient_net_b4, efficient_net_b5, efficient_net_b6, efficient_net_b7


class EfficientNetB0ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b0(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [16, 24, 40, 80, 112, 192, 320],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB1ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b1(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [16, 24, 40, 80, 112, 192, 320],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB2ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b2(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [16, 24, 48, 88, 120, 208, 352],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB3ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b3(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [24, 32, 48, 96, 136, 232, 384],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB4ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b4(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [24, 32, 56, 112, 160, 272, 448],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB5ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b5(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [24, 40, 64, 128, 176, 304, 512],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB6ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b6(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [32, 40, 72, 144, 200, 344, 576],
                         [2, 4, 8, 16, 16, 32, 32], layers)


class EfficientNetB7ReLUEncoder(EfficientNetEncoder):
    def __init__(self, activation='leaky_relu', layers=[1, 2, 4, 6], abn_block=ABN, pretrained=False):
        abn_params = {
            'activation': activation,
            'momentum': 0.1,
            'eps': 1e-5
        }
        super().__init__(efficient_net_b7(num_classes=1, abn_block=abn_block, abn_params=abn_params),
                         [32, 48, 80, 160, 224, 384, 640],
                         [2, 4, 8, 16, 16, 32, 32], layers)
