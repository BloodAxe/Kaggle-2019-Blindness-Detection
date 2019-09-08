from functools import partial

import torch.nn.functional as F
from catalyst.contrib.schedulers import OneCycleLR, ExponentialLR
from pytorch_toolbelt.modules import ABN
from pytorch_toolbelt.modules.encoders import *
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.rmsprop import RMSprop
from torchvision.models import densenet169, densenet121, densenet201

from retinopathy.losses import ClippedMSELoss, ClippedWingLoss, CumulativeLinkLoss, LabelSmoothingLoss, \
    SoftCrossEntropyLoss, ClippedHuber, CustomMSE, HybridCappaLoss, FocalLoss, WingLoss, RMSE, Huber, RegKappa, \
    CauchyLoss
from retinopathy.models.common import EncoderHeadModel
from retinopathy.models.dilated_senet import DilatedSEResNeXt50Encoder, DilatedSEResNeXt101Encoder
from retinopathy.models.efficientnet import EfficientNetB0ReLUEncoder, EfficientNetB7ReLUEncoder, \
    EfficientNetB6ReLUEncoder, EfficientNetB5ReLUEncoder, EfficientNetB4ReLUEncoder, EfficientNetB3ReLUEncoder, \
    EfficientNetB2ReLUEncoder, EfficientNetB1ReLUEncoder
from retinopathy.models.heads.fpn import FPNHeadModel
from retinopathy.models.heads.gap import GlobalAvgPoolHeadV2, GlobalAvgPoolHead
from retinopathy.models.heads.gwap import GlobalWeightedAvgPoolHead
from retinopathy.models.heads.max import GlobalMaxPoolHead, GlobalMaxPoolHeadV2
from retinopathy.models.heads.rank import RankPoolingHeadModelV2, RankPoolingHeadModel
from retinopathy.models.heads.rms import RMSPoolHead
from retinopathy.models.heads.rnn import RNNHead
from retinopathy.models.inceptionv4 import InceptionV4Encoder
from retinopathy.models.pnasnet import pnasnet5large
from retinopathy.opt import Lamb, AdamW, QHAdamW, RAdam, Ranger


class DenseNet121Encoder(EncoderModule):
    def __init__(self, pretrained=True):
        densenet = densenet121(pretrained=pretrained)
        super().__init__([1024], [32], [0])
        self.features = densenet.features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        return [x]


class DenseNet169Encoder(EncoderModule):
    def __init__(self, pretrained=True):
        densenet = densenet169(pretrained=pretrained)
        super().__init__([1664], [32], [0])
        self.features = densenet.features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        return [x]


class DenseNet201Encoder(EncoderModule):
    def __init__(self, pretrained=True):
        densenet = densenet201(pretrained=pretrained)
        super().__init__([1920], [32], [0])
        self.features = densenet.features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        return [x]


class PNasnet5LargeEncoder(EncoderModule):
    def __init__(self, pretrained=False):
        model = pnasnet5large(pretrained='imagenet+background' if pretrained else None)
        super().__init__([4320], [32], [0])
        model.last_linear = None  # Remove last linear block as we have our own
        self.extractor = model

    def forward(self, x):
        x = F.relu(self.extractor.features(x))
        return [x]


def get_model(model_name, num_classes, pretrained=True, dropout=0.0, **kwargs):
    keys = model_name.split('_')
    if len(keys) == 2:
        encoder_name, head_name = keys
        model = 'baseline'
    else:
        model, encoder_name, head_name = keys

    abn_block = ABN
    try:
        from inplace_abn import InPlaceABN
        abn_block = InPlaceABN
        print('Using InPlaceABN')
    except:
        print('InplaceABN not available, using classic BatchNorm+Act')

    ENCODERS = {
        'resnet18': Resnet18Encoder,
        'resnet34': Resnet34Encoder,
        'resnet50': Resnet50Encoder,
        'resnet101': Resnet101Encoder,
        'resnet152': Resnet152Encoder,
        'seresnext50': SEResNeXt50Encoder,
        'seresnext50d': partial(DilatedSEResNeXt50Encoder, dropout=0.25),
        'seresnext101': SEResNeXt101Encoder,
        'seresnext101d': partial(DilatedSEResNeXt101Encoder, dropout=0.25),
        'seresnet152': SEResnet152Encoder,
        'senet154': SENet154Encoder,
        'densenet121': DenseNet121Encoder,
        'densenet169': DenseNet169Encoder,
        'densenet201': DenseNet201Encoder,
        'inceptionv4': InceptionV4Encoder,
        'efficientb0': partial(EfficientNetB0ReLUEncoder, abn_block=abn_block),
        'efficientb1': partial(EfficientNetB1ReLUEncoder, abn_block=abn_block),
        'efficientb2': partial(EfficientNetB2ReLUEncoder, abn_block=abn_block),
        'efficientb3': partial(EfficientNetB3ReLUEncoder, abn_block=abn_block),
        'efficientb4': partial(EfficientNetB4ReLUEncoder, abn_block=abn_block),
        'efficientb5': partial(EfficientNetB5ReLUEncoder, abn_block=abn_block),
        'efficientb6': partial(EfficientNetB6ReLUEncoder, abn_block=abn_block),
        'efficientb7': partial(EfficientNetB7ReLUEncoder, abn_block=abn_block),
        'pnasnet5': PNasnet5LargeEncoder
    }

    encoder = ENCODERS[encoder_name](pretrained=pretrained)

    HEADS = {
        'gap': GlobalAvgPoolHead,
        'gapv2': GlobalAvgPoolHeadV2,
        'gwap': GlobalWeightedAvgPoolHead,
        'rms': RMSPoolHead,
        'max': GlobalMaxPoolHead,
        'maxv2': GlobalMaxPoolHeadV2,
        'fpn': FPNHeadModel,
        'rank': RankPoolingHeadModel,
        'rankv2': RankPoolingHeadModelV2,
        'rnn': RNNHead
    }

    MODELS = {
        'baseline': EncoderHeadModel,
    }

    head = HEADS[head_name](feature_maps=encoder.output_filters, num_classes=num_classes, dropout=dropout)

    model = MODELS[model](encoder, head)
    return model


def get_optimizable_parameters(model: nn.Module):
    return filter(lambda x: x.requires_grad, model.parameters())


def get_optimizer(optimizer_name: str, parameters, learning_rate: float, weight_decay=1e-5, **kwargs):
    if optimizer_name.lower() == 'sgd':
        return SGD(parameters, learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == 'adam':
        return Adam(parameters, learning_rate, weight_decay=weight_decay,
                    eps=1e-3,  # As Jeremy suggests
                    **kwargs)

    if optimizer_name.lower() == 'rms':
        return RMSprop(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == 'adamw':
        return AdamW(parameters, learning_rate, weight_decay=weight_decay,
                     eps=1e-3,
                     **kwargs)

    if optimizer_name.lower() == 'radam':
        return RAdam(parameters, learning_rate, weight_decay=weight_decay,
                     eps=1e-3,  # As Jeremy suggests
                     **kwargs)

    if optimizer_name.lower() == 'ranger':
        return Ranger(parameters, learning_rate, weight_decay=weight_decay,
                      **kwargs)

    if optimizer_name.lower() == 'qhadamw':
        return QHAdamW(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == 'lamb':
        return Lamb(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == 'lamb':
        return Lamb(parameters, learning_rate, weight_decay=weight_decay, **kwargs)

    raise ValueError("Unsupported optimizer name " + optimizer_name)


def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == 'bce':
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == 'ce':
        return CrossEntropyLoss(**kwargs)

    if loss_name.lower() == 'focal':
        return FocalLoss(alpha=None, **kwargs)

    if loss_name.lower() == 'mse':
        return CustomMSE(**kwargs)

    if loss_name.lower() == 'rmse':
        return RMSE(**kwargs)

    if loss_name.lower() == 'huber':
        return Huber(**kwargs)

    if loss_name.lower() == 'clipped_huber':
        return ClippedHuber(min=0, max=4, **kwargs)

    if loss_name.lower() == 'wing_loss':
        return WingLoss(width=4, curvature=0.3, **kwargs)

    if loss_name.lower() == 'clipped_wing_loss':
        return ClippedWingLoss(width=4, curvature=0.3, min=0, max=4, **kwargs)

    if loss_name.lower() == 'clipped_huber':
        raise NotImplementedError(loss_name)

    if loss_name.lower() == 'clipped_mse':
        return ClippedMSELoss(min=0, max=4, **kwargs)

    if loss_name.lower() == 'cauchy':
        return CauchyLoss(c=1.0, **kwargs)

    if loss_name.lower() == 'link':
        return CumulativeLinkLoss()

    if loss_name.lower() == 'smooth_kl':
        return LabelSmoothingLoss()

    if loss_name.lower() == 'soft_ce':
        return SoftCrossEntropyLoss(**kwargs)

    if loss_name.lower() == 'focal_kappa':
        return HybridCappaLoss(**kwargs)

    if loss_name.lower() == 'reg_kappa':
        return RegKappa(**kwargs)

    raise KeyError(loss_name)


def get_scheduler(scheduler_name: str,
                  optimizer,
                  lr,
                  num_epochs,
                  batches_in_epoch=None):
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None

    if scheduler_name.lower() in {'1cycle', 'one_cycle', 'onecycle'}:
        return OneCycleLR(optimizer,
                          lr_range=(lr, 1e-6, 1e-5),
                          num_steps=num_epochs,
                          warmup_fraction=0.05, decay_fraction=0.1)

    if scheduler_name.lower() == 'exp':
        return ExponentialLR(optimizer, gamma=0.95)

    if scheduler_name.lower() == 'multistep':
        return MultiStepLR(optimizer,
                           milestones=[
                               # int(num_epochs * 0.3),
                               int(num_epochs * 0.5),
                               int(num_epochs * 0.7),
                               int(num_epochs * 0.9),
                               int(num_epochs * 0.95)],
                           gamma=0.3)

    if scheduler_name.lower() == 'simple':
        # Reduce LR by factor of 10 after 1/3 of training and by factor of 10 more after 2/3 of training
        return MultiStepLR(optimizer,
                           milestones=[
                               int(num_epochs * 0.33),
                               int(num_epochs * 0.66)],
                           gamma=0.1)

    raise KeyError(scheduler_name)
