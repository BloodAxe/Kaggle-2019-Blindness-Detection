from functools import partial

import torch
from catalyst.contrib.schedulers import OneCycleLR, ExponentialLR
from pytorch_toolbelt.losses import FocalLoss
from pytorch_toolbelt.modules.encoders import *
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.rmsprop import RMSprop

from retinopathy.lib.losses import ClippedMSELoss, ClippedWingLoss, CumulativeLinkLoss, LabelSmoothingLoss, \
    SoftCrossEntropyLoss
from retinopathy.lib.models.heads import GlobalAvgPool2dHead, GlobalMaxPool2dHead, GlobalWeightedAvgPool2dHead, \
    GlobalWeightedMaxPool2dHead, ObjectContextPoolHead, \
    RMSPoolRegressionHead, GlobalMaxAvgPool2dHead, EncoderHeadModel, FourReluBlock, HyperPoolHead
from retinopathy.lib.models.ordinal import OrdinalEncoderHeadModel


def get_model(model_name, num_classes, pretrained=True, dropout=0.25, **kwargs):
    kind, encoder_name, head_name = model_name.split('_')

    ENCODERS = {
        'resnet18': Resnet18Encoder,
        'resnet50': Resnet50Encoder,
        'resnext50': SEResNeXt50Encoder,
        'resnext101': SEResNeXt101Encoder,
    }

    encoder = ENCODERS[encoder_name](layers=[1, 2, 3, 4], pretrained=pretrained)

    HEADS = {
        'gap': GlobalAvgPool2dHead,
        'gmp': GlobalMaxPool2dHead,
        'gwap': GlobalWeightedAvgPool2dHead,
        'gwmp': GlobalWeightedMaxPool2dHead,
        'ocp': partial(ObjectContextPoolHead, oc_features=encoder.output_filters[-1] // 4),
        'rms': RMSPoolRegressionHead,
        'maxavg': GlobalMaxAvgPool2dHead,
        'hyp': HyperPoolHead,
    }

    MODELS = {
        'reg': EncoderHeadModel,
        'cls': EncoderHeadModel,
        'ord': partial(OrdinalEncoderHeadModel, num_classes=num_classes)
    }

    if kind == 'reg' or kind == 'ord':
        num_classes = 1

    head_args = {
        'dropout': dropout
    }

    if kind == 'reg' and head_name != 'rms':
        head_args['head_block'] = partial(FourReluBlock, dropout=dropout)

    head = HEADS[head_name](encoder.output_filters, num_classes, **head_args)
    model = MODELS[kind](encoder, head)
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

    raise ValueError("Unsupported optimizer name " + optimizer_name)


def get_loss(loss_name: str, **kwargs):
    if loss_name.lower() == 'bce':
        return BCEWithLogitsLoss(**kwargs)

    if loss_name.lower() == 'ce':
        return CrossEntropyLoss(**kwargs)

    if loss_name.lower() == 'focal':
        return FocalLoss(**kwargs)

    if loss_name.lower() == 'mse':
        return MSELoss()

    if loss_name.lower() == 'huber':
        return SmoothL1Loss()

    if loss_name.lower() == 'wing_loss':
        return ClippedWingLoss(width=2, curvature=0.1, min=0, max=4)

    if loss_name.lower() == 'clipped_huber':
        raise NotImplementedError(loss_name)

    if loss_name.lower() == 'clipped_mse':
        return ClippedMSELoss(min=0, max=4)

    if loss_name.lower() == 'link':
        return CumulativeLinkLoss()

    if loss_name.lower() == 'smooth_kl':
        return LabelSmoothingLoss()

    if loss_name.lower() == 'soft_ce':
        return SoftCrossEntropyLoss()

    raise KeyError(loss_name)


def get_scheduler(scheduler_name: str,
                  optimizer,
                  lr,
                  num_epochs,
                  batches_in_epoch=None):
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None

    if scheduler_name.lower() in {'1cycle', 'one_cycle'}:
        return OneCycleLR(optimizer,
                          lr_range=(lr, 1e-6, 1e-5),
                          num_steps=batches_in_epoch,
                          warmup_fraction=0.05, decay_fraction=0.1)

    if scheduler_name.lower() == 'exp':
        return ExponentialLR(optimizer, gamma=0.95)

    if scheduler_name.lower() == 'multistep':
        return MultiStepLR(optimizer,
                           milestones=[
                               # int(num_epochs * 0.1),
                               # int(num_epochs * 0.2),
                               int(num_epochs * 0.3),
                               int(num_epochs * 0.4),
                               int(num_epochs * 0.5),
                               int(num_epochs * 0.6),
                               int(num_epochs * 0.7),
                               int(num_epochs * 0.8),
                               int(num_epochs * 0.9)],
                           gamma=0.75)

    raise KeyError(scheduler_name)


@torch.no_grad()
def test_wing_loss():
    loss_fn = ClippedWingLoss(width=2, curvature=0.1, reduction=None, min=0,
                              max=4)
    # loss_fn = WingLoss(width=2, curvature=0.1, reduction=None)
    x = torch.arange(-1, 5, 0.1)
    y0 = torch.tensor(0.0).expand_as(x)
    y1 = torch.tensor(1.0).expand_as(x)
    y2 = torch.tensor(2.0).expand_as(x)
    y3 = torch.tensor(3.0).expand_as(x)
    y4 = torch.tensor(4.0).expand_as(x)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y0)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y1)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y2)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y3)))
    plt.plot(to_numpy(x), to_numpy(loss_fn(x, y4)))
    plt.show()
