import torch
from catalyst.contrib.schedulers import OneCycleLR, ExponentialLR
from pytorch_toolbelt.losses import FocalLoss
from pytorch_toolbelt.modules.encoders import *
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SmoothL1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from retinopathy.lib.losses import ClippedMSELoss, ClippedWingLoss
from retinopathy.lib.models.classification import BaselineClassificationModel
from retinopathy.lib.models.regression import BaselineRegressionModel, \
    STNRegressionModel, MultipoolRegressionModel


def get_model(model_name, num_classes, pretrained=True, **kwargs):
    # Regression
    if model_name == 'reg_resnet18':
        assert num_classes == 1
        encoder = Resnet18Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder)

    if model_name == 'reg_resnet50':
        assert num_classes == 1
        encoder = Resnet50Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder, dropout=0.5)

    if model_name == 'reg_resnext50':
        assert num_classes == 1
        encoder = SEResNeXt50Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder, dropout=0.5)

    if model_name == 'reg_resnext101':
        assert num_classes == 1
        encoder = SEResNeXt101Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder, dropout=0.5)

    if model_name == 'reg_stn_resnet18':
        assert num_classes == 1
        encoder = Resnet18Encoder(pretrained=pretrained)
        return STNRegressionModel(encoder, pretrained=pretrained)

    if model_name == 'reg_resnext50':
        encoder = SEResNeXt50Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder, num_classes, dropout=0.5)

    if model_name == 'reg_resnext50_multi':
        encoder = SEResNeXt50Encoder(pretrained=pretrained)
        return MultipoolRegressionModel(encoder, num_classes, dropout=0.25)

    if model_name == 'reg_resnext101':
        encoder = SEResNeXt101Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder, num_classes, dropout=0.5)

    if model_name == 'reg_resnext101_multi':
        encoder = SEResNeXt101Encoder(pretrained=pretrained)
        return MultipoolRegressionModel(encoder, num_classes, dropout=0.5)

    if model_name == 'reg_effnet_b4':
        encoder = EfficientNetB4Encoder()
        return BaselineRegressionModel(encoder, num_classes)

    # Classification
    if model_name == 'cls_resnet18':
        encoder = Resnet18Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes)

    if model_name == 'cls_resnext50':
        encoder = SEResNeXt50Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes)

    if model_name == 'cls_resnext101':
        encoder = SEResNeXt101Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes)

    if model_name == 'cls_effnet_b4':
        encoder = EfficientNetB4Encoder()
        return BaselineClassificationModel(encoder, num_classes, dropout=0.5)

    if model_name == 'cls_resnext101':
        assert num_classes == 1
        encoder = SEResNeXt101Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes, dropout=0.5)

    raise ValueError(model_name)


def get_optimizable_parameters(model: nn.Module):
    return filter(lambda x: x.requires_grad, model.parameters())


def get_optimizer(optimizer_name: str, parameters, lr: float, weight_decay=1e-5, **kwargs):
    if optimizer_name.lower() == 'sgd':
        return SGD(parameters, lr, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == 'adam':
        return Adam(parameters, lr, **kwargs)

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
