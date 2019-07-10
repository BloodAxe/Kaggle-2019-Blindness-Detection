import cv2
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SmoothL1Loss
from torchcontrib.optim import SWA

from retinopathy.lib.augmentations import CropBlackRegions
from retinopathy.lib.losses import ClippedMSELoss
from retinopathy.lib.models.classification import BaselineClassificationModel
from retinopathy.lib.models.regression import BaselineRegressionModel, STNRegressionModel
from pytorch_toolbelt.modules.encoders import *
from pytorch_toolbelt.losses import FocalLoss
from torch.optim import SGD, Adam
import albumentations as A


def get_model(model_name, num_classes, pretrained=True, **kwargs):
    if model_name == 'cls_resnet18':
        encoder = Resnet18Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes)

    if model_name == 'reg_resnet18':
        assert num_classes == 1
        encoder = Resnet18Encoder(pretrained=pretrained)
        return BaselineRegressionModel(encoder)

    if model_name == 'reg_stn_resnet18':
        assert num_classes == 1
        encoder = Resnet18Encoder(pretrained=pretrained)
        return STNRegressionModel(encoder, pretrained=pretrained)

    if model_name == 'cls_resnext50':
        encoder = SEResNeXt50Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes)

    if model_name == 'cls_resnext101':
        encoder = SEResNeXt101Encoder(pretrained=pretrained)
        return BaselineClassificationModel(encoder, num_classes)

    if model_name == 'cls_effnet_b4':
        encoder = EfficientNetB4Encoder()
        return BaselineClassificationModel(encoder, num_classes)

    raise ValueError(model_name)


def get_optimizable_parameters(model: nn.Module):
    return filter(lambda x: x.requires_grad, model.parameters())


def get_optimizer(optimizer_name: str, parameters, lr: float, weight_decay=1e-4, **kwargs):
    if optimizer_name.lower() == 'sgd':
        return SGD(parameters, lr, momentum=0.9, nesterov=True, weight_decay=weight_decay, **kwargs)

    if optimizer_name.lower() == 'adam':
        return Adam(parameters, lr, **kwargs)

    if optimizer_name.lower() == 'adam-swa':
        optimizer = Adam(parameters, lr, **kwargs)
        return SWA(optimizer)

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

    if loss_name.lower() == 'clipped_huber':
        raise NotImplementedError(loss_name)

    if loss_name.lower() == 'clipped_mse':
        return ClippedMSELoss(min=0, max=4)

    raise KeyError(loss_name)


def get_train_aug(image_size, augmentation=None):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),
        A.CoarseDropout(),
        A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),

        A.OneOf([
            A.ISONoise(),
            A.GaussNoise(),
            A.GaussianBlur(),
            A.IAASharpen(),
            A.NoOp()
        ]),

        A.Compose([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.05, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha_affine=5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
        ], p=float(augmentation in {'hard'})),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(hue_shift_limit=5),
            A.CLAHE(),
            A.RGBShift()
        ]),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ], p=float(augmentation in {'hard'})),

        # Horizontal/Vertical flips
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip()
        ], p=float(augmentation in {'medium'})),

        A.Normalize()
    ])


def get_test_aug(image_size):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(image_size[0], image_size[1], border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize()
    ])
