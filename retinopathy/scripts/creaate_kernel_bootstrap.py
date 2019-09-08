import base64
import inspect
import os

from pytorch_toolbelt.utils.fs import read_rgb_image, id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

from retinopathy.augmentations import get_test_transform
from retinopathy.dataset import RetinopathyDataset, get_class_names, UNLABELED_CLASS
from retinopathy.factory import get_model, DenseNet121Encoder, DenseNet201Encoder, DenseNet169Encoder, \
    PNasnet5LargeEncoder
from retinopathy.inference import run_model_inference, cls_predictions_to_submission, \
    average_predictions, reg_predictions_to_submission, ApplySoftmaxToLogits, \
    FlipLRMultiheadTTA, image_with_name_in_dir, run_models_inference_via_dataset, run_models_inference, Flip4MultiheadTTA, MultiscaleFlipLRMultiheadTTA
from retinopathy.models.dilated_senet import DilatedSEResNeXt50Encoder, dilated_se_resnext50_32x4d, SENetD, \
    SEBottleneckD, SEResNetBottleneckD, SEResNeXtBottleneckD, drop_connect, SEModule, \
    initialize_pretrained_model_dilated, BottleneckD, DilatedSEResNeXt101Encoder, dilated_se_resnext101_32x4d
from retinopathy.models.efficientnet import EfficientNetB7ReLUEncoder, EfficientNetB6ReLUEncoder, \
    EfficientNetB5ReLUEncoder, EfficientNetB4ReLUEncoder, EfficientNetB3ReLUEncoder, EfficientNetB2ReLUEncoder, \
    EfficientNetB1ReLUEncoder, EfficientNetB0ReLUEncoder
from retinopathy.models.common import EncoderHeadModel, Flatten
from retinopathy.models.heads.fpn import FPNHeadModel, CoordDoubleConvBNRelu
from retinopathy.models.heads.gap import GlobalAvgPoolHeadV2, GlobalAvgPoolHead
from retinopathy.models.heads.gwap import GlobalWeightedAvgPoolHead
from retinopathy.models.heads.max import GlobalMaxPoolHeadV2, GlobalMaxPoolHead
from retinopathy.models.heads.rank import RankPoolingHeadModel, RankPoolingHeadModelV2
from retinopathy.models.heads.rms import RMSPoolHead, RMSPool2d
from retinopathy.models.heads.rnn import RNNHead, LSTMBottleneck
from retinopathy.models.inceptionv4 import InceptionV4Encoder, inceptionv4, InceptionV4, Inception_B, Inception_A, \
    Inception_C, Reduction_A, Mixed_5a, Mixed_3a, Mixed_4a, BasicConv2d, Reduction_B
from retinopathy.models.oc import ASP_OC_Module, BaseOC_Context_Module, SelfAttentionBlock2D, _SelfAttentionBlock
from retinopathy.models.ordinal import LogisticCumulativeLink, OrdinalEncoderHeadModel
from retinopathy.models.pnasnet import pnasnet5large
from retinopathy.models.regression import regression_to_class
from retinopathy.preprocessing import CropBlackRegions, crop_black, ChannelIndependentCLAHE, \
    clahe_preprocessing, get_preprocessing_transform, UnsharpMask, unsharp_mask, RedFree, red_free, UnsharpMaskV2, \
    unsharp_mask_v2
from retinopathy.rank_pooling import GlobalRankPooling
from retinopathy.rounder import OptimizedRounder
from retinopathy.scripts.preprocess_data import preprocess, convert_dir
from retinopathy.train_utils import report_checkpoint


def encode_archive(archive_name):
    with open(archive_name, "rb") as f:
        encodedZip = base64.b64encode(f.read())
        return str(encodedZip, "utf-8")


def decode_archive(archive_name, content):
    with open(archive_name, "wb") as f:
        f.write(base64.b64decode(content))


def main():
    lines = []
    lines.append('import base64\n')
    lines.append('import os\n')
    lines.append('\n')

    # Bootstrap scripts
    for function in [decode_archive]:
        source = inspect.getsource(function)
        lines.extend(source)
        lines.append('\n')

    # Install packages
    for file in ['deps/pytorch_toolbelt-0.1.3.tar.gz']:
        file_name = os.path.basename(file)
        content = encode_archive(file)
        lines.append(f'decode_archive(\'{file_name}\', \'{content}\')\n')
        lines.append(f'os.system(\'pip install {file_name}\')\n')
        lines.append('\n')

    # Main dependencies
    imports = [
        'import os',
        'import math',
        'import cv2',
        'import torch',
        'import pandas as pd',
        'import numpy as np',
        'import multiprocessing',
        'import albumentations as A',
        'from tqdm import tqdm',
        'from torch.utils.data import Dataset',
        'from torch import nn',
        'from functools import partial',
        'from pytorch_toolbelt.utils import fs',
        'from pytorch_toolbelt.utils.torch_utils import to_numpy',
        'from torch.utils.data import DataLoader',
        'from pytorch_toolbelt.inference.tta import *',
        'from pytorch_toolbelt.modules.encoders import *',
        'from pytorch_toolbelt.modules.activations import swish',
        'from pytorch_toolbelt.modules.pooling import *',
        'from pytorch_toolbelt.modules.scse import *',
        'import torch.nn.functional as F',
        'from pytorch_toolbelt.modules import ABN',
        'from torch.autograd import Variable',
        'from torchvision.models import densenet169, densenet121, densenet201',
        'import torch.utils.model_zoo as model_zoo',
        'from multiprocessing.pool import Pool',
        'from collections import OrderedDict, defaultdict',
        'from albumentations.augmentations.functional import longest_max_size',
        'import pytorch_toolbelt.inference.functional as FF',
        'from pytorch_toolbelt.modules.backbone.efficient_net import efficient_net_b0, efficient_net_b1, efficient_net_b2, efficient_net_b3, efficient_net_b4, efficient_net_b5, efficient_net_b6, efficient_net_b7',
        'from pytorch_toolbelt.modules.encoders import EfficientNetEncoder',
        'from typing import List',
        'from skimage.measure import label',
        'from skimage.morphology import remove_small_objects',
        'from pytorch_toolbelt.modules.decoders import FPNDecoder',
        'from pytorch_toolbelt.modules.fpn import FPNBottleneckBlockBN',
        'from pytorch_toolbelt.modules.hypercolumn import HyperColumn',
        'from pytorch_toolbelt.modules.coord_conv import AddCoords, append_coords',
        'from sklearn import metrics',
        'import scipy as sp',
        'from scipy.stats import trim_mean'
    ]
    functions = [
        tensor_from_rgb_image,
        id_from_fname,
        read_rgb_image,
        get_class_names,
        RetinopathyDataset,
        RMSPool2d,
        DenseNet121Encoder,
        DenseNet169Encoder,
        DenseNet201Encoder,
        drop_connect,
        initialize_pretrained_model_dilated,
        SEModule,
        BottleneckD,
        SEBottleneckD,
        SEResNetBottleneckD,
        SEResNeXtBottleneckD,
        SENetD,
        dilated_se_resnext50_32x4d,
        DilatedSEResNeXt50Encoder,
        dilated_se_resnext101_32x4d,
        DilatedSEResNeXt101Encoder,
        GlobalWeightedAvgPoolHead,
        GlobalAvgPoolHead,
        BasicConv2d,
        Mixed_3a,
        Mixed_4a,
        Mixed_5a,
        Reduction_A,
        Inception_A,
        Inception_B,
        Reduction_B,
        Inception_C,
        InceptionV4,
        inceptionv4,
        InceptionV4Encoder,
        LSTMBottleneck,

        # EfficientNet
        EfficientNetB0ReLUEncoder,
        EfficientNetB1ReLUEncoder,
        EfficientNetB2ReLUEncoder,
        EfficientNetB3ReLUEncoder,
        EfficientNetB4ReLUEncoder,
        EfficientNetB5ReLUEncoder,
        EfficientNetB6ReLUEncoder,
        EfficientNetB7ReLUEncoder,
        # PNasnet5Large
        # PNASNet5Large,
        pnasnet5large,

        # Heads
        CoordDoubleConvBNRelu,
        PNasnet5LargeEncoder,
        FlipLRMultiheadTTA,
        Flip4MultiheadTTA,
        MultiscaleFlipLRMultiheadTTA,
        ApplySoftmaxToLogits,
        OrdinalEncoderHeadModel,
        GlobalRankPooling,
        GlobalAvgPoolHeadV2,
        GlobalMaxPoolHeadV2,
        RankPoolingHeadModel,
        RankPoolingHeadModelV2,
        LogisticCumulativeLink,
        RMSPoolHead,
        RNNHead,
        GlobalMaxPoolHead,
        FPNHeadModel,
        Flatten,
        EncoderHeadModel,
        crop_black,
        CropBlackRegions,
        unsharp_mask,
        UnsharpMask,
        clahe_preprocessing,
        ChannelIndependentCLAHE,
        unsharp_mask_v2,
        UnsharpMaskV2,
        RedFree,
        red_free,
        get_model,
        get_preprocessing_transform,
        get_test_transform,
        preprocess,
        convert_dir,
        report_checkpoint,
        run_models_inference_via_dataset,
        run_models_inference,
        average_predictions,
        cls_predictions_to_submission,
        reg_predictions_to_submission,
        regression_to_class,
        image_with_name_in_dir,
        OptimizedRounder
    ]

    lines.append('# Imports\n')
    for import_statement in imports:
        lines.append(import_statement + '\n')
    lines.append('\n')
    lines.append(f'UNLABELED_CLASS = {UNLABELED_CLASS}\n')
    lines.append('pretrained_settings = None\n')
    lines.append('pretrained_settings_dilated = None\n')

    lines.append('# Functions\n')
    for function in functions:
        source = inspect.getsource(function)
        lines.append('\n')
        lines.append('\n')
        lines.extend(source)

    with open('kernel_bootstrap.py', 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
