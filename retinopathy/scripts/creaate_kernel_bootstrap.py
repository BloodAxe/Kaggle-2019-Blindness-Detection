import base64
import inspect
import os

from pytorch_toolbelt.utils.fs import read_rgb_image, id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

from retinopathy.augmentations import CropBlackRegions, get_test_transform, crop_black, ChannelIndependentCLAHE, \
    clahe_preprocessing
from retinopathy.dataset import RetinopathyDataset, get_class_names, UNLABELED_CLASS
from retinopathy.factory import get_model, DenseNet121Encoder, DenseNet201Encoder, DenseNet169Encoder
from retinopathy.inference import PickModelOutput, run_model_inference, cls_predictions_to_submission, \
    average_predictions, reg_predictions_to_submission, run_model_inference_via_dataset
from retinopathy.models import OrdinalEncoderHeadModel, LogisticCumulativeLink
from retinopathy.models.heads import RMSPool2d, EncoderHeadModel, GlobalAvgPool2d, GlobalAvgPool2dHead, \
    GlobalMaxAvgPool2dHead, \
    GlobalMaxPool2dHead, ObjectContextPoolHead, FourReluBlock, Flatten, CLSBlock, RMSPoolHead
from retinopathy.models.inceptionv4 import InceptionV4Encoder, inceptionv4, InceptionV4, Inception_B, Inception_A, \
    Inception_C, Reduction_A, Mixed_5a, Mixed_3a, Mixed_4a, BasicConv2d, Reduction_B
from retinopathy.models.oc import ASP_OC_Module, BaseOC_Context_Module, SelfAttentionBlock2D, _SelfAttentionBlock
from retinopathy.models.regression import regression_to_class


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
        'import torch.utils.model_zoo as model_zoo'
    ]
    functions = [
        tensor_from_rgb_image,
        id_from_fname,
        read_rgb_image,
        get_class_names,
        RetinopathyDataset,
        RMSPool2d,
        GlobalAvgPool2d,
        GlobalAvgPool2dHead,
        GlobalMaxPool2dHead,
        GlobalMaxAvgPool2dHead,
        ObjectContextPoolHead,
        DenseNet121Encoder,
        DenseNet169Encoder,
        DenseNet201Encoder,
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
        _SelfAttentionBlock,
        SelfAttentionBlock2D,
        BaseOC_Context_Module,
        OrdinalEncoderHeadModel,
        LogisticCumulativeLink,
        RMSPoolHead,
        FourReluBlock,
        CLSBlock,
        ASP_OC_Module,
        Flatten,
        EncoderHeadModel,
        crop_black,
        CropBlackRegions,
        clahe_preprocessing,
        ChannelIndependentCLAHE,
        get_model,
        get_test_transform,
        PickModelOutput,
        run_model_inference_via_dataset,
        run_model_inference,
        average_predictions,
        cls_predictions_to_submission,
        reg_predictions_to_submission,
        regression_to_class,
    ]

    lines.append('# Imports\n')
    for import_statement in imports:
        lines.append(import_statement + '\n')
    lines.append('\n')
    lines.append(f'UNLABELED_CLASS = {UNLABELED_CLASS}\n')
    lines.append('pretrained_settings = None\n')
    lines.append('# Functions\n')
    for function in functions:
        source = inspect.getsource(function)
        lines.extend(source)
        lines.append('\n')
        lines.append('\n')

    with open('kernel_bootstrap.py', 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
