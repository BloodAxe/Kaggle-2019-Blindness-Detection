import base64
import os
import inspect

from pytorch_toolbelt.utils.fs import read_rgb_image, id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

from retinopathy.lib.augmentations import CropBlackRegions, get_test_aug, crop_black, ChannelIndependentCLAHE, \
    clahe_preprocessing
from retinopathy.lib.dataset import RetinopathyDataset, get_class_names
from retinopathy.lib.factory import get_model
from retinopathy.lib.inference import PickModelOutput, run_model_inference, cls_predictions_to_submission, \
    average_predictions, reg_predictions_to_submission, run_model_inference_via_dataset
from retinopathy.lib.models.heads import RMSPool2d, EncoderHeadModel, GlobalAvgPool2d, GlobalAvgPool2dHead, \
    GlobalWeightedAvgPool2dHead, GlobalWeightedMaxPool2dHead, GlobalMaxAvgPool2dHead, RMSPoolRegressionHead, \
    GlobalMaxPool2dHead, ObjectContextPoolHead, FourReluBlock, Flatten, HyperPoolHead, CLSBlock
from retinopathy.lib.models.oc import ASP_OC_Module, BaseOC_Context_Module, SelfAttentionBlock2D, _SelfAttentionBlock
from retinopathy.lib.models.ordinal import OrdinalEncoderHeadModel, LogisticCumulativeLink
from retinopathy.lib.models.regression import regression_to_class


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
    for file in ['deps/pytorch_toolbelt-0.1.2.tar.gz']:
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
        'from torch.autograd import Variable'
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
        GlobalWeightedAvgPool2dHead,
        GlobalWeightedMaxPool2dHead,
        GlobalMaxPool2dHead,
        RMSPoolRegressionHead,
        GlobalMaxAvgPool2dHead,
        ObjectContextPoolHead,
        _SelfAttentionBlock,
        SelfAttentionBlock2D,
        BaseOC_Context_Module,
        OrdinalEncoderHeadModel,
        LogisticCumulativeLink,
        FourReluBlock,
        CLSBlock,
        HyperPoolHead,
        ASP_OC_Module,
        Flatten,
        EncoderHeadModel,
        crop_black,
        CropBlackRegions,
        clahe_preprocessing,
        ChannelIndependentCLAHE,
        get_model,
        get_test_aug,
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
