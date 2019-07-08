import base64
import os
import inspect

from pytorch_toolbelt.utils.fs import read_rgb_image, id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image

from retinopathy.lib.dataset import RetinopathyDataset, get_class_names
from retinopathy.lib.factory import get_model, get_test_aug
from retinopathy.lib.inference import PickModelOutput, run_model_inference, predictions_to_submission
from retinopathy.lib.models.classification import BaselineClassificationModel, ClassifierModule


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
        'import cv2',
        'import torch',
        'import pandas as pd',
        'import numpy as np',
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
    ]
    functions = [
        tensor_from_rgb_image,
        id_from_fname,
        read_rgb_image,
        get_class_names,
        RetinopathyDataset,
        BaselineClassificationModel,
        ClassifierModule,
        get_model,
        get_test_aug,
        PickModelOutput,
        run_model_inference,
        predictions_to_submission
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
