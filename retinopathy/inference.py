import multiprocessing
import os
from collections import defaultdict
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import pytorch_toolbelt.inference.functional as FF
import torch
from pytorch_toolbelt.utils.torch_utils import to_numpy
from scipy.stats import trim_mean
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from retinopathy.augmentations import get_test_transform
from retinopathy.dataset import get_class_names, RetinopathyDataset
from retinopathy.factory import get_model
from retinopathy.models.regression import regression_to_class
from retinopathy.train_utils import report_checkpoint
import torch.nn.functional as F


class FlipLRMultiheadTTA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        output = self.model(image)

        # Flip image input
        output2 = self.model(FF.torch_fliplr(image))

        if len(output['features'].size()) == 4:
            output2['features'] = FF.torch_fliplr(output2['features'])

        output['logits'] = (output['logits'] + output2['logits']) * 0.5
        output['ordinal'] = (output['ordinal'] + output2['ordinal']) * 0.5
        output['regression'] = (output['regression'] + output2['regression']) * 0.5
        output['features'] = (output['features'] + output2['features']) * 0.5
        return output


class MultiscaleFlipLRMultiheadTTA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        rows = image.size(2)
        cols = image.size(3)
        outputs = []
        for scale in [1.0, 1.15, 0.87]:
            image_i = F.interpolate(image, size=(int(rows * scale), int(cols * scale)), mode='bilinear', align_corners=True)

            output = self.model(image_i)
            outputs.append(output)

            # Flip image input
            output2 = self.model(FF.torch_fliplr(image_i))
            outputs.append(output2)

        for key in {'logits', 'features', 'regression', 'ordinal'}:
            for i in range(1, len(outputs)):
                outputs[0][key] += outputs[i][key]
            outputs[0][key] /= len(outputs)

        return outputs[0]


class Flip4MultiheadTTA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        outputs = []
        outputs.append(self.model(image))

        image_fliplr = FF.torch_fliplr(image)
        outputs.append(self.model(image_fliplr))

        image_flipud = FF.torch_flipud(image)
        outputs.append(self.model(image_flipud))

        image_fliplr_ud = FF.torch_fliplr(image_flipud)
        outputs.append(self.model(FF.torch_fliplr(image_fliplr_ud)))

        for key in {'logits', 'features', 'regression', 'ordinal'}:
            for i in range(1, len(outputs)):
                outputs[0][key] += outputs[i][key]

            outputs[0][key] /= len(outputs)

        return outputs[0]


class ApplySoftmaxToLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input['logits'] = input['logits'].softmax(dim=1)
        return input


def compute_cdf(targets):
    hist = np.bincount(targets)
    overall_cdf_valid = np.cumsum(hist) / float(sum(hist))
    return overall_cdf_valid


def regression_getScore(pred, cdf, valid=False):
    num = pred.shape[0]
    output = np.asarray([5] * num, dtype=int)
    rank = pred.argsort()
    output[rank[:int(num * cdf[0] - 1)]] = 1
    output[rank[int(num * cdf[0]):int(num * cdf[1] - 1)]] = 2
    output[rank[int(num * cdf[1]):int(num * cdf[2] - 1)]] = 3
    cutoff = [pred[rank[int(num * cdf[i] - 1)]] for i in range(4)]
    if valid:
        return output, cutoff
    return output


def run_model_inference_via_dataset(model_checkpoint: str,
                                    dataset: RetinopathyDataset,
                                    model_name=None,
                                    batch_size=None,
                                    tta=None,
                                    need_features=True,
                                    apply_softmax=True,
                                    workers=None) -> pd.DataFrame:
    if workers is None:
        workers = multiprocessing.cpu_count()

    checkpoint = torch.load(model_checkpoint)
    report_checkpoint(checkpoint)

    if model_name is None:
        model_name = checkpoint['checkpoint_data']['cmd_args']['model']

    if batch_size is None:
        batch_size = checkpoint['checkpoint_data']['cmd_args'].get('batch_size', 1)

    coarse_grading = checkpoint['checkpoint_data']['cmd_args'].get('coarse', False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_classes = len(get_class_names(coarse_grading=coarse_grading))
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    if apply_softmax:
        model = nn.Sequential(model, ApplySoftmaxToLogits())

    if tta == 'flip' or tta == 'fliplr':
        model = FlipLRMultiheadTTA(model)

    if tta == 'flip4':
        model = Flip4MultiheadTTA(model)

    if tta == 'fliplr_ms':
        model = MultiscaleFlipLRMultiheadTTA(model)

    with torch.no_grad():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.eval()

        data_loader = DataLoader(dataset, batch_size,
                                 pin_memory=True,
                                 num_workers=workers)

        predictions = defaultdict(list)

        for batch in tqdm(data_loader):
            input = batch['image'].cuda(non_blocking=True)
            outputs = model(input)

            predictions['image_id'].extend(batch['image_id'])
            if 'targets' in batch:
                predictions['diagnosis'].extend(to_numpy(batch['targets']).tolist())

            predictions['logits'].extend(to_numpy(outputs['logits']).tolist())
            predictions['regression'].extend(to_numpy(outputs['regression']).tolist())
            predictions['ordinal'].extend(to_numpy(outputs['ordinal']).tolist())
            if need_features:
                predictions['features'].extend(to_numpy(outputs['features']).tolist())

        predictions = pd.DataFrame.from_dict(predictions)

    del data_loader, model
    return predictions


@torch.no_grad()
def run_models_inference_via_dataset(model_checkpoints: List[str],
                                     dataset: RetinopathyDataset,
                                     batch_size=1,
                                     coarse_grading=False,
                                     tta=None,
                                     need_features=True,
                                     apply_softmax=True,
                                     workers=None) -> List[pd.DataFrame]:
    if workers is None:
        workers = multiprocessing.cpu_count()

    models = []
    models_predictions = []

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load models
    for model_checkpoint in model_checkpoints:
        checkpoint = torch.load(model_checkpoint)

        model_name = checkpoint['checkpoint_data']['cmd_args']['model']

        print(model_checkpoint, model_name)
        report_checkpoint(checkpoint)

        num_classes = len(get_class_names(coarse_grading=coarse_grading))
        model = get_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint

        if apply_softmax:
            model = nn.Sequential(model, ApplySoftmaxToLogits())

        if tta == 'flip' or tta == 'fliplr':
            model = FlipLRMultiheadTTA(model)

        if tta == 'flip4':
            model = Flip4MultiheadTTA(model)

        if tta == 'fliplr_ms':
            model = MultiscaleFlipLRMultiheadTTA(model)

        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.eval()

        models.append(model)
        models_predictions.append(defaultdict(list))

    data_loader = DataLoader(dataset, batch_size,
                             pin_memory=True,
                             num_workers=workers)

    for batch in tqdm(data_loader):
        input = batch['image'].cuda(non_blocking=True)

        for model, predictions in zip(models, models_predictions):
            outputs = model(input)

            predictions['image_id'].extend(batch['image_id'])
            if 'targets' in batch:
                predictions['diagnosis'].extend(to_numpy(batch['targets']).tolist())

            predictions['logits'].extend(to_numpy(outputs['logits']).tolist())
            predictions['regression'].extend(to_numpy(outputs['regression']).tolist())
            predictions['ordinal'].extend(to_numpy(outputs['ordinal']).tolist())
            if need_features:
                predictions['features'].extend(to_numpy(outputs['features']).tolist())

    models_predictions = [pd.DataFrame.from_dict(p) for p in models_predictions]

    del data_loader, models
    return models_predictions


def image_with_name_in_dir(dirname, image_id):
    for ext in ['png', 'jpg', 'jpeg', 'tif']:
        image_fname = os.path.join(dirname, f'{image_id}.{ext}')
        if os.path.isfile(image_fname):
            return image_fname
    raise FileNotFoundError(image_fname)


def run_model_inference(model_checkpoint: str,
                        test_csv: pd.DataFrame,
                        data_dir,
                        images_dir='test_images',
                        preprocessing=None,
                        image_size=None,
                        crop_black=True,
                        **kwargs) -> pd.DataFrame:
    checkpoint = torch.load(model_checkpoint)
    if preprocessing is None:
        preprocessing = checkpoint['checkpoint_data']['cmd_args'].get('preprocessing', None)

    if image_size is None:
        image_size = checkpoint['checkpoint_data']['cmd_args'].get('image_size', 512)
        image_size = (image_size, image_size)

    image_fnames = test_csv['id_code'].apply(lambda x: image_with_name_in_dir(os.path.join(data_dir, images_dir), x))

    if 'diagnosis' in test_csv:
        targets = test_csv['diagnosis'].values
    else:
        targets = None

    test_ds = RetinopathyDataset(image_fnames, targets, get_test_transform(image_size,
                                                                           preprocessing=preprocessing,
                                                                           crop_black=crop_black))
    return run_model_inference_via_dataset(model_checkpoint, test_ds, **kwargs)


def run_models_inference(model_checkpoints: List[str],
                         test_csv: pd.DataFrame,
                         data_dir,
                         images_dir='test_images',
                         preprocessing=None,
                         image_size=None,
                         crop_black=True,
                         **kwargs) -> List[pd.DataFrame]:
    checkpoint = torch.load(model_checkpoints[0])
    if preprocessing is None:
        preprocessing = checkpoint['checkpoint_data']['cmd_args'].get('preprocessing', None)

    if image_size is None:
        image_size = checkpoint['checkpoint_data']['cmd_args'].get('image_size', 512)
        image_size = (image_size, image_size)

    image_fnames = test_csv['id_code'].apply(lambda x: image_with_name_in_dir(os.path.join(data_dir, images_dir), x))

    if 'diagnosis' in test_csv:
        targets = test_csv['diagnosis'].values
    else:
        targets = None

    test_ds = RetinopathyDataset(image_fnames, targets, get_test_transform(image_size,
                                                                           preprocessing=preprocessing,
                                                                           crop_black=crop_black))
    return run_models_inference_via_dataset(model_checkpoints, test_ds, **kwargs)


def reg_cdf_predictions_to_submission(predictions, cdf) -> pd.DataFrame:
    predictions = predictions.copy()
    x = torch.from_numpy(predictions['diagnosis'].values)
    x = regression_getScore(x, cdf)
    predictions['diagnosis'] = to_numpy(x)
    return predictions


def average_predictions(predictions: List[pd.DataFrame], column: str,
                        method='mean', min=None, max=None) -> pd.DataFrame:
    preds = []
    for p in predictions:
        pred = to_numpy(p[column].values.tolist())
        preds.append(pred)

    preds = np.row_stack(preds)
    if min is not None or max is not None:
        preds = np.clip(preds, min, max)

    if method == 'mean':
        y_pred = np.mean(preds, axis=0)
    elif method == 'trim_mean':
        y_pred = trim_mean(preds, proportiontocut=0.1, axis=0)
    elif method == 'median':
        y_pred = np.median(preds, axis=0)
    else:
        raise KeyError(method)

    result = pd.DataFrame.from_dict({'id_code': predictions[0]['image_id'].values,
                                     'diagnosis': y_pred.tolist()})
    return result


def cls_predictions_to_submission(predictions) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions['diagnosis'] = predictions['diagnosis'].apply(lambda x: np.argmax(x))
    return predictions


def reg_predictions_to_submission(predictions, rounding_coefficients=None) -> pd.DataFrame:
    rounder = partial(regression_to_class, rounding_coefficients=rounding_coefficients)
    predictions = predictions.copy()
    predictions['diagnosis'] = rounder(predictions['diagnosis'].values)
    predictions['diagnosis'] = predictions['diagnosis'].apply(int)
    return predictions
