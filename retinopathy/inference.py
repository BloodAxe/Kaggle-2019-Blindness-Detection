import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.inference.tta import *
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import to_numpy
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from retinopathy.augmentations import get_test_transform
from retinopathy.dataset import get_class_names, RetinopathyDataset
from retinopathy.factory import get_model
from retinopathy.models.regression import regression_to_class


class PickModelOutput(nn.Module):
    def __init__(self, target_key='logits'):
        super().__init__()
        self.target_key = target_key

    def forward(self, input):
        return input[self.target_key]


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
                                    output_key='logits',
                                    model_name=None,
                                    batch_size=None,
                                    tta=None,
                                    apply_softmax=True,
                                    workers=None) -> pd.DataFrame:
    if workers is None:
        workers = multiprocessing.cpu_count()

    checkpoint = torch.load(model_checkpoint)
    if model_name is None:
        model_name = checkpoint['checkpoint_data']['cmd_args']['model']

    if batch_size is None:
        batch_size = checkpoint['checkpoint_data']['cmd_args']['batch_size']

    num_classes = len(get_class_names())
    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = nn.Sequential(model, PickModelOutput(output_key))

    if apply_softmax:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if tta == '10crop':
        model = TTAWrapper(model, tencrop_image2label, crop_size=(384, 384))

    if tta == 'd4':
        model = TTAWrapper(model, d4_image2label)

    if tta == 'flip':
        model = TTAWrapper(model, fliplr_image2label)

    with torch.no_grad():
        model = model.eval().cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        data_loader = DataLoader(dataset, batch_size,
                                 pin_memory=True,
                                 num_workers=workers)

        test_ids = []
        test_preds = []

        for batch in tqdm(data_loader):
            input = batch['image'].cuda(non_blocking=True)
            outputs = model(input)
            predictions = to_numpy(outputs)

            test_ids.extend(batch['image_id'])
            test_preds.extend(predictions)

        predictions = pd.DataFrame.from_dict({'id_code': test_ids, 'diagnosis': test_preds})

    del data_loader, model
    return predictions


def run_model_inference(model_checkpoint: str,
                        test_csv: pd.DataFrame,
                        data_dir,
                        output_key='logits',
                        model_name=None,
                        batch_size=None,
                        image_size=(512, 512),
                        images_dir='test_images',
                        tta=None,
                        apply_softmax=True,
                        workers=None) -> pd.DataFrame:
    image_fnames = test_csv['id_code'].apply(lambda x: os.path.join(data_dir, images_dir, f'{x}.png'))
    test_ds = RetinopathyDataset(image_fnames, None, get_test_transform(image_size))
    return run_model_inference_via_dataset(model_checkpoint, test_ds,
                                           model_name=model_name,
                                           output_key=output_key,
                                           batch_size=batch_size,
                                           tta=tta,
                                           apply_softmax=apply_softmax,
                                           workers=workers)


def cls_predictions_to_submission(predictions) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions['diagnosis'] = predictions['diagnosis'].apply(lambda x: np.argmax(x))
    return predictions


def reg_predictions_to_submission(predictions) -> pd.DataFrame:
    predictions = predictions.copy()
    x = torch.from_numpy(predictions['diagnosis'].values)
    x = regression_to_class(x)
    predictions['diagnosis'] = to_numpy(x)
    return predictions


def reg_cdf_predictions_to_submission(predictions, cdf) -> pd.DataFrame:
    predictions = predictions.copy()
    x = torch.from_numpy(predictions['diagnosis'].values)
    x = regression_getScore(x, cdf)
    predictions['diagnosis'] = to_numpy(x)
    return predictions


def average_predictions(predictions, method='mean', min=None, max=None):
    if method == 'mean':
        accumulator = np.zeros_like(predictions[0]['diagnosis'].values)
        for p in predictions:
            pred = p['diagnosis'].values
            if min is not None or max is not None:
                pred = np.clip(pred, min, max)
            accumulator += pred
        accumulator /= len(predictions)
    elif method == 'geom':
        accumulator = np.ones_like(predictions[0]['diagnosis'].values).astype(np.float32)
        for p in predictions:
            pred = p['diagnosis'].values
            if min is not None or max is not None:
                pred = np.clip(pred, min, max)
            accumulator *= pred
        accumulator = np.power(accumulator, 1. / len(predictions))
    else:
        raise KeyError(method)

    result = predictions[0].copy()
    result['diagnosis'] = accumulator
    return result


def main():
    train_csv = pd.read_csv('data/train.csv')
    # train_csv = train_csv[:100]

    checkpoints = [
        'runs/classification/cls_resnet18/fold_0/Jul08_14_51_ce/checkpoints/fold0_best.pth',
        'runs/classification/cls_resnet18/fold_1/Jul08_16_13_ce/checkpoints/fold1_best.pth',
        'runs/classification/cls_resnet18/fold_2/Jul09_00_19_ce/checkpoints/fold2_best.pth',
        'runs/classification/cls_resnet18/fold_3/Jul09_01_29_ce/checkpoints/fold3_best.pth'
    ]

    per_fold_scores = []
    predictions = []
    for checkpoint in checkpoints:
        df0 = run_model_inference(model_checkpoint=fs.auto_file(checkpoint),
                                  test_csv=train_csv,
                                  images_dir='train_images',
                                  data_dir='data',
                                  batch_size=8,
                                  tta=None)
        predictions.append(df0)

        fold_predictions = cls_predictions_to_submission(df0)

        score = cohen_kappa_score(fold_predictions['diagnosis'],
                                  train_csv['diagnosis'], weights='quadratic')

        per_fold_scores.append(score)
        print(df0.head())
        print(checkpoint, score)

    pred = average_predictions(predictions)
    print(pred.head())

    submit = cls_predictions_to_submission(pred)
    print(submit.head())

    print(per_fold_scores)
    score = cohen_kappa_score(submit['diagnosis'], train_csv['diagnosis'], weights='quadratic')
    print(score)


if __name__ == '__main__':
    main()
