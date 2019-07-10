import multiprocessing
import os
from functools import partial

import torch

from pytorch_toolbelt.utils import fs
import pandas as pd
import numpy as np
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.inference.tta import *
from sklearn.metrics import cohen_kappa_score
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from retinopathy.lib.dataset import get_class_names, RetinopathyDataset
from retinopathy.lib.factory import get_model, get_test_aug
from retinopathy.lib.models.regression import regression_to_class


class PickModelOutput(nn.Module):
    def __init__(self, target_key='logits'):
        super().__init__()
        self.target_key = target_key

    def forward(self, input):
        return input[self.target_key]


def run_model_inference(model_checkpoint: str,
                        test_csv: pd.DataFrame,
                        data_dir,
                        model_name=None,
                        batch_size=None,
                        image_size=(512, 512),
                        images_dir='test_images',
                        tta=None,
                        apply_softmax=True) -> pd.DataFrame:
    checkpoint = torch.load(model_checkpoint)
    if model_name is None:
        model_name = checkpoint['checkpoint_data']['cmd_args']['model']

    if batch_size is None:
        batch_size = checkpoint['checkpoint_data']['cmd_args']['batch_size']

    num_classes = len(get_class_names())
    if str.startswith(model_name, 'reg_'):
        num_classes = 1

    model = get_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = nn.Sequential(model, PickModelOutput('logits'))

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

        test_csv['image_fname'] = test_csv['id_code'].apply(lambda x: os.path.join(data_dir, images_dir, f'{x}.png'))
        test_ds = RetinopathyDataset(test_csv['image_fname'], None, get_test_aug(image_size))
        data_loader = DataLoader(test_ds, batch_size,
                                 pin_memory=True,
                                 num_workers=multiprocessing.cpu_count())

        test_ids = []
        test_preds = []

        for batch in tqdm(data_loader):
            input = batch['image'].cuda(non_blocking=True)
            outputs = model(input)
            predictions = to_numpy(outputs)

            test_ids.extend(batch['image_id'])
            test_preds.extend(predictions)

        predictions = pd.DataFrame.from_dict({'id_code': test_ids, 'diagnosis': test_preds})

    del model, data_loader
    return predictions


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


def average_predictions(predictions):
    accumulator = np.zeros_like(predictions[0]['diagnosis'].values)
    for p in predictions:
        accumulator += p['diagnosis']
    accumulator /= len(predictions)
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
                                  batch_size=4,
                                  tta='d4')
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
