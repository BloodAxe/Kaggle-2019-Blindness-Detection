import multiprocessing
import os
from functools import partial

import torch

from pytorch_toolbelt.utils import fs
import pandas as pd
import numpy as np
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.inference.tta import *
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from retinopathy.lib.dataset import get_class_names, RetinopathyDataset
from retinopathy.lib.factory import get_model, get_test_aug


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
                        tta=None,
                        apply_softmax=True) -> pd.DataFrame:
    checkpoint = torch.load(model_checkpoint)
    if model_name is None:
        model_name = checkpoint['checkpoint_data']['cmd_args']['model']

    if batch_size is None:
        batch_size = checkpoint['checkpoint_data']['cmd_args']['batch_size']

    model = get_model(model_name, pretrained=False, num_classes=len(get_class_names()))
    model.load_state_dict(checkpoint['model_state_dict'])

    model = nn.Sequential(model, PickModelOutput('logits'))

    if apply_softmax:
        model = nn.Sequential(model, nn.Softmax(dim=1))

    if tta == '10crop':
        model = TTAWrapper(model, tencrop_image2label, crop_size=(384, 384))

    if tta == 'flip':
        model = TTAWrapper(model, fliplr_image2label)

    with torch.no_grad():
        model = model.eval().cuda()

        test_csv['image_fname'] = test_csv['id_code'].apply(lambda x: os.path.join(data_dir, 'test_images', f'{x}.png'))
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


def predictions_to_submission(predictions) -> pd.DataFrame:
    predictions['diagnosis'] = predictions['diagnosis'].apply(lambda x: np.argmax(x))
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
    test_csv = pd.read_csv('data/test.csv')
    test_csv = test_csv[:100]
    checkpoints = [
        'runs/classification/cls_resnet18/fold_0/Jul08_14_51_ce/checkpoints/fold0_best.pth',
        'runs/classification/cls_resnet18/fold_1/Jul08_16_13_ce/checkpoints/fold1_best.pth',
        'runs/classification/cls_resnet18/fold_2/Jul09_00_19_ce/checkpoints/fold2_best.pth',
        'runs/classification/cls_resnet18/fold_3/Jul09_01_29_ce/checkpoints/fold3_best.pth'
    ]

    predictions = []
    for checkpoint in checkpoints:
        df0 = run_model_inference(model_checkpoint=fs.auto_file(checkpoint),
                                  test_csv=test_csv,
                                  data_dir='data',
                                  batch_size=4,
                                  tta='flip')
        print(df0.head())
        predictions.append(df0)

    pred = average_predictions(predictions)
    print(pred.head())

    submit = predictions_to_submission(pred)
    print(submit.head())


if __name__ == '__main__':
    main()
