import numpy as np
from pytorch_toolbelt.utils import fs
from sklearn.metrics import cohen_kappa_score

from retinopathy.lib.dataset import get_datasets
from retinopathy.lib.inference import run_model_inference_via_dataset, cls_predictions_to_submission


def evaluate(checkpoints, fast=False):
    num_folds = len(checkpoints)

    for tta in [None, 'flip', 'd4']:
        oof_predictions = []
        oof_scores = []

        for fold in range(4):
            _, valid_ds = get_datasets(use_aptos2015=not fast,
                                       use_aptos2019=True,
                                       use_messidor=not fast,
                                       use_idrid=not fast,
                                       fold=fold,
                                       folds=num_folds)

            checkpoint_file = fs.auto_file(checkpoints[fold])
            p = run_model_inference_via_dataset(model_checkpoint=checkpoint_file,
                                                dataset=valid_ds,
                                                tta=tta,
                                                batch_size=16,
                                                apply_softmax=True,
                                                workers=6)

            oof_diagnosis = cls_predictions_to_submission(p)['diagnosis'].values
            oof_score = cohen_kappa_score(oof_diagnosis, valid_ds.targets, weights='quadratic')
            oof_scores.append(oof_score)
            oof_predictions.append(p)

        # averaged_predictions = average_predictions(oof_predictions)
        print(tta, np.mean(oof_scores), np.std(oof_scores))


def main():
    checkpoints = [
        'cls_resnext50_gap_extra_data_18_07_fold0_best.pth',
        'cls_resnext50_gap_extra_data_18_07_fold1_best.pth',
        'cls_resnext50_gap_extra_data_18_07_fold2_best.pth',
        'cls_resnext50_gap_extra_data_18_07_fold3_best.pth'
    ]

    evaluate(checkpoints)


if __name__ == '__main__':
    main()
