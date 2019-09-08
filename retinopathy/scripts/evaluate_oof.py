import numpy as np
from pytorch_toolbelt.utils import fs
from sklearn.metrics import cohen_kappa_score

from retinopathy.dataset import get_datasets
from retinopathy.inference import run_model_inference_via_dataset, cls_predictions_to_submission, \
    reg_predictions_to_submission, average_predictions


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


def evaluate_averaging(checkpoints, use_aptos2015, use_aptos2019, use_messidor, use_idrid):
    for tta in [None]:
        predictions = []
        scores = []

        train_ds, valid_ds, _ = get_datasets(use_aptos2015=use_aptos2015,
                                             use_aptos2019=use_aptos2019,
                                             use_messidor=use_messidor,
                                             use_idrid=use_idrid)
        full_ds = train_ds + valid_ds
        targets = np.concatenate((train_ds.targets, valid_ds.targets))
        assert len(targets) == len(full_ds)
        for fold in range(4):
            checkpoint_file = fs.auto_file(checkpoints[fold])
            p = run_model_inference_via_dataset(model_checkpoint=checkpoint_file,
                                                dataset=full_ds,
                                                tta=tta,
                                                batch_size=16,
                                                apply_softmax=False,
                                                workers=6)

            oof_diagnosis = reg_predictions_to_submission(p)['diagnosis'].values
            oof_score = cohen_kappa_score(oof_diagnosis, targets, weights='quadratic')
            scores.append(oof_score)
            predictions.append(p)

        mean_predictions = average_predictions(predictions, 'mean')
        mean_score = cohen_kappa_score(reg_predictions_to_submission(mean_predictions)['diagnosis'].values,
                                       targets, weights='quadratic')
        # geom_predictions = average_predictions(predictions, 'geom')
        # geom_score = cohen_kappa_score(reg_predictions_to_submission(geom_predictions)['diagnosis'].values,
        #                                targets, weights='quadratic')

        name = ''
        if use_aptos2015: name += 'aptos2015'
        if use_aptos2019: name += 'aptos2019'
        if use_messidor: name += 'messidor'
        if use_idrid: name += 'idrid'
        print(name, 'TTA', tta, 'Mean', np.mean(scores), 'std', np.std(scores), 'MeanAvg', mean_score)


def main():
    checkpoints = [
        'pretrained/reg_resnext50_rms_fold0_best.pth',
        'pretrained/reg_resnext50_rms_fold1_best.pth',
        'pretrained/reg_resnext50_rms_fold2_best.pth',
        'pretrained/reg_resnext50_rms_fold3_best.pth'
    ]

    evaluate_averaging(checkpoints, use_aptos2015=False, use_aptos2019=False, use_idrid=False, use_messidor=True)
    evaluate_averaging(checkpoints, use_aptos2015=False, use_aptos2019=False, use_idrid=True, use_messidor=False)
    evaluate_averaging(checkpoints, use_aptos2015=False, use_aptos2019=True, use_idrid=False, use_messidor=False)
    evaluate_averaging(checkpoints, use_aptos2015=True, use_aptos2019=False, use_idrid=False, use_messidor=False)


if __name__ == '__main__':
    main()
