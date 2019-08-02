import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import to_numpy
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from retinopathy.dataset import get_datasets, get_class_names
from retinopathy.factory import get_model
from retinopathy.inference import run_model_inference_via_dataset, \
    reg_predictions_to_submission


def plot_confusion_matrix(cm, class_names,
                          figsize=(16, 16),
                          normalize=False,
                          title='Confusion matrix',
                          fname=None,
                          noshow=False):
    """Render the confusion matrix and return matplotlib's figure with it.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap = plt.cm.Oranges

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    f = plt.figure(figsize=figsize)
    plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    # f.tick_params(direction='inout')
    # f.set_xticklabels(varLabels, rotation=45, ha='right')
    # f.set_yticklabels(varLabels, rotation=45, va='top')

    plt.yticks(tick_marks, class_names)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if fname is not None:
        plt.savefig(fname=fname)

    if not noshow:
        plt.show()

    return f


def evaluate_generalization(checkpoints, fold=None, num_folds=4):
    num_datasets = len(checkpoints)
    kappa_matrix = np.zeros((num_datasets, num_datasets), dtype=np.float32)
    class_names = list(checkpoints.keys())
    checkpoint_files = list(checkpoints[name] for name in class_names)

    for i, dataset_name in enumerate(class_names):
        _, valid_ds, _ = get_datasets(use_aptos2015=dataset_name == 'aptos2015',
                                      use_aptos2019=dataset_name == 'aptos2019',
                                      use_messidor=dataset_name == 'messidor',
                                      use_idrid=dataset_name == 'idrid',
                                      fold=fold,
                                      folds=num_folds)

        for j, checkpoint_file in enumerate(checkpoint_files):
            print('Evaluating', dataset_name, 'on', checkpoint_file)
            p = run_model_inference_via_dataset(model_checkpoint=checkpoint_file,
                                                dataset=valid_ds,
                                                batch_size=32 * 3,
                                                apply_softmax=False)

            diagnosis = reg_predictions_to_submission(p)['diagnosis'].values
            score = cohen_kappa_score(diagnosis, valid_ds.targets, weights='quadratic')
            kappa_matrix[i, j] = score

    print(kappa_matrix)
    np.save('kappa_matrix', kappa_matrix)
    # kappa_matrix = np.array([
    #     [0.6204755, 0.508746, 0.47336853, 0.5163422],
    #     [0.80155796, 0.92287904, 0.7245792, 0.80202734],
    #     [0.7953327, 0.77868223, 0.8940796, 0.7031926],
    #     [0.7898711, 0.6854141, 0.6820601, 0.92435944]])
    plot_confusion_matrix(kappa_matrix, normalize=False, fname='kappa_matrix.png', class_names=class_names)


@torch.no_grad()
def evaluate_generalization(checkpoints, num_folds=4):
    num_datasets = len(checkpoints)
    # kappa_matrix = np.zeros((num_datasets, num_datasets), dtype=np.float32)
    class_names = list(checkpoints.keys())

    # results = {}

    for dataset_trained_on, checkpoints_per_fold in checkpoints.items():
        # For each dataset trained on

        for fold_trained_on, checkpoint_file in enumerate(checkpoints_per_fold):
            # For each checkpoint
            if checkpoint_file is None:
                continue

            # Load model
            checkpoint = torch.load(checkpoint_file)
            model_name = checkpoint['checkpoint_data']['cmd_args']['model']
            batch_size = 16  # checkpoint['checkpoint_data']['cmd_args']['batch_size']
            num_classes = len(get_class_names())
            model = get_model(model_name, pretrained=False, num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.eval().cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            for dataset_index, dataset_validate_on in enumerate(class_names):
                # For each available dataset

                for fold_validate_on in range(num_folds):
                    _, valid_ds, _ = get_datasets(use_aptos2015=dataset_validate_on == 'aptos2015',
                                                  use_aptos2019=dataset_validate_on == 'aptos2019',
                                                  use_messidor=dataset_validate_on == 'messidor',
                                                  use_idrid=dataset_validate_on == 'idrid',
                                                  fold=fold_validate_on,
                                                  folds=num_folds)

                    data_loader = DataLoader(valid_ds, batch_size * torch.cuda.device_count(),
                                             pin_memory=True,
                                             num_workers=8)

                    predictions = defaultdict(list)
                    for batch in tqdm(data_loader,
                                      desc=f'Evaluating {dataset_validate_on} fold {fold_validate_on} on {checkpoint_file}'):
                        input = batch['image'].cuda(non_blocking=True)
                        outputs = model(input)
                        logits = to_numpy(outputs['logits'].softmax(dim=1))
                        regression = to_numpy(outputs['regression'])
                        features = to_numpy(outputs['features'])

                        predictions['image_id'].extend(batch['image_id'])
                        predictions['diagnosis_true'].extend(to_numpy(batch['targets']))
                        predictions['logits'].extend(logits)
                        predictions['regression'].extend(regression)
                        predictions['features'].extend(features)

                    pickle_name = id_from_fname(
                        checkpoint_file) + f'_on_{dataset_validate_on}_fold{fold_validate_on}.pkl'

                    df = pd.DataFrame.from_dict(predictions)
                    df.to_pickle(pickle_name)

                # p = run_model_inference_via_dataset(model_checkpoint=checkpoint_file,
                #                                     dataset=valid_ds,
                #                                     batch_size=32 * 3,
                #                                     apply_softmax=False)

                # diagnosis = reg_predictions_to_submission(p)['diagnosis'].values
                # score = cohen_kappa_score(diagnosis, valid_ds.targets, weights='quadratic')
                # kappa_matrix[dataset_index, j] = score

    # print(kappa_matrix)
    # np.save('kappa_matrix', kappa_matrix)
    # kappa_matrix = np.array([
    #     [0.6204755, 0.508746, 0.47336853, 0.5163422],
    #     [0.80155796, 0.92287904, 0.7245792, 0.80202734],
    #     [0.7953327, 0.77868223, 0.8940796, 0.7031926],
    #     [0.7898711, 0.6854141, 0.6820601, 0.92435944]])
    # plot_confusion_matrix(kappa_matrix, normalize=False, fname='kappa_matrix.png', class_names=class_names)


if __name__ == '__main__':
    checkpoints = {
        # 'aptos2015': 'runs/regression/reg_resnext50_rms/Jul20_14_59/reg_resnext50_rms_romantic_roentgen_fold0_aptos2015/checkpoints/reg_resnext50_rms_romantic_roentgen_fold0_aptos2015_best.pth',
        'aptos2019': [
            'pretrained/reg_seresnext50_rms_512_medium_mse_aptos2019_fold0_awesome_babbage.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_aptos2019_fold1_hopeful_khorana.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_aptos2019_fold2_trusting_nightingale.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_aptos2019_fold3_epic_wing.pth'
        ],
        'idrid': [
            'pretrained/reg_seresnext50_rms_512_medium_mse_idrid_fold0_heuristic_ptolemy.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_idrid_fold1_gifted_visvesvaraya.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_idrid_fold2_sharp_brattain.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_idrid_fold3_vibrant_minsky.pth'
        ],
        'messidor': [
            'pretrained/reg_seresnext50_rms_512_medium_mse_messidor_fold0_admiring_sinoussi.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_messidor_fold1_jovial_snyder.pth',
            'pretrained/reg_seresnext50_rms_512_medium_mse_messidor_fold2_agitated_agnesi.pth',
            None
        ]
    }
    evaluate_generalization(checkpoints, num_folds=4)
