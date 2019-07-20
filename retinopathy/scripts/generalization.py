from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.visualization import plot_confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np
from retinopathy.lib.dataset import get_datasets
from retinopathy.lib.inference import run_model_inference, run_model_inference_via_dataset, reg_predictions_to_submission


def evaluate_generalization(checkpoints, fold=None, num_folds=4):
    num_datasets = len(checkpoints)
    kappa_matrix = np.zeros((num_datasets, num_datasets), dtype=np.float32)
    class_names = list(checkpoints.keys())
    checkpoint_files = list(checkpoints[name] for name in class_names)

    for i, dataset_name in enumerate(class_names):
        _, valid_ds = get_datasets(use_aptos2015=dataset_name == 'aptos2015',
                                   use_aptos2019=dataset_name == 'aptos2019',
                                   use_messidor=dataset_name == 'messidor',
                                   use_idrid=dataset_name == 'idrid',
                                   fold=fold,
                                   folds=num_folds)

        for j, checkpoint_file in enumerate(checkpoint_files):
            p = run_model_inference_via_dataset(model_checkpoint=checkpoint_file,
                                                dataset=valid_ds,
                                                batch_size=16,
                                                apply_softmax=False,
                                                workers=6)

            diagnosis = reg_predictions_to_submission(p)['diagnosis'].values
            score = cohen_kappa_score(diagnosis, valid_ds.targets, weights='quadratic')
            kappa_matrix[i, j] = score

    print(kappa_matrix)
    plot_confusion_matrix(kappa_matrix, normalize=False, fname='kappa_matrix.png', class_names=list(checkpoints.keys))


if __name__ == '__main__':
    checkpoints = {
        'aptos2015': fs.auto_file(''),
        'aptos2019': fs.auto_file(''),
        'idrid': fs.auto_file(''),
        'messidor': fs.auto_file(''),
    }
    evaluate_generalization(checkpoints, fold=0, num_folds=4)
