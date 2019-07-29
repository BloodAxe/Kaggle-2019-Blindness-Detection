import itertools

import numpy as np
from sklearn.metrics import cohen_kappa_score

from retinopathy.dataset import get_datasets
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



if __name__ == '__main__':
    checkpoints = {
        'aptos2015': 'runs/regression/reg_resnext50_rms/Jul20_14_59/reg_resnext50_rms_romantic_roentgen_fold0_aptos2015/checkpoints/reg_resnext50_rms_romantic_roentgen_fold0_aptos2015_best.pth',
        'aptos2019': 'runs/regression/reg_resnext50_rms/Jul20_12_47/reg_resnext50_rms_distracted_elion_fold0_aptos2019/checkpoints/best.pth',
        'idrid': 'runs/regression/reg_resnext50_rms/Jul20_00_17/reg_resnext50_rms_laughing_poitras_fold0_idrid/checkpoints/reg_resnext50_rms_laughing_poitras_fold0_idrid_best.pth',
        'messidor': 'runs/regression/reg_resnext50_rms/Jul19_22_26/reg_resnext50_rms_elated_khorana_fold0_messidor/checkpoints/reg_resnext50_rms_elated_khorana_fold0_messidor_best.pth',
    }
    evaluate_generalization(checkpoints, fold=0, num_folds=4)
