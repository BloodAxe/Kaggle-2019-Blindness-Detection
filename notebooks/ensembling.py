import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
import pytest
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import to_numpy
from scipy.stats import trim_mean
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from retinopathy.models.common import regression_to_class
from retinopathy.rounder import OptimizedRounder, OptimizedRounderV2

MODELS = {
    'heuristic_sinoussi': [
        '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth',
        '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold1_heuristic_sinoussi.pth',
        '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold2_heuristic_sinoussi.pth',
        '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold3_heuristic_sinoussi.pth'
    ],
    'modest_williams': [
        '../models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold0_modest_williams.pth',
        '../models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold1_modest_williams.pth',
        '../models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold2_modest_williams.pth',
        '../models/Aug23_12_37_seresnext101_gap_modest_williams/seresnext101_gap_512_medium_aptos2019_messidor_idrid_fold3_modest_williams.pth'
    ],
    'happy_wright': [
        '../models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold0_happy_wright.pth',
        '../models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold1_happy_wright.pth',
        '../models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold2_happy_wright.pth',
        '../models/Aug31_00_05_inceptionv4_gap_happy_wright/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_fold3_happy_wright.pth'
    ],
    'epic_shaw': [
        '../models/Sep05_23_40_inceptionv4_gap_512_medium_pl1_epic_shaw/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_pl1_fold0_epic_shaw_warmup.pth',
        '../models/Sep05_23_40_inceptionv4_gap_512_medium_pl1_epic_shaw/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_pl1_fold1_epic_shaw_warmup.pth',
        '../models/Sep05_23_40_inceptionv4_gap_512_medium_pl1_epic_shaw/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_pl1_fold2_epic_shaw_warmup.pth',
        '../models/Sep05_23_40_inceptionv4_gap_512_medium_pl1_epic_shaw/inceptionv4_gap_512_medium_aptos2019_messidor_idrid_pl1_fold3_epic_shaw_warmup.pth',
    ],
    'admiring_minsky': [
        '../models/Sep07_01_31_seresnext50_gap_pl1_admiring_minsky/seresnext50_gap_512_medium_aptos2019_messidor_idrid_pl1_fold0_admiring_minsky_warmup.pth',
        '../models/Sep07_01_31_seresnext50_gap_pl1_admiring_minsky/seresnext50_gap_512_medium_aptos2019_messidor_idrid_pl1_fold1_admiring_minsky_warmup.pth',
        '../models/Sep07_01_31_seresnext50_gap_pl1_admiring_minsky/seresnext50_gap_512_medium_aptos2019_messidor_idrid_pl1_fold2_admiring_minsky_warmup.pth',
        '../models/Sep07_01_31_seresnext50_gap_pl1_admiring_minsky/seresnext50_gap_512_medium_aptos2019_messidor_idrid_pl1_fold3_admiring_minsky_warmup.pth',
    ]
}


def get_predictions(models: List[str], datasets: List[str]) -> List[str]:
    models_predictions = []
    for dataset in datasets:
        assert dataset in {'aptos2015_test_private',
                           'aptos2015_test_public',
                           'aptos2015_train',
                           'aptos2019_test',
                           'messidor2_train',
                           'idrid_test'}

        for model_name in models:
            if model_name in MODELS:
                model_checkpoints = MODELS[model_name]  # Well-known models
            else:
                model_checkpoints = [model_name]  # Random stuff

            for model_checkpoint in model_checkpoints:
                predictions = fs.change_extension(model_checkpoint, f'_{dataset}_predictions.pkl')
                models_predictions.append(predictions)
    return models_predictions


def test_optimize_kappa_on_idrid():
    average_predictions = None
    for index, predictions, in enumerate(idrid_predictions):
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.read_pickle(predictions)

        y_true = predictions['diagnosis'].values
        y_pred = predictions['ordinal'].values

        if average_predictions is None:
            average_predictions = y_pred.copy()
        else:
            average_predictions += y_pred

        print('Score on Idrid-test', index, cohen_kappa_score(y_true, regression_to_class(y_pred), weights='quadratic'))

    average_predictions /= len(idrid_predictions)

    rounder = OptimizedRounder()
    rounder.fit(average_predictions, y_true)
    print(rounder.coefficients())
    print('Score on Idrid-test',
          cohen_kappa_score(y_true, regression_to_class(average_predictions), weights='quadratic'),
          cohen_kappa_score(y_true,
                            regression_to_class(average_predictions, rounding_coefficients=rounder.coefficients()),
                            weights='quadratic'))


def test_optimize_kappa_on_aptos2015():
    pl1 = pd.read_csv('../data/aptos-2015/test_private_pseudolabel_round_1.csv')
    labeled_gt = dict((row['id_code'], row['diagnosis']) for i, row in pl1.iterrows())

    ids, train_x, train_y, train_y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=False,
                                                                    use_predictions=True)
    mask = np.zeros(len(ids), dtype=np.bool)
    for i, id in enumerate(ids):
        mask[i] = labeled_gt[id] >= 0

    ids = ids[mask]
    train_x = train_x[mask]
    train_y = train_y[mask]
    train_y_avg = train_y_avg[mask]

    ids, val_x, val_y, val_y_avg = prepare_inference_datasets(idrid_predictions, use_features=False,
                                                              use_predictions=True)
    rounder = OptimizedRounder()
    rounder.fit(train_x, train_y)
    print(rounder.coefficients())
    print('Score on APTOS',
          cohen_kappa_score(train_y, regression_to_class(train_x), weights='quadratic'),
          cohen_kappa_score(train_y, regression_to_class(train_x, rounding_coefficients=rounder.coefficients()),
                            weights='quadratic'))

    print('Score on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_x), weights='quadratic'),
          cohen_kappa_score(val_y, regression_to_class(val_x, rounding_coefficients=rounder.coefficients()),
                            weights='quadratic'))

    # Vice versa
    rounder = OptimizedRounderV2()
    rounder.fit(val_x, val_y)
    print(rounder.coefficients())
    print('Score on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_x), weights='quadratic'),
          cohen_kappa_score(val_y, regression_to_class(val_x, rounding_coefficients=rounder.coefficients()),
                            weights='quadratic'))

    print('Score on APTOS',
          cohen_kappa_score(train_y, regression_to_class(train_x), weights='quadratic'),
          cohen_kappa_score(train_y, regression_to_class(train_x, rounding_coefficients=rounder.coefficients()),
                            weights='quadratic'))


def test_optimize_kappa_on_aptos2015_v2():
    pl1 = pd.read_csv('../data/aptos-2015/test_private_pseudolabel_round_1.csv')
    labeled_gt = dict((row['id_code'], row['diagnosis']) for i, row in pl1.iterrows())

    ids, train_x, train_y, train_y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=False,
                                                                    use_predictions=True)
    mask = np.zeros(len(ids), dtype=np.bool)
    for i, id in enumerate(ids):
        mask[i] = labeled_gt[id] >= 0

    ids = ids[mask]
    train_x = train_x[mask]
    train_y = train_y[mask]
    train_y_avg = train_y_avg[mask]

    ids, val_x, val_y, val_y_avg = prepare_inference_datasets(idrid_predictions, use_features=False,
                                                              use_predictions=True)
    rounder = OptimizedRounderV2()
    rounder.fit(train_x, train_y)
    print(rounder.coefficients())
    print('Score on APTOS',
          cohen_kappa_score(train_y, regression_to_class(train_y_avg), weights='quadratic'),
          cohen_kappa_score(train_y, rounder.predict(train_x), weights='quadratic'))

    print('Score on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_y_avg), weights='quadratic'),
          cohen_kappa_score(val_y, rounder.predict(val_x), weights='quadratic'))

    # Vice versa
    rounder = OptimizedRounderV2()
    rounder.fit(val_x, val_y)
    print(rounder.coefficients())
    print('Score on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_y_avg), weights='quadratic'),
          cohen_kappa_score(val_y, rounder.predict(val_x), weights='quadratic'))

    print('Score on APTOS',
          cohen_kappa_score(train_y, regression_to_class(train_y_avg), weights='quadratic'),
          cohen_kappa_score(train_y, rounder.predict(train_x), weights='quadratic'))


def test_stack_with_ada_boost():
    pl1 = pd.read_csv('../data/aptos-2015/test_private_pseudolabel_round_1.csv')
    labeled_gt = dict((row['id_code'], row['diagnosis']) for i, row in pl1.iterrows())

    ids, train_x, train_y, train_y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=False,
                                                                    use_predictions=True)
    mask = np.zeros(len(ids), dtype=np.bool)
    for i, id in enumerate(ids):
        mask[i] = labeled_gt[id] >= 0

    ids = ids[mask]
    train_x = train_x[mask]
    train_y = train_y[mask]
    train_y_avg = train_y_avg[mask]

    _, val_x, val_y, val_y_avg = prepare_inference_datasets(idrid_predictions, use_features=False,
                                                            use_predictions=True)

    from sklearn.ensemble import AdaBoostClassifier

    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, train_x, train_y, cv=5)
    print(scores)
    print(scores.mean())

    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(train_x, train_y)

    y_pred = clf.predict(train_x)
    df = pd.DataFrame.from_dict({
        'id_code': ids,
        'y_true': train_y,
        'y_pred': y_pred
    })

    negatives = df[df['y_pred'] != df['y_true']]
    negatives.to_csv('aptos_negatives.csv', index=None)

    print('Score on APTOS',
          cohen_kappa_score(train_y, regression_to_class(train_y_avg), weights='quadratic'),
          cohen_kappa_score(train_y, clf.predict(train_x), weights='quadratic'))

    print('Score on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_y_avg), weights='quadratic'),
          cohen_kappa_score(val_y, clf.predict(val_x), weights='quadratic'))


def test_median_kappa_on_idrid():
    y_preds = []
    for index, predictions, in enumerate(idrid_predictions):
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.read_pickle(predictions)

        y_true = predictions['diagnosis'].values
        y_pred = predictions['ordinal'].values
        y_preds.append(y_pred)

        print('Score on Idrid-test', index,
              cohen_kappa_score(y_true, regression_to_class(y_pred), weights='quadratic'))

    y_preds = np.row_stack(y_preds)
    y_pred_median = np.median(y_preds, axis=0)

    y_pred_avg = np.mean(y_preds, axis=0)
    print('Score on Idrid-test',
          cohen_kappa_score(y_true, regression_to_class(y_pred_avg), weights='quadratic'),
          cohen_kappa_score(y_true, regression_to_class(y_pred_median), weights='quadratic'))


def test_logistic_regression_on_idrid():
    ids, train_x, train_y, train_y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=False,
                                                                    use_predictions=True)
    ids, val_x, val_y, val_y_avg = prepare_inference_datasets(idrid_predictions, use_features=False,
                                                              use_predictions=True)

    params_lr = {
        'class_weight': ['balanced', None],
        'multi_class': ['multinomial', 'auto', 'ovr'],
        'solver': ['newton-cg', 'lbfgs'],
        'max_iter': [100, 250, 500, 1000, 2000, 5000],
        'fit_intercept': [True, False],
        'random_state': [42]
    }

    # {'class_weight': None, 'fit_intercept': True, 'max_iter': 100, 'multi_class': 'multinomial', 'random_state': 42, 'solver': 'lbfgs'}
    # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #                    intercept_scaling=1, l1_ratio=None, max_iter=100,
    #                    multi_class='multinomial', n_jobs=None, penalty='l2',
    #                    random_state=42, solver='lbfgs', tol=0.0001, verbose=0,
    #                    warm_start=False)
    # LR on Train 0.8058995695509186 0.8408521184285193
    # LR on IDRID 0.8260826380611839 0.8655784001198091

    lr_gs = GridSearchCV(LogisticRegression(), params_lr, cv=5, verbose=1)
    lr_gs.fit(train_x, train_y)

    print(lr_gs.best_params_)
    print(lr_gs.best_estimator_)

    print('LR on Train',
          cohen_kappa_score(train_y, regression_to_class(train_y_avg), weights='quadratic'),
          cohen_kappa_score(train_y, lr_gs.best_estimator_.predict(train_x), weights='quadratic'))

    print('LR on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_y_avg), weights='quadratic'),
          cohen_kappa_score(val_y, lr_gs.best_estimator_.predict(val_x), weights='quadratic'))

    with open('logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_gs.best_estimator_, f)


def test_knn_on_idrid():
    ids, train_x, train_y, train_y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=False,
                                                                    use_predictions=True)
    ids, val_x, val_y, val_y_avg = prepare_inference_datasets(idrid_predictions, use_features=False,
                                                              use_predictions=True)

    # {'algorithm': 'ball_tree', 'leaf_size': 8, 'n_neighbors': 64, 'p': 1, 'weights': 'distance'}
    # KNeighborsClassifier(algorithm='ball_tree', leaf_size=8, metric='minkowski',
    #                      metric_params=None, n_jobs=None, n_neighbors=64, p=1,
    #                      weights='distance')
    # LR on Train 0.8058995695509186 1.0
    # LR on IDRID 0.8260826380611839 0.8692778993435448

    # create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': [8, 16, 32, 64, 128],
                  'weights': ['uniform', 'distance'],
                  'p': [1, 2],
                  'algorithm': ['ball_tree', 'kd_tree'],
                  'leaf_size': [8, 16, 32, 64, 128]
                  }

    knn_gs = GridSearchCV(KNeighborsClassifier(), params_knn, cv=5, verbose=1, n_jobs=4)
    knn_gs.fit(train_x, train_y)

    print(knn_gs.best_params_)
    print(knn_gs.best_estimator_)

    print('LR on Train',
          cohen_kappa_score(train_y, regression_to_class(train_y_avg), weights='quadratic'),
          cohen_kappa_score(train_y, knn_gs.best_estimator_.predict(train_x), weights='quadratic'))

    print('LR on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_y_avg), weights='quadratic'),
          cohen_kappa_score(val_y, knn_gs.best_estimator_.predict(val_x), weights='quadratic'))

    with open('knn.pkl', 'wb') as f:
        pickle.dump(knn_gs.best_estimator_, f)


def test_rf_on_idrid():
    ids, train_x, train_y, train_y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=False,
                                                                    use_predictions=True)
    ids, val_x, val_y, val_y_avg = prepare_inference_datasets(idrid_predictions, use_features=False,
                                                              use_predictions=True)

    # {'criterion': 'gini', 'max_depth': 12, 'n_estimators': 64}
    # RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                        max_depth=12, max_features='auto', max_leaf_nodes=None,
    #                        min_impurity_decrease=0.0, min_impurity_split=None,
    #                        min_samples_leaf=1, min_samples_split=2,
    #                        min_weight_fraction_leaf=0.0, n_estimators=64,
    #                        n_jobs=None, oob_score=False, random_state=None,
    #                        verbose=0, warm_start=False)
    # LR on Train 0.8058995695509186 0.9164135768322392
    # LR on IDRID 0.8260826380611839 0.8900600400266845

    # create a dictionary of all values we want to test for n_neighbors
    params_rf = {'n_estimators': [8, 16, 32, 64, 128],
                 'criterion': ['gini', 'entropy'],
                 'max_depth': [2, 4, 6, 8, 12],
                 }

    forest_gs = GridSearchCV(RandomForestClassifier(), params_rf, cv=5, verbose=1, n_jobs=4)
    forest_gs.fit(train_x, train_y)

    print(forest_gs.best_params_)
    print(forest_gs.best_estimator_)

    print('LR on Train',
          cohen_kappa_score(train_y, regression_to_class(train_y_avg), weights='quadratic'),
          cohen_kappa_score(train_y, forest_gs.best_estimator_.predict(train_x), weights='quadratic'))

    print('LR on IDRID',
          cohen_kappa_score(val_y, regression_to_class(val_y_avg), weights='quadratic'),
          cohen_kappa_score(val_y, forest_gs.best_estimator_.predict(val_x), weights='quadratic'))

    with open('forest.pkl', 'wb') as f:
        pickle.dump(forest_gs.best_estimator_, f)


def test_pseudolabeling_aptos2019_round1():
    ids, x, y_true, y_average = prepare_inference_datasets(aptos2019_predictions, use_features=False,
                                                           use_predictions=True)

    y_round = to_numpy(regression_to_class(x))
    y_major = majority_voting(y_round, axis=1)

    y_agreement = y_round == np.expand_dims(y_major, -1)

    y_agreement_all = np.all(y_agreement, axis=1)
    y_agreement_all = np.sum(y_agreement, axis=1) >= 16
    print('Agreement', np.mean(y_agreement_all))
    print('Distribution', np.bincount(y_major[y_agreement_all]))

    y_true[y_agreement_all] = y_major[y_agreement_all]
    print(y_round)
    df = pd.DataFrame.from_dict({'id_code': ids, 'diagnosis': y_true})
    df.to_csv('../data/aptos-2019/test_pseudolabel_round_1.csv', index=None)


@pytest.mark.parametrize(['predictions', 'output_csv'], [
    (
            get_predictions(models=['heuristic_sinoussi', 'modest_williams', 'happy_wright'],
                            datasets=['aptos2015_train']),
            '../data/aptos-2015/aptos2015_train_pseudolabel_round_1.csv'
    ),
    (
            get_predictions(models=['heuristic_sinoussi', 'modest_williams', 'happy_wright'],
                            datasets=['aptos2015_test_public']),
            '../data/aptos-2015/aptos2015_test_public_pseudolabel_round_1.csv'
    ),
    (
            get_predictions(models=['heuristic_sinoussi', 'modest_williams', 'happy_wright'],
                            datasets=['aptos2015_test_private']),
            '../data/aptos-2015/aptos2015_test_private_pseudolabel_round_1.csv'
    ),
])
def test_pseudolabeling_aptos2015_round1(predictions, output_csv):
    print('Saving pseudolabels to ', output_csv)
    num_models = len(predictions)
    ids, x, y_true, y_average = prepare_inference_datasets(predictions,
                                                           use_features=False,
                                                           use_predictions=True)

    for i in range(num_models):
        print(fs.id_from_fname(predictions[i]),
              cohen_kappa_score(y_true, regression_to_class(x[:, i]), weights='quadratic'))

    y_round = to_numpy(regression_to_class(x))
    y_major = majority_voting(y_round, axis=1)

    y_agreement = y_round == np.expand_dims(y_major, -1)

    # y_agreement_all = np.all(y_agreement, axis=1)
    # y_agreement_all = np.sum(y_agreement, axis=1) >= 16
    y_agreement_all = y_major == y_true

    print('Agreement', np.mean(y_agreement_all))
    print('Distribution', np.bincount(y_major[y_agreement_all]))

    y_true[~y_agreement_all] = -100
    print(y_round)
    df = pd.DataFrame.from_dict({'id_code': ids, 'diagnosis': y_true})
    df.to_csv(output_csv, index=None)


@pytest.mark.parametrize(['predictions', 'output_csv'], [
    (
            get_predictions(models=['heuristic_sinoussi', 'modest_williams', 'happy_wright'],
                            datasets=['messidor2_train']),
            '../data/messidor_2/train_labels_pseudolabel_round_1.csv'
    )
])
def test_pseudolabeling_messirod_2_round1(predictions, output_csv):
    ids, x, y_true, y_average = prepare_inference_datasets(predictions,
                                                           use_features=False,
                                                           use_predictions=True)

    y_round = to_numpy(regression_to_class(x))
    y_major = majority_voting(y_round, axis=1)

    y_agreement = y_round == np.expand_dims(y_major, -1)

    num_models = x.shape[1]
    y_agreement_most = np.sum(y_agreement, axis=1) >= int(0.75 * num_models)
    # y_agreement_all = y_major == y_true

    print('Agreement', np.mean(y_agreement_most))
    print('Distribution', np.bincount(y_major[y_agreement_most]))

    y_major[~y_agreement_most] = -100
    print(y_round)
    df = pd.DataFrame.from_dict({'id_code': ids, 'diagnosis': y_major})
    df.to_csv(output_csv, index=None)


@pytest.mark.parametrize(['predictions'], [
    # No pseudolabeling
    (
            get_predictions(
                models=[
                    '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth'],
                datasets=['aptos2015_train']),
    ),
    (
            get_predictions(
                models=[
                    '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth'],
                datasets=['aptos2015_test_private']),
    ),
    (
            get_predictions(
                models=[
                    '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth'],
                datasets=['aptos2015_test_public']),
    ),
    (
            get_predictions(
                models=[
                    '../models/Aug22_17_29_seresnext50_gap_heuristic_sinoussi/seresnext50_gap_512_medium_aptos2019_messidor_idrid_fold0_heuristic_sinoussi.pth'],
                datasets=['idrid_test']),
    ),
    # With pseudolabeling
    (
            get_predictions(
                models=[
                    '../models/Sep05_12_15_seresnext50_gap_pl1_vibrant_johnson/warmup/checkpoints/seresnext50_gap_512_medium_fold0_vibrant_johnson.pth'],
                datasets=['aptos2015_train']),
    ),
    (
            get_predictions(
                models=[
                    '../models/Sep05_12_15_seresnext50_gap_pl1_vibrant_johnson/warmup/checkpoints/seresnext50_gap_512_medium_fold0_vibrant_johnson.pth'],
                datasets=['aptos2015_test_private']),
    ),
    (
            get_predictions(
                models=[
                    '../models/Sep05_12_15_seresnext50_gap_pl1_vibrant_johnson/warmup/checkpoints/seresnext50_gap_512_medium_fold0_vibrant_johnson.pth'],
                datasets=['aptos2015_test_public']),
    ),
    (
            get_predictions(
                models=[
                    '../models/Sep05_12_15_seresnext50_gap_pl1_vibrant_johnson/warmup/checkpoints/seresnext50_gap_512_medium_fold0_vibrant_johnson.pth'],
                datasets=['idrid_test']),
    ),
])
def test_evaluate_model(predictions):
    num_models = len(predictions)
    ids, x, y_true, y_average = prepare_inference_datasets(predictions,
                                                           use_features=False,
                                                           use_predictions=True)

    for i in range(num_models):
        print(fs.id_from_fname(predictions[i]),
              cohen_kappa_score(y_true, regression_to_class(x), weights='quadratic'))


@pytest.mark.parametrize(['train', 'validation'], [
    (
            get_predictions(
                models=['heuristic_sinoussi', 'modest_williams', 'happy_wright', 'epic_shaw', 'admiring_minsky'],
                datasets=['aptos2015_test_private']),
            get_predictions(
                models=['heuristic_sinoussi', 'modest_williams', 'happy_wright', 'epic_shaw', 'admiring_minsky'],
                datasets=['idrid_test']),
    )
])
def test_evaluate_model_v2(train, validation):
    num_models = len(train)
    ids, train_x, train_y_true, train_y_average = prepare_inference_datasets(train,
                                                                             use_features=False,
                                                                             use_predictions=True)

    ids, valid_x, valid_y_true, valid_y_average = prepare_inference_datasets(validation,
                                                                             use_features=False,
                                                                             use_predictions=True)

    for i in range(num_models):
        print(fs.id_from_fname(train[i]),
              cohen_kappa_score(train_y_true, regression_to_class(train_x[:, i]), weights='quadratic'),
              cohen_kappa_score(train_y_true, regression_to_class(valid_x[:, i]), weights='quadratic'),
              )

    print('Averaged',
          cohen_kappa_score(train_y_true, regression_to_class(train_y_average), weights='quadratic'),
          cohen_kappa_score(valid_y_true, regression_to_class(valid_y_average), weights='quadratic'))

    print('Median  ',
          cohen_kappa_score(train_y_true, regression_to_class(np.median(train_x, axis=1)), weights='quadratic'),
          cohen_kappa_score(valid_y_true, regression_to_class(np.median(valid_x, axis=1)), weights='quadratic'))

    print('TrimMean',
          cohen_kappa_score(train_y_true, regression_to_class(trim_mean(train_x, proportiontocut=0.1, axis=1)),
                            weights='quadratic'),
          cohen_kappa_score(valid_y_true, regression_to_class(trim_mean(valid_x, proportiontocut=0.1, axis=1)),
                            weights='quadratic'))

    rounder = OptimizedRounder()
    rounder.fit(train_y_average, train_y_true)

    print(rounder.coefficients())
    print('Optimized',
          cohen_kappa_score(train_y_true, rounder.predict(train_y_average, rounder.coefficients()),
                            weights='quadratic'),
          cohen_kappa_score(valid_y_true, rounder.predict(valid_y_average, rounder.coefficients()),
                            weights='quadratic'))


def _drop_features(df: pd.DataFrame):
    if 'features' in df:
        df = df.drop(columns=['features'])
    return df


def majority_voting(predictions, axis=0):
    predictions = np.array(predictions, dtype=int)
    maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=axis, arr=predictions)
    return maj


def prepare_inference_datasets(models, use_features=False, use_predictions=True) -> Tuple[np.ndarray,
                                                                                          np.ndarray, np.ndarray, np.ndarray]:
    x = []
    y_true = None
    y_average = []
    ids = None

    for model_i, predictions, in enumerate(models):
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.read_pickle(predictions)

        if 'diagnosis' in predictions:
            y_true = predictions['diagnosis'].values
        else:
            y_true = np.array([-100] * len(predictions))

        ids = predictions['image_id']

        if use_predictions:
            # logits = np.array(df['logits'].values.tolist())
            # x.append(logits)

            ordinal = np.array(predictions['ordinal'].tolist()).reshape(-1, 1)
            x.append(ordinal)
            y_average.append(ordinal)

        if use_features:
            features = np.array(predictions['features'].values.tolist())
            x.append(features)

    if len(x) > 1:
        x = np.concatenate(x, axis=1)

        y_average = np.concatenate(y_average, axis=1)
        y_average = np.mean(y_average, axis=1)
    else:
        x = x[0]
        y_average = y_average[0]

    return ids, x, y_true, y_average


def evaluate_on_datasets(predictor: ClassifierMixin, datasets):
    y_preds = []
    mean_kappa = []
    for i, (x, y_true) in enumerate(datasets):
        y_pred = predictor.predict(x)
        y_preds.append(y_pred)

        kappa_hold = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        mean_kappa.append(kappa_hold)

    print(np.mean(mean_kappa), mean_kappa)
    return y_preds


def test_knn():
    use_features = True

    holdout_x, holdout_y, holdout_y_avg = prepare_inference_datasets(idrid_predictions, use_features=use_features,
                                                                     use_predictions=True)
    print('Holdout', holdout_x.shape, holdout_y.shape)
    print('Holdout base score', cohen_kappa_score(holdout_y, holdout_y_avg, weights='quadratic'))

    x, y, y_avg = prepare_inference_datasets(aptos2015_predictions, use_features=use_features, use_predictions=True)
    print('Train', x.shape, y.shape)
    print('Train base score', cohen_kappa_score(y, y_avg, weights='quadratic'))

    steps = []

    if use_features:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        holdout_x = scaler.transform(holdout_x)

        pca = PCA(n_components=1024, random_state=42)
        print('Computing PCA')
        x = pca.fit_transform(x)
        holdout_x = pca.transform(holdout_x)
        print(x.shape)

        steps.append(('scaler', scaler))
        steps.append(('pca', pca))
    else:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        holdout_x = scaler.transform(holdout_x)
        steps.append(('scaler', scaler))

    eval_datasets = [(x, y), (holdout_x, holdout_y)]

    # create new a knn model
    knn = KNeighborsClassifier()
    # create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 128, 4),
                  'weights': ['uniform', 'distance']
                  }
    # use gridsearch to test all values for n_neighbors
    # knn_gs = GridSearchCV(knn, params_knn, cv=5, n_jobs=8, verbose=1)
    knn_gs = RandomizedSearchCV(knn, params_knn, n_iter=100, cv=5, n_jobs=4, random_state=42, verbose=1)
    # fit model to training data
    knn_gs.fit(x, y)

    # save best model
    knn_best = knn_gs.best_estimator_
    # check best n_neigbors value
    print(knn_gs.best_params_)
    evaluate_on_datasets(knn_best, eval_datasets)
