import lightgbm as lgb
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.torch_utils import to_numpy
from sklearn.model_selection import train_test_split

from retinopathy.callbacks import cohen_kappa_score
from retinopathy.models.common import regression_to_class


def main():
    f0_aptos15 = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_aptos2019_fold0_awesome_babbage_on_aptos2019_fold0.pkl'))
    f0_aptos15['fold'] = 0

    f0_idrid = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_idrid_fold0_heuristic_ptolemy_on_aptos2019_fold0.pkl'))
    f0_idrid['fold'] = 0

    f1_aptos15 = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_aptos2019_fold1_hopeful_khorana_on_aptos2019_fold1.pkl'))
    f1_aptos15['fold'] = 1

    f1_idrid = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_idrid_fold1_gifted_visvesvaraya_on_aptos2019_fold1.pkl'))
    f1_idrid['fold'] = 1

    f2_aptos15 = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_aptos2019_fold2_trusting_nightingale_on_aptos2019_fold2.pkl'))
    f2_aptos15['fold'] = 2

    f2_idrid = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_idrid_fold2_sharp_brattain_on_aptos2019_fold2.pkl'))
    f2_idrid['fold'] = 2

    f3_aptos15 = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_aptos2019_fold3_epic_wing_on_aptos2019_fold3.pkl'))
    f3_aptos15['fold'] = 3

    f3_idrid = pd.read_pickle(fs.auto_file('reg_seresnext50_rms_512_medium_mse_idrid_fold3_vibrant_minsky_on_aptos2019_fold3.pkl'))
    f3_idrid['fold'] = 3

    df_aptos15 = pd.concat([f0_aptos15, f1_aptos15, f2_aptos15, f3_aptos15])
    df_idrid = pd.concat([f0_idrid, f1_idrid, f2_idrid, f3_idrid])

    print(len(f0_aptos15), len(f1_aptos15), len(f2_aptos15), len(f3_aptos15), len(df_aptos15))

    # logits = np.array(df_aptos15['logits'].values.tolist())
    regression = np.array(df_aptos15['regression'].values.tolist())

    X = np.hstack((np.array(df_aptos15['features'].values.tolist()),
                   np.array(df_idrid['features'].values.tolist())))

    Y = np.array(df_aptos15['diagnosis_true'].values.tolist())

    print(X.shape, Y.shape)

    x_train, x_test, y_train, y_test, y_hat_train, y_hat_test = train_test_split(X, Y,
                                                                                 to_numpy(regression_to_class(regression)),
                                                                                 stratify=Y,
                                                                                 test_size=0.25,
                                                                                 random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # d_train = lgb.Dataset(x_train, label=y_train.astype(np.float32))
    #
    # params = {}
    # params['learning_rate'] = 0.003
    # params['boosting_type'] = 'gbdt'
    # params['objective'] = 'regression'
    # params['metric'] = 'mse'
    # params['sub_feature'] = 0.5
    # params['num_leaves'] = 10
    # params['min_data'] = 50
    # # params['max_depth'] = 4
    #
    # clf = lgb.train(params, d_train, 1000)
    #
    # y_pred = clf.predict(x_test)
    # y_pred_class = regression_to_class(y_pred)

    cls = LGBMClassifier(n_estimators=64, max_depth=10, random_state=0, num_leaves=256)
    cls.fit(x_train, y_train)
    y_pred_class = cls.predict(x_test)

    raw_score, num, denom = cohen_kappa_score(y_hat_test, y_test, weights='quadratic')
    print('raw_score', raw_score)
    lgb_score, num, denom = cohen_kappa_score(y_pred_class, y_test, weights='quadratic')
    print('lgb_score', lgb_score)


if __name__ == '__main__':
    main()
