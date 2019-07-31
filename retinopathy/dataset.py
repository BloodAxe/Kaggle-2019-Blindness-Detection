import os
from typing import Tuple, List

import albumentations as A
import cv2
import math
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from retinopathy.augmentations import get_train_transform, get_test_transform


def get_class_names():
    CLASS_NAMES = [
        'No DR',
        'Mild',
        'Moderate',
        'Severe',
        'Proliferative DR'
    ]
    return CLASS_NAMES


UNLABELED_CLASS = -100


class RetinopathyDataset(Dataset):
    def __init__(self, images, targets,
                 transform: A.Compose,
                 target_as_array=False,
                 dtype=int,
                 meta_features=False):
        if targets is not None:
            targets = np.array(targets)
            unique_targets = set(targets)
            if len(unique_targets.difference({0, 1, 2, 3, 4, UNLABELED_CLASS})):
                raise ValueError('Unexpected targets in Y ' + str(unique_targets))

        self.meta_features = meta_features
        self.images = np.array(images)
        self.targets = targets
        self.transform = transform
        self.target_as_array = target_as_array
        self.dtype = dtype

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])  # Read with OpenCV instead PIL. It's faster
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        diagnosis = UNLABELED_CLASS
        if self.targets is not None:
            diagnosis = self.targets[item]

        data = self.transform(image=image, diagnosis=diagnosis)
        diagnosis = data['diagnosis']
        data = {'image': tensor_from_rgb_image(data['image']),
                'image_id': id_from_fname(self.images[item])}

        if self.meta_features:
            log_height = math.log(height)
            log_width = math.log(width)
            aspect_ratio = log_height / log_width
            mean = np.mean(image, axis=(0, 1))

            meta_features = np.array([
                log_height,
                log_width,
                aspect_ratio,
                mean[0],
                mean[1],
                mean[2]
            ])
            data['meta_features'] = meta_features

        diagnosis = self.dtype(diagnosis)
        if self.target_as_array:
            data['targets'] = np.array([diagnosis])
        else:
            data['targets'] = diagnosis

        return data


class RetinopathyDatasetV2(Dataset):
    """
    Implementation of dataset for use with unsupervised learning
    """

    def __init__(self, images, targets,
                 transform: A.Compose,
                 normalize: A.Compose,
                 target_as_array=False,
                 dtype=int,
                 meta_features=False):
        if targets is not None:
            targets = np.array(targets)
            unique_targets = set(targets)
            if len(unique_targets.difference({0, 1, 2, 3, 4, -100})):
                raise ValueError('Unexpected targets in Y ' + str(unique_targets))

        self.meta_features = meta_features
        self.images = np.array(images)
        self.targets = targets
        self.transform = transform
        self.normalize = normalize
        self.target_as_array = target_as_array
        self.dtype = dtype

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])  # Read with OpenCV instead PIL. It's faster
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        original = self.normalize(image=image)['image']
        transformed = self.transform(image=image)['image']

        data = {'image': tensor_from_rgb_image(transformed),
                'original': tensor_from_rgb_image(original),
                'image_id': id_from_fname(self.images[item])}

        if self.meta_features:
            log_height = math.log(height)
            log_width = math.log(width)
            aspect_ratio = log_height / log_width
            mean = np.mean(image, axis=(0, 1))

            meta_features = np.array([
                log_height,
                log_width,
                aspect_ratio,
                mean[0],
                mean[1],
                mean[2]
            ])

            data['meta_features'] = meta_features

        if self.targets is not None:
            target = self.dtype(self.targets[item])
            if self.target_as_array:
                data['targets'] = np.array([target])
            else:
                data['targets'] = target

        return data


def get_aptos2019(data_dir,
                  random_state=42,
                  fold=None,
                  folds=4):
    aptos2019_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # Remove duplicates and wrong annotations
    ids_to_skip = set(APTOS2019_MISLABELED_DUPLICATES2 + APTOS2019_DUPLICATES)
    size_before = len(aptos2019_train)
    aptos2019_train = aptos2019_train[~aptos2019_train['id_code'].isin(ids_to_skip)]
    size_after = len(aptos2019_train)
    print('Dropped', size_before - size_after, 'bad samples')

    x = np.array(aptos2019_train['id_code'].apply(lambda x: os.path.join(data_dir, 'train_images_768', f'{x}.png')))
    y = np.array(aptos2019_train['diagnosis'], dtype=int)

    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if fold is not None:
        assert 0 <= fold < folds
        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]
                break
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(x, y,
                                                              random_state=random_state,
                                                              test_size=0.1,
                                                              shuffle=True,
                                                              stratify=y)

    return train_x, valid_x, train_y, valid_y


def get_aptos2019_test(data_dir,
                       random_state=42,
                       fold=None,
                       folds=4):
    aptos2019_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    x = np.array(aptos2019_test['id_code'].apply(lambda x: os.path.join(data_dir, 'test_images_768', f'{x}.png')))
    y = np.array([UNLABELED_CLASS] * len(x), dtype=int)

    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if fold is not None:
        assert 0 <= fold < folds
        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]
                break
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(x, y,
                                                              random_state=random_state,
                                                              test_size=0.1,
                                                              shuffle=True,
                                                              stratify=y)

    return train_x, valid_x, train_y, valid_y


def get_aptos2015(dataset_dir,
                  random_state=42,
                  fold=None,
                  folds=4):
    aptos2015_train = pd.read_csv(os.path.join(dataset_dir, 'train_labels.csv'))
    aptos2015_train['image_path'] = aptos2015_train['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'train_images_768', f'{x}.png'))

    aptos2015_test = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))
    aptos2015_test['image_path'] = aptos2015_test['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'test_images_768', f'{x}.png'))

    aptos2015 = aptos2015_train.append(aptos2015_test, sort=True)

    x = np.array(aptos2015['image_path'])
    y = np.array(aptos2015['diagnosis'], dtype=int)

    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if fold is not None:
        assert 0 <= fold < folds
        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]
                break
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(x, y,
                                                              random_state=random_state,
                                                              test_size=0.1,
                                                              shuffle=True,
                                                              stratify=y)

    return train_x, valid_x, train_y, valid_y


def get_idrid(dataset_dir,
              random_state=42,
              fold=None,
              folds=4):
    idrid_train = pd.read_csv(os.path.join(dataset_dir, 'train_labels.csv'))
    idrid_train['image_path'] = idrid_train['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'train_images_768', f'{x}.png'))

    idrid_test = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))
    idrid_test['image_path'] = idrid_test['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'test_images_768', f'{x}.png'))

    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if fold is not None:
        assert 0 <= fold < folds

        idrid_full = idrid_train.append(idrid_test, sort=True)
        x = np.array(idrid_full['image_path'])
        y = np.array(idrid_full['diagnosis'], dtype=int)

        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]
                break
    else:
        train_x = np.array(idrid_train['image_path'])
        train_y = np.array(idrid_train['diagnosis'], dtype=int)

        valid_x = np.array(idrid_test['image_path'])
        valid_y = np.array(idrid_test['diagnosis'], dtype=int)

    return train_x, valid_x, train_y, valid_y


def get_messidor(dataset_dir,
                 random_state=42,
                 fold=None,
                 folds=4):
    messidor_train = pd.read_csv(os.path.join(dataset_dir, 'train_labels.csv'))
    messidor_train['image_path'] = messidor_train['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'train_images_768', f'{x}.png'))

    x = np.array(messidor_train['image_path'])
    y = np.array(messidor_train['diagnosis'], dtype=int)

    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if fold is not None:
        assert 0 <= fold < folds
        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]
                break
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(x, y,
                                                              random_state=random_state,
                                                              test_size=0.1,
                                                              shuffle=True,
                                                              stratify=y)

    return train_x, valid_x, train_y, valid_y


APTOS2015_NOISE = {
    # https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/1440280229
    '25867_right_0': UNLABELED_CLASS,
    '25867_left_0': UNLABELED_CLASS,

    # https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/14402#80264
    '26064_left': UNLABELED_CLASS,
    '25360_left': UNLABELED_CLASS,
    '22026_left': 0,
    '21118_left': 0,

    # https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/1440280264
    '31202_right': 0,
    '31160_left': 0,
    '27481_right': 0,
    '26064_right': 0,

    '37244_right': 0,
    '34689_left': 0,
    '32541_left': 0,
    '32253_right': 0,

    '43457_left': 0,
    '42130_left': 0,
    '41693_right': 0,
    '41176_left': 0,

    '766_left': UNLABELED_CLASS,

    '2881_left': 0,
    '2516_left': 0,
    '1986_left': 0,
    '1557_left': UNLABELED_CLASS,

    '7470_right': 0,
    '7445_left': 0,
    '6511_right': 0,
    '3829_left': UNLABELED_CLASS,

    '20271_left': 0,
    '18972_left': UNLABELED_CLASS,
    '18085_right': 0,
    '15222_left': 0
}

APTOS2019_CORRECTIONS = {
    '06b71823f9cd': 4,
    'c1e6fa1ad314': 4,
    'f0098e9d4aee': 4,  # ??
    '29f44aea93a4': 1,  # ?

    # # Blurry
    # '6f923b60934b': UNLABELED_CLASS,
    # # '041f09eec1e8': UNLABELED_CLASS,
    # '22221cf5c7935': UNLABELED_CLASS,

}

APTOS2019_DUPLICATES = [
    '1632c4311fc9',
    '36041171f441',
    '6e92b1c5ac8e',
    '89ee1fa16f90',
    '111898ab463d',
    '7877be80901c',
    'f9e1c439d4c8',
    'b91ef82e723a',
    '4ce74e5eb51d',
    'aeed1f251ceb',
    '5a36cea278ae',
    '91b6ebaa3678',
    'd994203deb64',
    '7d261f986bef',
    '3044022c6969',
    '14515b8f19b6',
    'f9ecf1795804',
    '7550966ef777',
    'f920ccd926db',
    '2b48daf24be0',
    'bd5013540a13',
    'dc3c0d8ee20b',
    'e135d7ba9a0e',
    'ea05c22d92e9',
    'cd3fd04d72f5',
    '0161338f53cc',
    '3a1d3ce00f0c',
    '9b32e8ef0ca0',
    '98104c8c67eb',
    '1f07dae3cadb',
    '530d78467615',
    'f6f3ea0d2693',
    'd567a1a22d33',
    'bb5083fae98f',
    '4d7d6928534a',
    '14e3f84445f7',
    '00cb6555d108',
    '8a759f94613a',
    '05a5183c92d0',
    'b376def52ccc',
    '9e3510963315',
    'be161517d3ac',
    '81b0a2651c45',
    'bb7e0a2544cd',
    '5e7db41b3bee',
    '26fc2358a38d',
    '76cfe8967f7d'
]

# Here we list what should be ignored
APTOS2019_MISLABELED_DUPLICATES = [

    # diagnosis                                    path
    # 2          4.. / input / train_images / 6b00cb764237.png
    # 3          2.. / input / train_images / 64678182d8a8.png
    '6b00cb764237'  # Wrong annotation

    # diagnosis                                    path
    # 4          2.. / input / train_images / 8273fdb4405e.png
    # 5          1.. / input / train_images / f0098e9d4aee.png
    # Different images, keep both

    # diagnosis                                    path
    # 6          2.. / input / train_images / d801c0a66738.png
    # 7          4.. / input / train_images / 68332fdcaa70.png
    # IGNORE?

    # diagnosis                                    path
    # 8          0.. / input / train_images / ba735b286d62.png
    # 9          4.. / input / train_images / ed3a0fc5b546.png
    'ba735b286d62',  # Wrong annotation

    # diagnosis                                    path
    # 10          2.. / input / train_images / 36677b70b1ef.png
    # 11          0.. / input / train_images / 7bf981d9c7fe.png
    '7bf981d9c7fe',  # Wrong annotation

    # Need resolve
    # diagnosis                                    path
    # 12          3.. / input / train_images / 19e350c7c83c.png
    # 13          2.. / input / train_images / 19722bff5a09.png
    # Unclear stage

    # diagnosis                                    path
    # 14          3.. / input / train_images / 435d900fa7b2.png
    # 15          0.. / input / train_images / 1006345f70b7.png
    '1006345f70b7',  # Wrong annotation

    # diagnosis                                    path
    # 16          2.. / input / train_images / 278aa860dffd.png
    # 17          0.. / input / train_images / f066db7a2efe.png
    'f066db7a2efe',  # Wrong annotation

    # diagnosis                                    path
    # 20          0.. / input / train_images / f4d3777f2710.png
    # 21          1.. / input / train_images / 5dc23e440de3.png
    'f4d3777f2710',  # Wrong annotation

    # diagnosis                                    path
    # 24          1.. / input / train_images / a4012932e18d.png ?!?!?!
    # 25          0.. / input / train_images / 906d02fb822d.png
    '906d02fb822d',  # Wrong annotation

    # diagnosis                                    path
    # 26          4.. / input / train_images / 8fc09fecd22f.png
    # 27          0.. / input / train_images / d1cad012a254.png
    '8fc09fecd22f',  # Wrong annotation

    # diagnosis                                    path
    # 32          2.. / input / train_images / 7a0cff4c24b2.png
    # 33          0.. / input / train_images / 86baef833ae0.png
    '86baef833ae0',  # Wrong annotation

    # diagnosis                                    path
    # 34          0.. / input / train_images / 8ef2eb8c51c4.png
    # 35          2.. / input / train_images / 8446826853d0.png
    '8ef2eb8c51c4',  # Wrong annotation

    # Need resolve
    # diagnosis                                    path
    # 40          0.. / input / train_images / ca6842bfcbc9.png
    # 41          3.. / input / train_images / c027e5482e8c.png
    # 42          2.. / input / train_images / 7a3ea1779b13.png
    # 43          2.. / input / train_images / a8582e346df0.png
    'ca6842bfcbc9',  # Blurry
    'c027e5482e8c',  # Images
    '7a3ea1779b13',  # Ignore
    'a8582e346df0',  # Just in case

    # Need resolve
    # diagnosis                                    path
    # 44          1.. / input / train_images / 9a3c03a5ad0f.png
    # 45          0.. / input / train_images / f03d3c4ce7fb.png
    '9a3c03a5ad0f',  # 3?
    'f03d3c4ce7fb',  # 3?

    # diagnosis                                    path
    # 46          0.. / input / train_images / 1a1b4b2450ca.png
    # 47          3.. / input / train_images / 92b0d27fc0ec.png
    '1a1b4b2450ca',  # Wrong annotation

    # diagnosis                                    path
    # 52          0.. / input / train_images / 3c53198519f7.png
    # 53          1.. / input / train_images / 1c5e6cdc7ee1.png
    '3c53198519f7',  # Wrong annotation

    # diagnosis                                    path
    # 62          0.. / input / train_images / 8cb6b0efaaac.png
    # 63          0.. / input / train_images / 42a850acd2ac.png
    # 64          4.. / input / train_images / 51131b48f9d4.png
    '8cb6b0efaaac',  # Wrong annotation
    '42a850acd2ac',  # Wrong annotation

    # diagnosis                                    path
    # 65          2.. / input / train_images / 4a44cc840ebe.png
    # 66          0.. / input / train_images / 0cb14014117d.png
    '0cb14014117d',  # Wrong annotation

    # diagnosis                                    path
    # 69          0.. / input / train_images / 29f44aea93a4.png
    # 70          2.. / input / train_images / 7e6e90a93aa5.png
    # Corrected

    # TODO: Need resolve
    # diagnosis                                    path
    # 71          2.. / input / train_images / 7b691d9ced34.png
    # 72          4.. / input / train_images / d51c2153d151.png

    # diagnosis                                    path
    # 73          2.. / input / train_images / 9bf060db8376.png
    # 74          0.. / input / train_images / 4fecf87184e6.png
    # 75          0.. / input / train_images / f7edc074f06b.png
    '4fecf87184e6',  # Wrong annotation
    'f7edc074f06b',  # Wrong annotation

    # diagnosis                                    path
    # 76          0.. / input / train_images / aca88f566228.png
    # 77          2.. / input / train_images / c05b7b4c22fe.png
    'aca88f566228',  # Wrong annotation

    # Need resolve
    # diagnosis                                    path
    # 78          2.. / input / train_images / 878a3a097436.png
    # 79          0.. / input / train_images / 80feb1f7ca5e.png

    # Need resolve
    # diagnosis                                    path
    # 80          1.. / input / train_images / 46cdc8b685bd.png
    # 81          0.. / input / train_images / e4151feb8443.png
    'e4151feb8443',

    # diagnosis                                    path
    # 82          2.. / input / train_images / ea9e0fb6fb0b.png
    # 83          0.. / input / train_images / 23d7ca170bdb.png
    '23d7ca170bdb',

    # diagnosis                                    path
    # 84          4.. / input / train_images / 3dbfbc11e105.png
    # 85          0.. / input / train_images / d0079cc188e9.png
    'd0079cc188e9',  # Wrong annotation

    # diagnosis                                    path
    # 88          4.. / input / train_images / 4d9fc85a8259.png
    # 89          0.. / input / train_images / 16ce555748d8.png
    '16ce555748d8',

    # Need resolve
    # diagnosis                                    path
    # 90          1.. / input / train_images / 79ce83c07588.png
    # 91          0.. / input / train_images / 71c1a3cdbe47.png

    # diagnosis                                    path
    # 94          1.. / input / train_images / c8d2d32f7f29.png TODO: Verify grade
    # 95          0.. / input / train_images / 034cb07a550f.png
    '034cb07a550f',

    # Need resolve
    # diagnosis                                    path
    # 96          4.. / input / train_images / 38fe9f854046.png
    # 97          2.. / input / train_images / 1dfbede13143.png
    '38fe9f854046',  # Wrong annotation

    # Need resolve
    # diagnosis                                    path
    # 104          0.. / input / train_images / 98e8adcf085c.png
    # 105          1.. / input / train_images / 026dcd9af143.png
    # Need resolve
    # diagnosis                                    path
    # 108          0.. / input / train_images / e12d41e7b221.png
    # 109          4.. / input / train_images / bacfb1029f6b.png

    # Need resolve
    # diagnosis                                    path
    # 110          0.. / input / train_images / b13d72ceea26.png
    # 111          2.. / input / train_images / da0a1043abf7.png

    # Need resolve
    # diagnosis                                    path
    # 112          1.. / input / train_images / 0c7e82daf5a0.png
    # 113          2.. / input / train_images / 3e86335bc2fd.png

    # Need resolve
    # diagnosis                                    path
    # 114          0.. / input / train_images / 6165081b9021.png
    # 115          4.. / input / train_images / 42985aa2e32f.png

    # Need resolve
    # diagnosis                                    path
    # 116          0.. / input / train_images / 9c5dd3612f0c.png
    # 117          2.. / input / train_images / c9f0dc2c8b43.png
    # Need resolve
    # diagnosis                                    path
    # 120          0.. / input / train_images / 521d3e264d71.png
    # 121          4.. / input / train_images / fe0fc67c7980.png

    # Need resolve
    # diagnosis                                    path
    # 122          1.. / input / train_images / e8d1c6c07cf2.png
    # 123          0.. / input / train_images / f23902998c21.png

    # Need resolve
    # diagnosis                                    path
    # 124          0.. / input / train_images / 155e2df6bfcf.png
    # 125          4.. / input / train_images / 415f2d2bd2a1.png

    # Need resolve
    # diagnosis                                    path
    # 126          0.. / input / train_images / 9b7b6e4db1d5.png
    # 127          2.. / input / train_images / 9f4132bd6ed6.png
    # Need resolve
    # diagnosis                                    path
    # 130          0.. / input / train_images / 65e51e18242b.png
    # 131          1.. / input / train_images / cc12453ea915.png
    # Need resolve
    # diagnosis                                    path
    # 134          4.. / input / train_images / 76095c338728.png
    # 135          0.. / input / train_images / bd34a0639575.png
    # 136          2.. / input / train_images / de55ed25e0e8.png
    # 137          2.. / input / train_images / 84b79243e430.png
    # Need resolve
    # diagnosis                                    path
    # 140          2.. / input / train_images / ff0740cb484a.png
    # 141          0.. / input / train_images / b8ac328009e0.png
    # Need resolve
    # diagnosis                                    path
    # 142          3.. / input / train_images / b9127e38d9b9.png
    # 143          0.. / input / train_images / e39b627cf648.png
    # Need resolve
    # diagnosis                                    path
    # 144          3.. / input / train_images / f1a761c68559.png
    # 145          0.. / input / train_images / ff52392372d3.png
    # Need resolve
    # diagnosis                                    path
    # 146          2.. / input / train_images / 36ec36c301c1.png
    # 147          0.. / input / train_images / 26e231747848.png
    # Need resolve
    # diagnosis                                    path
    # 150          1.. / input / train_images / 0dce95217626.png
    # 151          4.. / input / train_images / 94372043d55b.png
    # Need resolve
    # diagnosis                                    path
    # 154          1.. / input / train_images / badb5ff8d3c7.png
    # 155          0.. / input / train_images / 2923971566fe.png
    # Need resolve
    # diagnosis                                    path
    # 156          2.. / input / train_images / 33778d136069.png
    # 157          3.. / input / train_images / 4ccfa0b4e96c.png
    # Need resolve
    # diagnosis                                    path
    # 160          2.. / input / train_images / 86b3a7929bec.png
    # 161          0.. / input / train_images / 1b4625877527.png
    # Need resolve
    # diagnosis                                    path
    # 162          1.. / input / train_images / 43fb6eda9b97.png
    # 163          0.. / input / train_images / e4e343eaae2a.png
    # Need resolve
    # diagnosis                                    path
    # 166          2.. / input / train_images / 135575dc57c9.png
    # 167          1.. / input / train_images / 2c2aa057afc5.png
    # Need resolve
    # diagnosis                                    path
    # 168          3.. / input / train_images / 40e9b5630438.png
    # 169          1.. / input / train_images / 77a9538b8362.png
    # Need resolve
    # diagnosis                                    path
    # 170          0.. / input / train_images / a8b637abd96b.png
    # 171          2.. / input / train_images / e2c3b037413b.png
    # Need resolve
    # diagnosis                                    path
    # 172          0.. / input / train_images / 1b862fb6f65d.png
    # 173          2.. / input / train_images / 0a4e1a29ffff.png
    # Need resolve
    # diagnosis                                    path
    # 174          0.. / input / train_images / bf7b4eae7ad0.png
    # 175          4.. / input / train_images / 496155f71d0a.png
    # Need resolve
    # diagnosis                                    path
    # 176          4.. / input / train_images / 81914ceb4e74.png
    # 177          0.. / input / train_images / d6b109c82067.png
    # 178          0.. / input / train_images / 1b398c0494d1.png
    # Need resolve
    # diagnosis                                    path
    # 179          2.. / input / train_images / 11242a67122d.png
    # 180          0.. / input / train_images / 65c958379680.png
    # Need resolve
    # diagnosis                                    path
    # 181          1.. / input / train_images / ea15a290eb96.png
    # 182          0.. / input / train_images / 1c9c583c10bf.png
    # Need resolve
    # diagnosis                                    path
    # 185          0.. / input / train_images / 668a319c2d23.png
    # 186          2.. / input / train_images / 4d167ca69ea8.png
    # Need resolve
    # diagnosis                                    path
    # 189          0.. / input / train_images / 7525ebb3434d.png
    # 190          2.. / input / train_images / 3cd801ffdbf0.png
    # Need resolve
    # diagnosis                                    path
    # 191          2.. / input / train_images / 1ee1eb7943db.png
    # 192          0.. / input / train_images / c2d2b4f536da.png
    # Need resolve
    # diagnosis                                    path
    # 193          0.. / input / train_images / 857002ed4e49.png
    # 194          2.. / input / train_images / 840527bc6628.png
    # Need resolve
    # diagnosis                                    path
    # 195          1.. / input / train_images / a3b2e93d058b.png
    # 196          0.. / input / train_images / 3fd7df6099e3.png
    # Need resolve
    # diagnosis                                    path
    # 201          0.. / input / train_images / c546670d9684.png
    # 202          2.. / input / train_images / 30cab14951ac.png
    # Need resolve
    # diagnosis                                    path
    # 203          0.. / input / train_images / 60f15dd68d30.png
    # 204          1.. / input / train_images / 772af553b8b7.png
    # 205          0.. / input / train_images / fcc6aa6755e6.png
    # Need resolve
    # diagnosis                                    path
    # 206          2.. / input / train_images / 3b018e8b7303.png
    # 207          1.. / input / train_images / 0243404e8a00.png
    # 208          0.. / input / train_images / 3ddb86eb530e.png
    # Need resolve
    # diagnosis                                    path
    # 211          0.. / input / train_images / 1e8a1fdee5b9.png
    # 212          3.. / input / train_images / a47432cd41e7.png
    # 213          0.. / input / train_images / b8ebedd382de.png
    # Need resolve
    # diagnosis                                    path
    # 214          3.. / input / train_images / 7005be54cab1.png
    # 215          2.. / input / train_images / 3ee4841936ef.png
    # Need resolve
    # diagnosis                                    path
    # 216          0.. / input / train_images / a7b0d0c51731.png
    # 217          2.. / input / train_images / 1cb814ed6332.png
    # Need resolve
    # diagnosis                                    path
    # 218          2.. / input / train_images / fea14b3d44b0.png
    # 219          0.. / input / train_images / 80d24897669f.png
    # Need resolve
    # diagnosis                                    path
    # 222          4.. / input / train_images / 35aa7f5c2ec0.png
    # 223          0.. / input / train_images / 1c4d87baaffc.png
    # Need resolve
    # diagnosis                                    path
    # 224          0.. / input / train_images / 7e980424868e.png
    # 225          2.. / input / train_images / b10fca20c885.png
    # Need resolve
    # diagnosis                                    path
    # 226          2.. / input / train_images / 98f7136d2e7a.png
    # 227          0.. / input / train_images / e740af6ac6ea.png
    # Need resolve
    # diagnosis                                    path
    # 234          0.. / input / train_images / df4913ca3712.png
    # 235          2.. / input / train_images / d51b3fe0fa1b.png
    # Need resolve
    # diagnosis                                    path
    # 236          0.. / input / train_images / 3ca637fddd56.png
    # 237          3.. / input / train_images / 3b4a5fcbe5e0.png
    # Need resolve
    # diagnosis                                    path
    # 238          0.. / input / train_images / e037643244b7.png
    # 239          2.. / input / train_images / 5b76117c4bcb.png
    # Need resolve
    # diagnosis                                    path
    # 240          2.. / input / train_images / 2f7789c1e046.png
    # 241          1.. / input / train_images / a8e88d4891c4.png
    # Need resolve
    # diagnosis                                    path
    # 242          0.. / input / train_images / 48c49f662f7d.png
    # 243          1.. / input / train_images / 6cb98da77e3e.png
    # Need resolve
    # diagnosis                                    path
    # 248          2.. / input / train_images / a56230242a95.png
    # 249          0.. / input / train_images / 1c6d119c3d70.png
    # Need resolve
    # diagnosis                                    path
    # 250          0.. / input / train_images / 9f1efb799b7b.png
    # 251          2.. / input / train_images / cd4e7f9fa1a9.png
    # Need resolve
    # diagnosis                                    path
    # 252          3.. / input / train_images / 5eb311bcb5f9.png
    # 253          2.. / input / train_images / a9e984b57556.png
    # Need resolve
    # diagnosis                                    path
    # 254          3.. / input / train_images / ce887b196c23.png
    # 255          2.. / input / train_images / e7a7187066ad.png
    # Need resolve
    # diagnosis                                    path
    # 256          0.. / input / train_images / 1e143fa3de57.png
    # 257          2.. / input / train_images / 144b01e7b993.png
    # Need resolve
    # diagnosis                                    path
    # 258          2.. / input / train_images / 8acffaf1f4b9.png
    # 259          0.. / input / train_images / 1411c8ab7161.png
    # Need resolve
    # diagnosis                                    path
    # 264          4.. / input / train_images / 1638404f385c.png
    # 265          2.. / input / train_images / 576e189d23d4.png
    # Need resolve
    # diagnosis                                    path
    # 268          0.. / input / train_images / 9f1b14dfa14c.png
    # 269          2.. / input / train_images / 435414ccccf7.png
    # Need resolve
    # diagnosis                                    path
    # 270          4.. / input / train_images / 6c3745a222da.png
    # 271          2.. / input / train_images / eadc57064154.png
    # Need resolve
    # diagnosis                                    path
    # 272          4.. / input / train_images / 2b21d293fdf2.png
    # 273          3.. / input / train_images / 2a3a1ed1c285.png
    # Need resolve
    # diagnosis                                    path
    # 276          1.. / input / train_images / d144144a2f3f.png
    # 277          2.. / input / train_images / b06dabab4f09.png
    # Need resolve
    # diagnosis                                    path
    # 282          3.. / input / train_images / 80964d8e0863.png
    # 283          1.. / input / train_images / ab50123abadb.png
    # Need resolve
    # diagnosis                                    path
    # 284          2.. / input / train_images / fda39982a810.png
    # 285          0.. / input / train_images / 0ac436400db4.png
    # Need resolve
    # diagnosis                                    path
    # 286          0.. / input / train_images / d85ea1220a03.png
    # 287          0.. / input / train_images / bfefa7344e7d.png
    # 288          2.. / input / train_images / 8688f3d0fcaf.png
    # Need resolve
    # diagnosis                                    path
    # 293          1.. / input / train_images / e1fb532f55df.png
    # 294          0.. / input / train_images / b019a49787c1.png
    # Need resolve
    # diagnosis                                    path
    # 295          1.. / input / train_images / cd93a472e5cd.png
    # 296          0.. / input / train_images / d035c2bd9104.png
    # 47

]

APTOS2019_MISLABELED_DUPLICATES2 = [
    # Need resolve
    '6b00cb764237',  # 4
    '64678182d8a8',  # 2

    # Need resolve
    '8273fdb4405e',  # 2
    'f0098e9d4aee',  # 1

    # Need resolve
    'd801c0a66738',  # 2
    '68332fdcaa70',  # 4

    # Need resolve
    'ba735b286d62',  # 0
    'ed3a0fc5b546',  # 4

    # Need resolve
    '36677b70b1ef',  # 2
    '7bf981d9c7fe',  # 0

    # Need resolve
    '19e350c7c83c',  # 3
    '19722bff5a09',  # 2

    # Need resolve
    '435d900fa7b2',  # 3
    '1006345f70b7',  # 0

    # Need resolve
    '278aa860dffd',  # 2
    'f066db7a2efe',  # 0

    # Need resolve
    'f4d3777f2710',  # 0
    '5dc23e440de3',  # 1

    # Need resolve
    'a4012932e18d',  # 1
    '906d02fb822d',  # 0

    # Need resolve
    '8fc09fecd22f',  # 4
    'd1cad012a254',  # 0

    # Need resolve
    '7a0cff4c24b2',  # 2
    '86baef833ae0',  # 0

    # Need resolve
    '8ef2eb8c51c4',  # 0
    '8446826853d0',  # 2

    # Need resolve
    'ca6842bfcbc9',  # 0
    'c027e5482e8c',  # 3
    '7a3ea1779b13',  # 2
    'a8582e346df0',  # 2

    # Need resolve
    '9a3c03a5ad0f',  # 1
    'f03d3c4ce7fb',  # 0

    # Need resolve
    '1a1b4b2450ca',  # 0
    '92b0d27fc0ec',  # 3

    # Need resolve
    '3c53198519f7',  # 0
    '1c5e6cdc7ee1',  # 1

    # Need resolve
    '8cb6b0efaaac',  # 0
    '42a850acd2ac',  # 0
    '51131b48f9d4',  # 4

    # Need resolve
    '4a44cc840ebe',  # 2
    '0cb14014117d',  # 0

    # Need resolve
    '29f44aea93a4',  # 0
    '7e6e90a93aa5',  # 2

    # Need resolve
    '7b691d9ced34',  # 2
    'd51c2153d151',  # 4

    # Need resolve
    '9bf060db8376',  # 2
    '4fecf87184e6',  # 0
    'f7edc074f06b',  # 0

    # Need resolve
    'aca88f566228',  # 0
    'c05b7b4c22fe',  # 2

    # Need resolve
    '878a3a097436',  # 2
    '80feb1f7ca5e',  # 0

    # Need resolve
    '46cdc8b685bd',  # 1
    'e4151feb8443',  # 0

    # Need resolve
    'ea9e0fb6fb0b',  # 2
    '23d7ca170bdb',  # 0

    # Need resolve
    '3dbfbc11e105',  # 4
    'd0079cc188e9',  # 0

    # Need resolve
    '4d9fc85a8259',  # 4
    '16ce555748d8',  # 0

    # Need resolve
    '79ce83c07588',  # 1
    '71c1a3cdbe47',  # 0

    # Need resolve
    'c8d2d32f7f29',  # 1
    '034cb07a550f',  # 0

    # Need resolve
    '38fe9f854046',  # 4
    '1dfbede13143',  # 2

    # Need resolve
    '98e8adcf085c',  # 0
    '026dcd9af143',  # 1

    # Need resolve
    'e12d41e7b221',  # 0
    'bacfb1029f6b',  # 4

    # Need resolve
    'b13d72ceea26',  # 0
    'da0a1043abf7',  # 2

    # Need resolve
    '0c7e82daf5a0',  # 1
    '3e86335bc2fd',  # 2

    # Need resolve
    '6165081b9021',  # 0
    '42985aa2e32f',  # 4

    # Need resolve
    '9c5dd3612f0c',  # 0
    'c9f0dc2c8b43',  # 2

    # Need resolve
    '521d3e264d71',  # 0
    'fe0fc67c7980',  # 4

    # Need resolve
    'e8d1c6c07cf2',  # 1
    'f23902998c21',  # 0

    # Need resolve
    '155e2df6bfcf',  # 0
    '415f2d2bd2a1',  # 4

    # Need resolve
    '9b7b6e4db1d5',  # 0
    '9f4132bd6ed6',  # 2

    # Need resolve
    '65e51e18242b',  # 0
    'cc12453ea915',  # 1

    # Need resolve
    '76095c338728',  # 4
    'bd34a0639575',  # 0
    'de55ed25e0e8',  # 2
    '84b79243e430',  # 2

    # Need resolve
    'ff0740cb484a',  # 2
    'b8ac328009e0',  # 0

    # Need resolve
    'b9127e38d9b9',  # 3
    'e39b627cf648',  # 0

    # Need resolve
    'f1a761c68559',  # 3
    'ff52392372d3',  # 0

    # Need resolve
    '36ec36c301c1',  # 2
    '26e231747848',  # 0

    # Need resolve
    '0dce95217626',  # 1
    '94372043d55b',  # 4

    # Need resolve
    'badb5ff8d3c7',  # 1
    '2923971566fe',  # 0

    # Need resolve
    '33778d136069',  # 2
    '4ccfa0b4e96c',  # 3

    # Need resolve
    '86b3a7929bec',  # 2
    '1b4625877527',  # 0

    # Need resolve
    '43fb6eda9b97',  # 1
    'e4e343eaae2a',  # 0

    # Need resolve
    '135575dc57c9',  # 2
    '2c2aa057afc5',  # 1

    # Need resolve
    '40e9b5630438',  # 3
    '77a9538b8362',  # 1

    # Need resolve
    'a8b637abd96b',  # 0
    'e2c3b037413b',  # 2

    # Need resolve
    '1b862fb6f65d',  # 0
    '0a4e1a29ffff',  # 2

    # Need resolve
    'bf7b4eae7ad0',  # 0
    '496155f71d0a',  # 4

    # Need resolve
    '81914ceb4e74',  # 4
    'd6b109c82067',  # 0
    '1b398c0494d1',  # 0

    # Need resolve
    '11242a67122d',  # 2
    '65c958379680',  # 0

    # Need resolve
    'ea15a290eb96',  # 1
    '1c9c583c10bf',  # 0

    # Need resolve
    '668a319c2d23',  # 0
    '4d167ca69ea8',  # 2

    # Need resolve
    '7525ebb3434d',  # 0
    '3cd801ffdbf0',  # 2

    # Need resolve
    '1ee1eb7943db',  # 2
    'c2d2b4f536da',  # 0

    # Need resolve
    '857002ed4e49',  # 0
    '840527bc6628',  # 2

    # Need resolve
    'a3b2e93d058b',  # 1
    '3fd7df6099e3',  # 0

    # Need resolve
    'c546670d9684',  # 0
    '30cab14951ac',  # 2

    # Need resolve
    '60f15dd68d30',  # 0
    '772af553b8b7',  # 1
    'fcc6aa6755e6',  # 0

    # Need resolve
    '3b018e8b7303',  # 2
    '0243404e8a00',  # 1
    '3ddb86eb530e',  # 0

    # Need resolve
    '1e8a1fdee5b9',  # 0
    'a47432cd41e7',  # 3
    'b8ebedd382de',  # 0

    # Need resolve
    '7005be54cab1',  # 3
    '3ee4841936ef',  # 2

    # Need resolve
    'a7b0d0c51731',  # 0
    '1cb814ed6332',  # 2

    # Need resolve
    'fea14b3d44b0',  # 2
    '80d24897669f',  # 0

    # Need resolve
    '35aa7f5c2ec0',  # 4
    '1c4d87baaffc',  # 0

    # Need resolve
    '7e980424868e',  # 0
    'b10fca20c885',  # 2

    # Need resolve
    '98f7136d2e7a',  # 2
    'e740af6ac6ea',  # 0

    # Need resolve
    'df4913ca3712',  # 0
    'd51b3fe0fa1b',  # 2

    # Need resolve
    '3ca637fddd56',  # 0
    '3b4a5fcbe5e0',  # 3

    # Need resolve
    'e037643244b7',  # 0
    '5b76117c4bcb',  # 2

    # Need resolve
    '2f7789c1e046',  # 2
    'a8e88d4891c4',  # 1

    # Need resolve
    '48c49f662f7d',  # 0
    '6cb98da77e3e',  # 1

    # Need resolve
    'a56230242a95',  # 2
    '1c6d119c3d70',  # 0

    # Need resolve
    '9f1efb799b7b',  # 0
    'cd4e7f9fa1a9',  # 2

    # Need resolve
    '5eb311bcb5f9',  # 3
    'a9e984b57556',  # 2

    # Need resolve
    'ce887b196c23',  # 3
    'e7a7187066ad',  # 2

    # Need resolve
    '1e143fa3de57',  # 0
    '144b01e7b993',  # 2

    # Need resolve
    '8acffaf1f4b9',  # 2
    '1411c8ab7161',  # 0

    # Need resolve
    '1638404f385c',  # 4
    '576e189d23d4',  # 2

    # Need resolve
    '9f1b14dfa14c',  # 0
    '435414ccccf7',  # 2

    # Need resolve
    '6c3745a222da',  # 4
    'eadc57064154',  # 2

    # Need resolve
    '2b21d293fdf2',  # 4
    '2a3a1ed1c285',  # 3

    # Need resolve
    'd144144a2f3f',  # 1
    'b06dabab4f09',  # 2

    # Need resolve
    '80964d8e0863',  # 3
    'ab50123abadb',  # 1

    # Need resolve
    'fda39982a810',  # 2
    '0ac436400db4',  # 0

    # Need resolve
    'd85ea1220a03',  # 0
    'bfefa7344e7d',  # 0
    '8688f3d0fcaf',  # 2

    # Need resolve
    'e1fb532f55df',  # 1
    'b019a49787c1',  # 0

    # Need resolve
    'cd93a472e5cd',  # 1
    'd035c2bd9104',  # 0

]


def get_datasets(
        data_dir='data',
        image_size=(512, 512),
        augmentation='medium',
        preprocessing=None,
        use_aptos2019=True,
        use_aptos2015=False,
        use_idrid=False,
        use_messidor=False,
        use_unsupervised=False,
        target_dtype=int,
        random_state=42,
        fold=None,
        folds=4) -> Tuple[RetinopathyDataset, RetinopathyDataset, List]:
    assert use_aptos2019 or use_aptos2015 or use_idrid or use_messidor
    trainset_sizes = []
    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if use_aptos2019:
        dataset_dir = os.path.join(data_dir, 'aptos-2019')
        tx, vx, ty, vy = get_aptos2019(dataset_dir, random_state, fold, folds)

        trainset_sizes.append(len(tx))
        train_x.extend(tx)
        train_y.extend(ty)
        valid_x.extend(vx)
        valid_y.extend(vy)

    if use_aptos2015:
        dataset_dir = os.path.join(data_dir, 'aptos-2015')
        tx, vx, ty, vy = get_aptos2015(dataset_dir, random_state, fold, folds)

        trainset_sizes.append(len(tx))
        train_x.extend(tx)

        if use_unsupervised:
            train_y.extend([UNLABELED_CLASS] * len(tx))
        else:
            train_y.extend(ty)

        valid_x.extend(vx)
        valid_y.extend(vy)

    if use_idrid:
        dataset_dir = os.path.join(data_dir, 'idrid')
        tx, vx, ty, vy = get_idrid(dataset_dir, random_state, fold, folds)

        trainset_sizes.append(len(tx))
        train_x.extend(tx)
        train_y.extend(ty)
        valid_x.extend(vx)
        valid_y.extend(vy)

    if use_messidor:
        dataset_dir = os.path.join(data_dir, 'messidor')
        tx, vx, ty, vy = get_messidor(dataset_dir, random_state, fold, folds)

        trainset_sizes.append(len(tx))
        train_x.extend(tx)
        train_y.extend(ty)
        valid_x.extend(vx)
        valid_y.extend(vy)

    train_transform = get_train_transform(image_size,
                                          augmentation=augmentation,
                                          preprocessing=preprocessing,
                                          crop_black=False)
    valid_transform = get_test_transform(image_size,
                                         preprocessing=preprocessing,
                                         crop_black=False)

    if use_unsupervised:
        dataset_dir = os.path.join(data_dir, 'aptos-2019')
        tx, vx, ty, vy = get_aptos2019_test(dataset_dir, random_state, fold, folds)

        trainset_sizes.append(len(tx))
        train_x.extend(tx)
        train_y.extend(ty)
        train_x.extend(vx)  # We append all data to train set
        train_y.extend(vy)  # We append all data to train set

        train_ds = RetinopathyDatasetV2(train_x, train_y,
                                        transform=train_transform,
                                        normalize=valid_transform,
                                        dtype=target_dtype)
    else:
        train_ds = RetinopathyDataset(train_x, train_y,
                                      transform=train_transform,
                                      dtype=target_dtype)

    valid_ds = RetinopathyDataset(valid_x, valid_y,
                                  transform=valid_transform,
                                  dtype=target_dtype)
    return train_ds, valid_ds, trainset_sizes


def get_dataloaders(train_ds, valid_ds,
                    batch_size,
                    num_workers,
                    fast=False,
                    train_sizes=None,
                    balance=False,
                    balance_datasets=False,
                    balance_unlabeled=False):
    sampler = None
    weights = None
    num_samples = 0

    if balance_unlabeled:
        labeled_mask = (train_ds.targets != UNLABELED_CLASS).astype(np.uint8)
        weights = compute_sample_weight('balanced', labeled_mask)
        num_samples = int(np.mean(train_sizes))

    if balance:
        class_weights = np.array([1, 2, 2, 2, 2, 2])
        weights = class_weights[train_ds.targets]
        num_samples = len(train_ds.targets)

    if balance_datasets:
        assert train_sizes is not None
        dataset_balancing_term = []

        for subset_size in train_sizes:
            full_dataset_size = float(sum(train_sizes))
            dataset_balancing_term.extend([full_dataset_size / subset_size] * subset_size)

        dataset_balancing_term = np.array(dataset_balancing_term)
        if weights is None:
            weights = np.ones(len(train_ds.targets))

        weights = weights * dataset_balancing_term
        num_samples = int(np.mean(train_sizes))

    # If we do balancing, let's go for fixed number of batches (half of dataset)
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples)

    if fast:
        weights = np.ones(len(train_ds))
        sampler = WeightedRandomSampler(weights, 16)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=sampler is None, sampler=sampler,
                          pin_memory=True, drop_last=True,
                          num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          pin_memory=True, drop_last=False,
                          num_workers=num_workers)

    return train_dl, valid_dl
