import math
import os
from typing import Tuple, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from retinopathy.augmentations import get_train_transform, get_test_transform


def get_class_names(coarse_grading=False):
    if coarse_grading:
        return [
            'No DR',
            'DR (Mild/Moderate/Severe)',
            'Proliferative DR'
        ]

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
        if image is None:
            raise FileNotFoundError(self.images[item])
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


def count_targets(targets):
    targets = np.array(targets)
    counts = []
    for i in range(len(get_class_names())):
        counts.append(np.sum(targets == i))

    return counts


def split_train_valid(x, y, fold=None, folds=4, random_state=42):
    """
    Common train/test split function
    :param x:
    :param y:
    :param fold:
    :param folds:
    :param random_state:
    :return:
    """
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
                                                              test_size=1.0 / folds,
                                                              shuffle=True,
                                                              stratify=y)

    assert len(train_x) and len(train_y) and len(valid_x) and len(valid_y)
    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    return train_x, valid_x, train_y, valid_y


def get_aptos2019_train(data_dir):
    aptos2019_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # Remove duplicates and wrong annotations
    ids_to_skip = set(APTOS2019_MISLABELED_DUPLICATES2 + APTOS2019_DUPLICATES)
    size_before = len(aptos2019_train)
    aptos2019_train = aptos2019_train[~aptos2019_train['id_code'].isin(ids_to_skip)]
    size_after = len(aptos2019_train)
    print('Dropped', size_before - size_after, 'bad samples')

    x = np.array(aptos2019_train['id_code'].apply(lambda x: os.path.join(data_dir, 'train_images_768', f'{x}.png')))
    y = np.array(aptos2019_train['diagnosis'], dtype=int)
    return x, y


def get_aptos2019_test(data_dir):
    aptos2019_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    x = np.array(aptos2019_test['id_code'].apply(lambda x: os.path.join(data_dir, 'test_images_768', f'{x}.png')))
    y = np.array([UNLABELED_CLASS] * len(x), dtype=int)
    return x, y


def get_aptos2015_train(dataset_dir, healthy_eye_fraction=1):
    aptos2015_train = pd.read_csv(os.path.join(dataset_dir, 'train_labels.csv'))
    aptos2015_train['image_path'] = aptos2015_train['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'train_images_768', f'{x}.png'))
    x = np.array(aptos2015_train['image_path'])
    y = np.array(aptos2015_train['diagnosis'], dtype=int)

    if healthy_eye_fraction != 1:
        healthy = y == 0
        num_healthy = int(np.sum(healthy) * healthy_eye_fraction)
        x = np.concatenate((x[~healthy], x[healthy][:num_healthy]))
        y = np.concatenate((y[~healthy], y[healthy][:num_healthy]))
    return x, y


def get_aptos2015_test_public(dataset_dir, healthy_eye_fraction=1):
    aptos2015_test = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))
    aptos2015_test = aptos2015_test[aptos2015_test['Usage'] == 'Public']
    aptos2015_test['image_path'] = aptos2015_test['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'test_images_768', f'{x}.png'))
    x = np.array(aptos2015_test['image_path'])
    y = np.array(aptos2015_test['diagnosis'], dtype=int)

    if healthy_eye_fraction != 1:
        healthy = y == 0
        num_healthy = int(np.sum(healthy) * healthy_eye_fraction)
        x = np.concatenate((x[~healthy], x[healthy][:num_healthy]))
        y = np.concatenate((y[~healthy], y[healthy][:num_healthy]))
    return x, y


def get_aptos2015_test_private(dataset_dir, healthy_eye_fraction=1):
    aptos2015_test = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))
    aptos2015_test = aptos2015_test[aptos2015_test['Usage'] == 'Private']
    aptos2015_test['image_path'] = aptos2015_test['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'test_images_768', f'{x}.png'))
    x = np.array(aptos2015_test['image_path'])
    y = np.array(aptos2015_test['diagnosis'], dtype=int)

    if healthy_eye_fraction != 1:
        healthy = y == 0
        num_healthy = int(np.sum(healthy) * healthy_eye_fraction)
        x = np.concatenate((x[~healthy], x[healthy][:num_healthy]))
        y = np.concatenate((y[~healthy], y[healthy][:num_healthy]))
    return x, y


def get_idrid_train(dataset_dir):
    idrid_train = pd.read_csv(os.path.join(dataset_dir, 'train_labels.csv'))
    idrid_train['image_path'] = idrid_train['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'train_images_768', f'{x}.png'))

    x = np.array(idrid_train['image_path'])
    y = np.array(idrid_train['diagnosis'], dtype=int)
    return x, y


def get_idrid_test(dataset_dir):
    idrid_test = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))
    idrid_test['image_path'] = idrid_test['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'test_images_768', f'{x}.png'))
    x = np.array(idrid_test['image_path'])
    y = np.array(idrid_test['diagnosis'], dtype=int)
    return x, y


def get_messidor(dataset_dir, include_grade_3=False):
    messidor_train = pd.read_csv(os.path.join(dataset_dir, 'train_labels.csv'))
    messidor_train['image_path'] = messidor_train['id_code'].apply(
        lambda x: os.path.join(dataset_dir, 'train_images_768', f'{x}.png'))

    x = np.array(messidor_train['image_path'])
    y = np.array(messidor_train['diagnosis'], dtype=int)

    # Grade 3 of Messidor grading protocol includes Neovascularization which is stage 4.
    # For consistency we drop grade images by default
    if not include_grade_3:
        x = x[y != 3]
        y = y[y != 3]
    return x, y


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


def append_train_test(existing, to_add):
    train_x, train_y, valid_x, valid_y = existing
    tx, vx, ty, vy = to_add
    train_x.extend(tx)
    train_y.extend(ty)
    valid_x.extend(vx)
    valid_y.extend(vy)
    return train_x, train_y, valid_x, valid_y


def get_dataset(datasets: List[str], folds=4, data_dir='data', random_state=42):
    """

    :param datasets: List "aptos2015-train/fold0", "messidor",
    :param fold:
    :param folds:
    :return:
    """

    all_x = []
    all_y = []
    sizes = []

    for ds in datasets:
        dataset_name = ds
        suffix = None
        if '/' in ds:
            dataset_name, suffix = ds.split('/')

        if dataset_name == 'aptos-2019-train':
            x, y = get_aptos2019_train(os.path.join(data_dir, 'aptos-2019'))
        elif dataset_name == 'aptos-2015-train':
            x, y = get_aptos2015_train(os.path.join(data_dir, 'aptos-2015'))
        elif dataset_name == 'aptos-2015-test-private':
            x, y = get_aptos2015_test_private(os.path.join(data_dir, 'aptos-2015'))
        elif dataset_name == 'aptos-2015-test-public':
            x, y = get_aptos2015_test_public(os.path.join(data_dir, 'aptos-2015'))
        elif dataset_name == 'idrid-train':
            x, y = get_idrid_train(os.path.join(data_dir, 'idrid'))
        elif dataset_name == 'idrid-test':
            x, y = get_idrid_test(os.path.join(data_dir, 'idrid'))
        elif dataset_name == 'messidor':
            x, y = get_messidor(os.path.join(data_dir, 'messidor'), include_grade_3=False)
        else:
            raise ValueError(dataset_name)

        if suffix is not None:
            fold = int(suffix)
            _, valid_x, _, valid_y = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
            x = valid_y
            y = valid_y

        all_x.extend(x)
        all_y.extend(y)
        sizes.append(len(x))

    return all_x, all_y, sizes


def get_datasets_universal(
        train_on: List[str],
        valid_on: List[str],
        data_dir='data',
        image_size=(512, 512),
        augmentation='medium',
        preprocessing=None,
        target_dtype=int,
        random_state=42,
        coarse_grading=False,
        folds=4) -> Tuple[RetinopathyDataset, RetinopathyDataset, List]:
    train_x, train_y, sizes = get_dataset(train_on, folds=folds, data_dir=data_dir, random_state=random_state)
    valid_x, valid_y, _ = get_dataset(valid_on, folds=folds, data_dir=data_dir, random_state=random_state)

    train_transform = get_train_transform(image_size,
                                          augmentation=augmentation,
                                          preprocessing=preprocessing,
                                          crop_black=False)
    valid_transform = get_test_transform(image_size,
                                         preprocessing=preprocessing,
                                         crop_black=False)

    train_ds = RetinopathyDataset(train_x, train_y,
                                  transform=train_transform,
                                  dtype=target_dtype)

    valid_ds = RetinopathyDataset(valid_x, valid_y,
                                  transform=valid_transform,
                                  dtype=target_dtype)

    return train_ds, valid_ds, sizes


def get_datasets(
        data_dir='data',
        image_size=(512, 512),
        augmentation='medium',
        preprocessing=None,
        use_aptos2019=True,
        use_aptos2019_test_pl1=False,
        use_aptos2015_pl1=False,
        use_aptos2015=False,
        use_aptos2015_test_private=False,
        use_idrid=False,
        use_messidor=False,
        use_messidor2_pl1=False,
        use_unsupervised=False,
        target_dtype=int,
        random_state=42,
        coarse_grading=False,
        fold=None,
        folds=4) -> Tuple[RetinopathyDataset, RetinopathyDataset, List]:
    assert use_aptos2019 or use_aptos2015 or use_aptos2015_test_private or use_idrid or use_messidor
    assert not (use_aptos2015 and use_aptos2015_pl1)

    trainset_sizes = []
    data_split = [], [], [], []

    aptos2019_dir = os.path.join(data_dir, 'aptos-2019')
    aptos2015_dir = os.path.join(data_dir, 'aptos-2015')

    if use_aptos2019:
        x, y = get_aptos2019_train(aptos2019_dir)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

    if use_aptos2015_pl1:
        # Add training data
        aptos2015_train_pseudolabel_round_1 = pd.read_csv(
            os.path.join(aptos2015_dir, 'aptos2015_train_pseudolabel_round_1.csv'))
        aptos2015_train_pseudolabel_round_1 = aptos2015_train_pseudolabel_round_1[
            aptos2015_train_pseudolabel_round_1['diagnosis'] != -100]
        x = np.array(aptos2015_train_pseudolabel_round_1['id_code'].apply(
            lambda x: os.path.join(aptos2015_dir, 'train_images_768', f'{x}.png')))
        y = np.array(aptos2015_train_pseudolabel_round_1['diagnosis'], dtype=int)

        # For training part of aptos2015 - add it conventionaly
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

        # For public test validation data add only unhealthy samples to train set
        aptos2015_test_public_pl1 = pd.read_csv(
            os.path.join(aptos2015_dir, 'aptos2015_test_public_pseudolabel_round_1.csv'))
        aptos2015_test_public_pl1 = aptos2015_test_public_pl1[aptos2015_test_public_pl1['diagnosis'] != -100]
        x = np.array(aptos2015_test_public_pl1['id_code'].apply(
            lambda x: os.path.join(aptos2015_dir, 'test_images_768', f'{x}.png')))
        y = np.array(aptos2015_test_public_pl1['diagnosis'], dtype=int)

        # For pseudolabeled data, we add only one fold of it to clear training data
        # From test set add only unhealthy
        train_x, valid_x, train_y, valid_y = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)

        train_x = train_x[train_y > 0]
        train_y = train_y[train_y > 0]

        split = train_x, valid_x, train_y, valid_y
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(train_x[0])

        # Add Aptos2015 private test to validation set entirely
        aptos2015_test_private_pl1 = pd.read_csv(os.path.join(aptos2015_dir, 'aptos2015_test_private_pseudolabel_round_1.csv'))
        aptos2015_test_private_pl1 = aptos2015_test_private_pl1[aptos2015_test_private_pl1['diagnosis'] != -100]
        x = np.array(aptos2015_test_private_pl1['id_code'].apply(lambda x: os.path.join(aptos2015_dir, 'test_images_768', f'{x}.png')))
        y = np.array(aptos2015_test_private_pl1['diagnosis'], dtype=int)

        # From test set add only unhealthy
        x = x[y > 0]
        y = y[y > 0]
        data_split = append_train_test(data_split, ([], x, [], y))

    if use_messidor2_pl1:
        messidor2_dir = os.path.join(data_dir, 'messidor_2')
        messidor2_pseudolabel_round_1 = pd.read_csv(os.path.join(messidor2_dir, 'train_labels_pseudolabel_round_1.csv'))
        confident_labels_mask = messidor2_pseudolabel_round_1['diagnosis'] != -100
        messidor2_pseudolabel_round_1 = messidor2_pseudolabel_round_1[confident_labels_mask]

        x = np.array(messidor2_pseudolabel_round_1['id_code'].apply(
            lambda x: os.path.join(messidor2_dir, 'train_images_768', f'{x}.png')))
        y = np.array(messidor2_pseudolabel_round_1['diagnosis'], dtype=int)

        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

    if use_aptos2015:
        x, y = get_aptos2015_train(aptos2015_dir, healthy_eye_fraction=0.2)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

    if use_aptos2015_test_private:
        x, y = get_aptos2015_test_private(aptos2015_dir, healthy_eye_fraction=0.2)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

    if use_idrid:
        x, y = get_idrid_train(os.path.join(data_dir, 'idrid'))
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

    if use_messidor:
        x, y = get_messidor(os.path.join(data_dir, 'messidor'), include_grade_3=False)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(split[0])

    train_x, train_y, valid_x, valid_y = data_split

    # Regardless of used datasets let's use some data from validation (holdout)
    data_idrid_test = get_idrid_test(os.path.join(data_dir, 'idrid'))
    valid_x.extend(data_idrid_test[0])
    valid_y.extend(data_idrid_test[1])

    if use_aptos2015:
        data_aptos15_public = get_aptos2015_test_public(aptos2015_dir, healthy_eye_fraction=0.1)
        valid_x.extend(data_aptos15_public[0])
        valid_y.extend(data_aptos15_public[1])

    train_transform = get_train_transform(image_size,
                                          augmentation=augmentation,
                                          preprocessing=preprocessing,
                                          crop_black=False)
    valid_transform = get_test_transform(image_size,
                                         preprocessing=preprocessing,
                                         crop_black=False)

    if coarse_grading:
        assert not use_unsupervised

        coarse_grading_map = np.array([0, 1, 1, 1, 2])

        train_y = coarse_grading_map[np.array(train_y)]
        valid_y = coarse_grading_map[np.array(valid_y)]

    print('Train', count_targets(train_y), "Valid", count_targets(valid_y))

    if use_unsupervised:
        aptos2019, _ = get_aptos2019_test(aptos2019_dir)
        print('Adding', len(aptos2019), 'unlabeled samples from aptos2019 (test)')

        diaretdb0_v_1_1 = fs.find_images_in_dir(os.path.join(data_dir, 'diaretdb0_v_1_1', 'train_images_768'))
        print('Adding', len(diaretdb0_v_1_1), 'unlabeled samples from diaretdb0_v_1_1')

        diaretdb1_v_1_1 = fs.find_images_in_dir(os.path.join(data_dir, 'diaretdb1_v_1_1', 'train_images_768'))
        print('Adding', len(diaretdb1_v_1_1), 'unlabeled samples from diaretdb1_v_1_1')

        origa1 = fs.find_images_in_dir(os.path.join(data_dir, 'origa', 'glaucoma_768'))
        print('Adding', len(origa1), 'unlabeled samples from origa1')

        origa2 = fs.find_images_in_dir(os.path.join(data_dir, 'origa', 'sanas_768'))
        print('Adding', len(origa2), 'unlabeled samples from origa2')

        stare = fs.find_images_in_dir(os.path.join(data_dir, 'stare', 'train_images_768'))
        print('Adding', len(stare), 'unlabeled samples from stare')

        unlabeled_samples = diaretdb0_v_1_1 + diaretdb1_v_1_1 + stare + origa1 + origa2 + aptos2019.tolist()

        if not use_messidor:
            messidor = fs.find_images_in_dir(os.path.join(data_dir, 'messidor', 'train_images_768'))
            unlabeled_samples += messidor
            print('Adding', len(messidor), 'unlabeled samples from Messidor')

        if not use_aptos2015:
            dataset_dir = os.path.join(data_dir, 'aptos-2015')
            x, y = get_aptos2015_train(dataset_dir, healthy_eye_fraction=0.1)
            unlabeled_samples += x.tolist()
            print('Adding', len(x), 'unlabeled samples from Aptos 2015')

        if not use_aptos2015_test_private:
            dataset_dir = os.path.join(data_dir, 'aptos-2015')
            x, y = get_aptos2015_test_private(dataset_dir, healthy_eye_fraction=0.1)
            unlabeled_samples += x.tolist()
            print('Adding', len(x), 'unlabeled samples from Aptos 2015 Test (Private)')

        unlabeled_targets = [UNLABELED_CLASS] * len(unlabeled_samples)
        print('Using', len(unlabeled_samples), 'unlabeled samples')

        train_x.extend(unlabeled_samples)
        train_y.extend(unlabeled_targets)

        train_ds = RetinopathyDatasetV2(train_x, train_y,
                                        transform=train_transform,
                                        normalize=valid_transform,
                                        dtype=target_dtype)
        trainset_sizes.append(len(unlabeled_samples))
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
        weights = compute_sample_weight('balanced', train_ds.targets)
        hist = np.bincount(train_ds.targets)
        min_class_counts = int(min(hist))
        num_classes = len(np.unique(train_ds.targets))
        num_samples = min_class_counts * num_classes

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
