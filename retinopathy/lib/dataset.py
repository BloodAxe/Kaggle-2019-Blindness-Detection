import os

import albumentations as A
import cv2
import math
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from retinopathy.lib.augmentations import get_train_aug, get_test_aug


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

        image = self.transform(image=image)['image']
        data = {'image': tensor_from_rgb_image(image),
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


def get_datasets(
        data_dir='data',
        image_size=(512, 512),
        augmentation='medium',
        use_aptos2019=True,
        use_aptos2015=False,
        use_idrid=False,
        use_messidor=False,
        use_unsupervised=False,
        target_dtype=int,
        random_state=42,
        fold=None,
        folds=4):
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
        train_y.extend([UNLABELED_CLASS] * len(tx))
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

    train_transform = get_train_aug(image_size, augmentation, crop_black=False)
    valid_transform = get_test_aug(image_size, crop_black=False)

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
                    balance_datasets=False):
    sampler = None
    weights = None

    if balance:
        # class_weights = compute_class_weight('balanced', np.arange(5), train_ds.targets)
        class_weights = np.array([1, 0.5, 0.5, 0.5, 0.5])
        weights = class_weights[train_ds.targets]
        # weights = compute_sample_weight('balanced', train_ds.targets)

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

    # If we do balancing, let's go for fixed number of batches (half of dataset)
    if weights is not None:
        sampler = WeightedRandomSampler(weights, len(weights) // 2)

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
