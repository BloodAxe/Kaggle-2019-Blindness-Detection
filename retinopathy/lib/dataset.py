import os

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


class RetinopathyDataset(Dataset):
    def __init__(self, images, targets, transform: A.Compose,
                 target_as_array=False, dtype=int):
        targets = np.array(targets) if targets is not None else None
        unique_targets = set(np.unique(targets))
        if len(unique_targets.difference({0, 1, 2, 3, 4})):
            raise ValueError('Unexpected targets in Y ' + str(unique_targets))

        self.images = np.array(images)
        self.targets = targets
        self.transform = transform
        self.target_as_array = target_as_array
        self.dtype = dtype

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(
            self.images[item])  # Read with OpenCV instead PIL. It's faster
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

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

        image = self.transform(image=image)['image']
        data = {'image': tensor_from_rgb_image(image),
                'image_id': id_from_fname(self.images[item]),
                'meta_features': meta_features}

        if self.targets is not None:
            target = self.dtype(self.targets[item])
            if self.target_as_array:
                data['targets'] = np.array([target])
            else:
                data['targets'] = target

        return data


def get_datasets(
        data_dir='data',
        image_size=(512, 512),
        augmentation='medium',
        use_aptos2019=True,
        use_aptos2015=False,
        use_idrid=False,
        use_messidor=False,
        target_dtype=int,
        fast=False,
        fold=None,
        folds=4):
    assert use_aptos2019 or use_aptos2015 or use_idrid or use_messidor

    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if use_aptos2019:
        dataset_dir = os.path.join(data_dir, 'aptos-2019')
        aptos2019_train = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
        aptos2019_train['image_path'] = aptos2019_train['id_code'].apply(
            lambda x: os.path.join(dataset_dir, 'train_images', f'{x}.png'))

        aptos2019_train_x, aptos2019_valid_x, aptos2019_train_y, aptos2019_valid_y = train_test_split(
            aptos2019_train['image_path'], aptos2019_train['diagnosis'],
            random_state=42, test_size=0.1, shuffle=True,
            stratify=aptos2019_train['diagnosis'])

        train_x.extend(aptos2019_train_x)
        train_y.extend(aptos2019_train_y)
        valid_x.extend(aptos2019_valid_x)
        valid_y.extend(aptos2019_valid_y)

    if use_aptos2015:
        dataset_dir = os.path.join(data_dir, 'aptos-2015')
        aptos2015_train = pd.read_csv(
            os.path.join(dataset_dir, 'train_labels.csv'))
        aptos2015_train['image_path'] = aptos2015_train['id_code'].apply(
            lambda x: os.path.join(dataset_dir, 'train_images', f'{x}.jpeg'))

        aptos2015_test = pd.read_csv(
            os.path.join(dataset_dir, 'test_labels.csv'))
        aptos2015_test['image_path'] = aptos2015_test['id_code'].apply(
            lambda x: os.path.join(dataset_dir, 'test_images', f'{x}.jpeg'))

        aptos2015 = aptos2015_train.append(aptos2015_test)

        aptos2015_train_x, aptos2015_valid_x, aptos2015_train_y, aptos2015_valid_y = train_test_split(
            aptos2015['image_path'], aptos2015['diagnosis'],
            random_state=42, test_size=0.1, shuffle=True,
            stratify=aptos2015['diagnosis'])

        train_x.extend(aptos2015_train_x)
        train_y.extend(aptos2015_train_y)
        valid_x.extend(aptos2015_valid_x)
        valid_y.extend(aptos2015_valid_y)

    if use_idrid:
        dataset_dir = os.path.join(data_dir, 'idrid')
        idrid_train = pd.read_csv(
            os.path.join(dataset_dir, 'train_labels.csv'))
        idrid_train['image_path'] = idrid_train['id_code'].apply(
            lambda x: os.path.join(dataset_dir, 'train_images', f'{x}.jpg'))

        idrid_test = pd.read_csv(os.path.join(dataset_dir, 'test_labels.csv'))
        idrid_test['image_path'] = idrid_test['id_code'].apply(
            lambda x: os.path.join(dataset_dir, 'test_images', f'{x}.jpg'))

        train_x.extend(idrid_train['image_path'])
        train_y.extend(idrid_train['diagnosis'])

        valid_x.extend(idrid_test['image_path'])
        valid_y.extend(idrid_test['diagnosis'])

    if use_messidor:
        dataset_dir = os.path.join(data_dir, 'messidor')
        messidor_train = pd.read_csv(
            os.path.join(dataset_dir, 'train_labels.csv'))
        messidor_train['image_path'] = messidor_train['id_code'].apply(
            lambda x: os.path.join(dataset_dir, 'train_images', f'{x}.tif'))

        messidor_train_x, messidor_valid_x, messidor_train_y, messidor_valid_y = train_test_split(
            messidor_train['image_path'], messidor_train['diagnosis'],
            random_state=42, test_size=0.1, shuffle=True,
            stratify=messidor_train['diagnosis'])

        train_x.extend(messidor_train_x)
        train_y.extend(messidor_train_y)
        valid_x.extend(messidor_valid_x)
        valid_y.extend(messidor_valid_y)

    if fold is not None:
        assert fold >= 0 and fold < folds

        x = train_x + valid_x
        y = train_y + valid_y

        skf = StratifiedKFold(n_splits=folds, random_state=13, shuffle=True)
        skf.get_n_splits(x, y)

        for fold_index, (train_index, test_index) in enumerate(
                skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]

    if fast:
        train_x = train_x[:32]
        train_y = train_y[:32]

        valid_x = valid_x[:32]
        valid_y = valid_y[:32]

    train_ds = RetinopathyDataset(train_x, train_y,
                                  transform=get_train_aug(image_size, augmentation),
                                  dtype=target_dtype)
    valid_ds = RetinopathyDataset(valid_x, valid_y,
                                  transform=get_test_aug(image_size),
                                  dtype=target_dtype)
    return train_ds, valid_ds


def get_dataloaders(train_ds, valid_ds,
                    batch_size,
                    num_workers,
                    balance=False):
    sampler = None
    if balance:
        weights = compute_sample_weight('balanced', train_ds.targets)
        sampler = WeightedRandomSampler(weights, len(train_ds))

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=sampler is None, sampler=sampler,
                          pin_memory=True, drop_last=True,
                          num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          pin_memory=True, drop_last=False,
                          num_workers=num_workers)

    return train_dl, valid_dl
