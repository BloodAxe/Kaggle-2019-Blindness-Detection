from sklearn.utils import compute_sample_weight
from torch.utils.data import WeightedRandomSampler, DataLoader

from retinopathy.dataset import get_datasets


def test_aptos2019():
    train_ds, valid_ds = get_datasets(data_dir='../data',
                                      use_aptos2019=True,
                                      use_aptos2015=False,
                                      use_idrid=False,
                                      use_messidor=False,
                                      image_size=(512, 512))
    print(len(train_ds), len(valid_ds))
    assert train_ds[0] is not None
    assert valid_ds[0] is not None

    num_train_samples = len(train_ds)
    for i in range(num_train_samples):
        data = train_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512

    num_valid_samples = len(valid_ds)
    for i in range(num_valid_samples):
        data = valid_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512


def test_aptos2015():
    train_ds, valid_ds = get_datasets(data_dir='../data',
                                      use_aptos2019=False,
                                      use_aptos2015=True,
                                      use_idrid=False,
                                      use_messidor=False,
                                      image_size=(512, 512))
    print(len(train_ds), len(valid_ds))
    assert train_ds[0] is not None
    assert valid_ds[0] is not None

    num_train_samples = len(train_ds)
    for i in range(num_train_samples):
        data = train_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512

    num_valid_samples = len(valid_ds)
    for i in range(num_valid_samples):
        data = valid_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512


def test_idrid():
    train_ds, valid_ds = get_datasets(data_dir='../data',
                                      use_aptos2019=False,
                                      use_aptos2015=False,
                                      use_idrid=True,
                                      use_messidor=False,
                                      image_size=(512, 512))
    print(len(train_ds), len(valid_ds))
    assert train_ds[0] is not None
    assert valid_ds[0] is not None

    num_train_samples = len(train_ds)
    for i in range(num_train_samples):
        data = train_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512

    num_valid_samples = len(valid_ds)
    for i in range(num_valid_samples):
        data = valid_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512


def test_messidor():
    train_ds, valid_ds = get_datasets(data_dir='../data',
                                      use_aptos2019=False,
                                      use_aptos2015=False,
                                      use_idrid=False,
                                      use_messidor=True,
                                      image_size=(512, 512))
    print(len(train_ds), len(valid_ds))
    assert train_ds[0] is not None
    assert valid_ds[0] is not None

    num_train_samples = len(train_ds)
    for i in range(num_train_samples):
        data = train_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512

    num_valid_samples = len(valid_ds)
    for i in range(num_valid_samples):
        data = valid_ds[i]
        assert data['image'].size(0) == 3
        assert data['image'].size(1) == 512
        assert data['image'].size(2) == 512


def test_all():
    train_ds, valid_ds = get_datasets(data_dir='../data',
                                      use_aptos2019=True,
                                      use_aptos2015=True,
                                      use_idrid=True,
                                      use_messidor=True,
                                      image_size=(512, 512))
    num_train_samples = len(train_ds)
    num_valid_samples = len(valid_ds)
    assert num_train_samples > num_valid_samples
    print(num_train_samples, num_valid_samples)

    train_ds, valid_ds = get_datasets(data_dir='../data',
                                      use_aptos2019=True,
                                      use_aptos2015=True,
                                      use_idrid=True,
                                      use_messidor=True,
                                      image_size=(512, 512), fold=1, folds=4)
    num_train_samples = len(train_ds)
    num_valid_samples = len(valid_ds)
    assert num_train_samples > num_valid_samples
    print(num_train_samples, num_valid_samples)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_balancing():
    train = pd.read_csv('aptos2015_train.csv')
    test = pd.read_csv('aptos2015_test.csv')
    train = pd.concat((train, test), sort=False)

    dataset_size = len(train)
    print(dataset_size)

    x = np.arange(dataset_size)
    y = train['level'].values

    weights = compute_sample_weight('balanced', y)
    weights = np.sqrt(weights)
    # class_weight = compute_class_weight('balanced', np.arange(5), y)
    # weights = class_weight[y]

    sampler = WeightedRandomSampler(weights, dataset_size)
    loader = DataLoader(x, sampler=sampler, batch_size=60)
    hits = np.zeros(dataset_size)

    plt.figure()
    plt.hist(y)
    plt.title('Original distribution')
    plt.show()

    labels = []
    for batch in loader:
        for image in batch:
            label = y[image]
            labels.append(label)
            hits[image] += 1

    plt.figure()
    plt.hist(labels)
    plt.title('Balanced distribution')
    plt.show()

    plt.figure()
    plt.hist(hits)
    plt.title('Hits')
    plt.show()
