from retinopathy.lib.dataset import get_datasets


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
        assert data['image'].size(0) == 512
        assert data['image'].size(0) == 512

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
        assert data['image'].size(0) == 512
        assert data['image'].size(0) == 512

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
    print(len(train_ds), len(valid_ds))
