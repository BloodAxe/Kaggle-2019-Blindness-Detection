import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytorch_toolbelt.utils import fs

from retinopathy.augmentations import AddMicroaneurisms, create_microaneurisms, \
    CropBlackRegions, create_cotton_wool_spots, AddCottonWools
from retinopathy.preprocessing import crop_black, clahe_preprocessing, unsharp_mask, red_free


def test_resizing():
    image = '19_right.jpeg'
    image = cv2.imread(image)

    t_cubic = A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(512, interpolation=cv2.INTER_CUBIC)
    ])

    t_lanczos = A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(512, interpolation=cv2.INTER_LANCZOS4)
    ])

    t_linear = A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(512, interpolation=cv2.INTER_LINEAR_EXACT)
    ])

    cv2.imshow('Cubic', t_cubic(image=image)['image'])
    cv2.imshow('Lanczos', t_lanczos(image=image)['image'])
    cv2.imshow('Linear', t_linear(image=image)['image'])
    cv2.waitKey(-1)


def test_adddr_transform():
    for image_fname in [
        # '4_left.png',
        # '35_left.png',
        '44_right.png',
        # '68_right.png',
        # '92_left.png'
    ]:
        image = cv2.imread(image_fname)

        aug = AddMicroaneurisms(p=1)
        data = aug(image=image, diagnosis=0)
        assert data['diagnosis'] == 1
        cv2.imshow('image', image)
        cv2.imshow('image after', data['image'])
        cv2.waitKey(-1)

        data = aug(image=image, diagnosis=2)
        assert data['diagnosis'] == 2


def test_augment_microaneurisms():
    aug = AddMicroaneurisms(p=1)
    for image_fname in [
        '4_left.png',
        # '35_left.png',
        '44_right.png',
        '68_right.png',
        # '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_with_aneurisms = aug(image=image, diagnosis=0)['image']

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_with_aneurisms = cv2.cvtColor(image_with_aneurisms, cv2.COLOR_RGB2BGR)
        mask = unsharp_mask(image_with_aneurisms)

        cv2.imshow('image', image)
        cv2.imshow('overlay', image_with_aneurisms)
        cv2.imshow('clahe', mask)
        cv2.waitKey(-1)


def test_generate_wools():
    aug = AddCottonWools(p=1)

    for image_fname in [
        '4_left.png',
        # '35_left.png',
        '44_right.png',
        '68_right.png',
        # '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_with_wools = aug(image=image, diagnosis=0)['image']

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = cv2.cvtColor(image_with_wools, cv2.COLOR_RGB2BGR)
        mask = unsharp_mask(overlay)

        cv2.imshow('image', image)
        cv2.imshow('overlay', overlay)
        cv2.imshow('clahe', mask)
        cv2.waitKey(-1)


def test_crop_black_regions():
    for image_fname in [
        # 'data/train_images/0a4e1a29ffff.png',
        # 'data/train_images/0a61bddab956.png',
        # 'tests/19150_right.jpeg',
        fs.auto_file('3704_left.jpeg', where='..')
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # image = A.CLAHE(always_apply=True)(image=image)['image']
        cropped = crop_black(image)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)


def brightness_and_contrast_auto(image, clipHistPercent=0):
    histSize = 256

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if clipHistPercent == 0:
        # keep full available range
        minGray, maxGray, minLoc, maxLoc = cv2.minMaxLoc(gray)
    else:

        # float range[] = { 0, 256 }
        # const float* histRange = { range };
        # bool uniform = true;
        # bool accumulate = false;
        hist = cv2.calcHist([gray], [0], mask=None, histSize=[256], ranges=[0, 256], accumulate=False)
        # r = cv2.calcHist(gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        # calculate cumulative distribution from the histogram
        accumulator = np.cumsum(hist[:, 0])

        # locate points that cuts at required value
        max = accumulator[-1]
        clipHistPercent *= ((max) / 100.0)  # make percent as absolute
        clipHistPercent /= 2.0  # left and right wings
        # locate left cut
        minGray = 0
        while (accumulator[minGray] < clipHistPercent):
            minGray += 1

        # locate right cut
        maxGray = histSize - 1
        while (accumulator[maxGray] >= (max - clipHistPercent)):
            maxGray -= 1

    inputRange = maxGray - minGray

    if inputRange == 0:
        return image

    alpha = (histSize - 1) / inputRange
    beta = -minGray * alpha
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def test_brightness_and_contrast_auto():
    for image_fname in [
        '4_left.png',
        '35_left.png',
        '44_right.png',
        '68_right.png',
        '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = A.CLAHE(always_apply=True)(image=image)['image']
        image = crop_black(image, tolerance=5)
        cropped = brightness_and_contrast_auto(image, 5)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)


def test_drop_red_channel():
    for image_fname in [
        '4_left.png',
        '35_left.png',
        '44_right.png',
        '68_right.png',
        '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = A.CLAHE(always_apply=True)(image=image)['image']
        image = crop_black(image, tolerance=5)
        cropped = red_free(image)

        median = cv2.medianBlur(image, ksize=15)
        norm = cv2.addWeighted(image, 0.5, median, -0.5, 128)
        # cropped[..., 0] = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        f, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        ax[2].imshow(norm)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)


def test_unsharp_mask():
    for image_fname in [
        '4_left.png',
        '35_left.png',
        '44_right.png',
        '68_right.png',
        '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # image = A.CLAHE(always_apply=True)(image=image)['image']
        image = crop_black(image, tolerance=5)
        cropped = unsharp_mask(image)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)


def test_enchance_contrast_clahe():
    for image_fname in [
        '4_left.png',
        '35_left.png',
        '44_right.png',
        '68_right.png',
        '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # image = A.CLAHE(always_apply=True)(image=image)['image']
        image = crop_black(image)
        cropped = clahe_preprocessing(image, clip_limit=3, tile_grid_size=(32, 32))

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)


def test_pca_whitening():
    for image_fname in [
        '4_left.png',
        '35_left.png',
        '44_right.png',
        '68_right.png',
        '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = image.astype(float).copy()

        img = img / 255.0  # rescale to 0 to 1 range

        # flatten image to columns of 3x3 RGB patches
        img_rs = img.reshape(-1, 3 * 9)
        # img_rs shape (640000, 3)

        # center mean
        img_centered = img_rs - np.mean(img_rs, axis=0)

        # paper says 3x3 covariance matrix
        img_cov = np.cov(img_centered, rowvar=False)

        # eigen values and eigen vectors
        eig_vals, eig_vecs = np.linalg.eigh(img_cov)

        #     eig_vals [0.00154689 0.00448816 0.18438678]

        #     eig_vecs [[ 0.35799106 -0.74045435 -0.56883192]
        #      [-0.81323938  0.05207541 -0.57959456]
        #      [ 0.45878547  0.67008619 -0.58352411]]

        # sort values and vector
        sort_perm = eig_vals[::-1].argsort()
        eig_vals[::-1].sort()
        eig_vecs = eig_vecs[:, sort_perm]
        print(eig_vecs)

        eig_matrix = eig_vecs.reshape((27, 3, 3, 3))

        f, ax = plt.subplots(eig_matrix.shape[0], 1, figsize=(1, eig_matrix.shape[0]))
        for i in range(eig_matrix.shape[0]):
            img = eig_matrix[i]

            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)
            ax[i].imshow(img, cmap='jet')
            ax[i].axis('off')
        f.show()

        print(eig_matrix)
        # ax[0].imshow(image)
        # ax[1].imshow(cropped)

        # # get [p1, p2, p3]
        # m1 = np.column_stack((eig_vecs))
        #
        # f, ax = plt.subplots(1, 2)
        # ax[0].imshow(image)
        # ax[1].imshow(cropped)
        # f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)
