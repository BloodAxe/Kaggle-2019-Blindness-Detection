import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from albumentations.augmentations.functional import elastic_transform
from pytorch_toolbelt.utils import fs

from retinopathy.augmentations import crop_black, unsharp_mask, clahe_preprocessing


def create_microaneurisms(image, location=(256, 256), radius=140, num=5, aneurism_radius=(1, 3)):
    mask = image.copy()
    aneurism_mask = np.zeros_like(image)
    for i in range(num):
        x = int(random.gauss(location[0], radius))
        y = int(random.gauss(location[0], radius))
        r = int(random.uniform(aneurism_radius[0], aneurism_radius[1]))
        cv2.circle(aneurism_mask, (x, y), r, (255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        cv2.circle(mask, (x, y), r, (0, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    aneurism_mask = elastic_transform(aneurism_mask, alpha=5, sigma=2, alpha_affine=0)
    cv2.GaussianBlur(aneurism_mask, ksize=(5, 5), sigmaX=0, dst=aneurism_mask)
    aneurism_mask = 1.0 - aneurism_mask / 255.

    overlay = cv2.addWeighted(image, 0.8,
                              image * aneurism_mask, 0.2, 0, dtype=cv2.CV_8U)

    return overlay


def test_augment_microaneurisms():
    for image_fname in [
        # '4_left.png',
        # '35_left.png',
        # '44_right.png',
        '68_right.png',
        # '92_left.png'
    ]:
        image = cv2.imread(image_fname)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        location = (300, 300)
        radius = 140
        num = 10
        aneurism_radius = 1, 3

        overlay = create_microaneurisms(image, location, radius, num, aneurism_radius)
        cv2.imshow('image', image)
        cv2.imshow('overlay', overlay)
        cv2.imshow('clahe', unsharp_mask(overlay))
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


def test_enchance_contrast():
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
        cropped = clahe_preprocessing(image)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)
