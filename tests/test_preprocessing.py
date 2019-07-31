import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import log
from pytorch_toolbelt.utils import fs

from retinopathy.augmentations import crop_black, unsharp_mask, clahe_preprocessing, AddMildDR, create_microaneurisms


def test_adddr_transform():
    for image_fname in [
        # '4_left.png',
        # '35_left.png',
        '44_right.png',
        # '68_right.png',
        # '92_left.png'
    ]:
        image = cv2.imread(image_fname)

        aug = AddMildDR(p=1)
        data = aug(image=image, diagnosis=0)
        assert data['diagnosis'] == 1
        cv2.imshow('image', image)
        cv2.imshow('image after', data['image'])
        cv2.waitKey(-1)

        data = aug(image=image, diagnosis=2)
        assert data['diagnosis'] == 2


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


def test_generate_wools():
    iterations = 100
    num_circles = 10
    acc = np.zeros((256, 256), dtype=np.float32)
    for i in range(iterations):
        img = np.zeros_like(acc)

        for j in range(num_circles):
            r = int(random.uniform(5,15))
            x = int(random.gauss(128, 21))
            y = int(random.gauss(128, 21))
            pt = int(x), int(y)
            cv2.circle(img, pt, r, color=1, thickness=cv2.FILLED)

        # cv2.imshow('Img', (img * 255).astype(np.uint8))
        # cv2.waitKey(30)
        acc += img

    acc /= iterations
    cv2.imshow('Acc', (acc * 255).astype(np.uint8))
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
