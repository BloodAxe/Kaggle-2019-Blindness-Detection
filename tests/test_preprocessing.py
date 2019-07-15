import cv2
import matplotlib.pyplot as plt
from pytorch_toolbelt.utils import fs

from retinopathy.lib.augmentations import crop_black, contrast_enchance


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
        fs.auto_file('0a4e1a29ffff.png', where='..'),
        fs.auto_file('0a61bddab956.png', where='..'),
        fs.auto_file('19150_right.jpeg', where='..'),
        fs.auto_file('3704_left.jpeg', where='..')
    ]:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # image = A.CLAHE(always_apply=True)(image=image)['image']
        image = crop_black(image, tolerance=5)
        cropped = contrast_enchance(image)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)
