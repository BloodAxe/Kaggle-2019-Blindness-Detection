import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytorch_toolbelt.utils import fs

from retinopathy.lib.augmentations import CropBlackRegions


def correct(image, w):
    map_x = np.zeros(image.shape[:2], dtype=np.float32)
    map_y = np.zeros(image.shape[:2], dtype=np.float32)

    Cx = image.shape[1] / 2.0
    Cy = image.shape[0] / 2.0

    for x in np.arange(-1.0, 1.0, 1.0 / Cx):
        for y in np.arange(-1.0, 1.0, 1.0 / Cy):
            ru = math.sqrt(x * x + y * y)
            rd = (1.0 / w) * math.atan(2.0 * ru * math.tan(w / 2.0))

            map_x[int(y * Cy + Cy), int(x * Cx + Cx)] = rd / ru * x * Cx + Cx
            map_y[int(y * Cy + Cy), int(x * Cx + Cx)] = rd / ru * y * Cy + Cy

    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return undistorted


def test_optical_correction():
    old_image = fs.read_rgb_image('../data/aptos-2015/test_images_768/6_right.png')
    new_image = fs.read_rgb_image('../data/aptos-2019/test_images_768/1c62728dd31b.png')

    plt.figure()
    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(old_image)
    ax[0, 1].imshow(new_image)

    # define W (185/2*CV_PI/180)
    W = 0
    old_image = correct(old_image, W)

    ax[1, 0].imshow(old_image)
    ax[1, 1].imshow(new_image)
    f.show()


def removeFisheyeLensDist(distorted, K, D, DIM=None):
    if DIM is None:
        DIM = distorted.shape[1], distorted.shape[0]

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(distorted, map1, map2,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    # undistorted = cv2.fisheye.undistortImage(distorted, K, D)
    return undistorted_img


import albumentations as A


@pytest.mark.parametrize('image_fname', ['44_right.png'])
def test_fisheye_undistortion(image_fname):
    image = cv2.cvtColor(cv2.imread(image_fname), cv2.COLOR_RGB2BGR)

    transform = A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(512),
        A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT)
    ])

    image = transform(image=image)['image']

    def update(*args, **kwargs):
        fx = cv2.getTrackbarPos('fx', 'Test')
        k = cv2.getTrackbarPos('k', 'Test')
        k_real = (k - 200) / 400

        print(fx, k_real)

        K = np.array([[fx, 0, 256],
                      [0, fx, 256],
                      [0, 0, 1]], dtype=np.float32)
        # D = np.array([-2.57614020e-01, 8.77086999e-02, -2.56970803e-04, -5.93390389e-04])
        D = np.array([[k_real], [k_real], [0], [0], ], dtype=np.float32)

        und = removeFisheyeLensDist(image, K, D, DIM=(768,768))
        cv2.imshow('Test', und)
        # cv2.waitKey(1)

    cv2.namedWindow('Test')
    cv2.createTrackbar('fx', 'Test', 400, 1024, update)
    cv2.createTrackbar('k', 'Test', 200, 400, update)
    update()
    while cv2.waitKey(30) != 'q':
        pass

    # plt.figure(figsize=(20,10))
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[0].axis('off')
    # ax[1].imshow(und)
    # ax[1].axis('off')
    # plt.title(image_fname)
    # plt.tight_layout()
    # plt.show()
