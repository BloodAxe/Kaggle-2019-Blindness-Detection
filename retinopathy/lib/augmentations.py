import cv2
import albumentations as A
import numpy as np


def crop_black(image, tolerance=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return image[y:y + h, x:x + w]


class CropBlackRegions(A.ImageOnlyTransform):
    def __init__(self, tolerance=15):
        super().__init__(always_apply=True, p=1)
        self.tolerance = tolerance

    def apply(self, img, **params):
        return crop_black(img, self.tolerance)

    def get_transform_init_args_names(self):
        return ('tolerance', )

def test_crop_black_regions():
    for image_fname in ['data/train_images/0a4e1a29ffff.png', 'data/train_images/0a61bddab956.png']:
        image = cv2.imread(image_fname)
        cropped = crop_black(image)
        cv2.imshow('cropped', cropped)
        cv2.imshow('image', image)
        cv2.waitKey(-1)
