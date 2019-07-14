import albumentations as A
import cv2
import numpy as np


def crop_black(image, tolerance=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > tolerance
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    # Sanity check for very dark images
    non_black_area = w * h
    image_area = image.shape[0] * image.shape[1]
    fg_ratio = non_black_area / image_area

    # If area of black region is more than half of the whole image area,
    # we do not crop it.
    if fg_ratio < 0.5:
        return image

    return image[y:y + h, x:x + w]


class CropBlackRegions(A.ImageOnlyTransform):
    def __init__(self, tolerance=15):
        super().__init__(always_apply=True, p=1)
        self.tolerance = tolerance

    def apply(self, img, **params):
        return crop_black(img, self.tolerance)

    def get_transform_init_args_names(self):
        return ('tolerance',)


def get_train_aug(image_size, augmentation=None):
    if augmentation is None:
        augmentation = 'none'

    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HARD = 3

    LEVELS = {
        'none': NONE,
        'light': LIGHT,
        'medium': MEDIUM,
        'hard': HARD
    }
    assert augmentation in LEVELS.keys()
    augmentation = LEVELS[augmentation]

    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),

        A.Compose([
            A.CoarseDropout(),
        ], p=float(augmentation > LIGHT)),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),

        A.OneOf([
            A.ISONoise(),
            A.GaussNoise(),
            A.GaussianBlur(),
            A.IAASharpen(),
            A.NoOp()
        ], p=float(augmentation > LIGHT)),

        A.Compose([
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.05,
                               rotate_limit=45,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha_affine=5, border_mode=cv2.BORDER_CONSTANT,
                               value=0),
            A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
        ], p=float(augmentation == HARD)),

        A.OneOf([
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(hue_shift_limit=5),
            A.CLAHE(),
            A.RGBShift(r_shift_limit=20, b_shift_limit=10, g_shift_limit=10)
        ], p=float(augmentation >= MEDIUM)),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ], p=float(augmentation == HARD)),

        # Horizontal/Vertical flips
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip()
        ], p=float(augmentation >= LIGHT)),

        A.Normalize()
    ])


def get_test_aug(image_size):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions(),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),
        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize()
    ])


def test_crop_black_regions():
    import matplotlib.pyplot as plt

    for image_fname in ['data/train_images/0a4e1a29ffff.png',
                        'data/train_images/0a61bddab956.png',
                        'tests/19150_right.jpeg']:
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = A.CLAHE(always_apply=True)(image=image)['image']
        cropped = crop_black(image)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[1].imshow(cropped)
        f.show()

        # cv2.imshow('cropped', cropped)
        # cv2.imshow('image', image)
        # cv2.waitKey(-1)
