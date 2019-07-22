import random

import albumentations as A
import cv2
from albumentations.augmentations.functional import brightness_contrast_adjust, center_crop, pad_with_params, gaussian_blur


def crop_black(image, tolerance=5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.threshold(gray, tolerance, 255, type=cv2.THRESH_BINARY, dst=gray)
    cv2.medianBlur(gray, 7, gray)

    x, y, w, h = cv2.boundingRect(gray)

    # Sanity check for very dark images
    non_black_area = w * h
    image_area = image.shape[0] * image.shape[1]
    fg_ratio = non_black_area / image_area

    # If area of black region is more than half of the whole image area,
    # we do not crop it.
    if fg_ratio < 0.5:
        return image

    return image[y:y + h, x:x + w]


def unsharp_mask(image, sigmaX=10):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def clahe_preprocessing(image, clip_limit=4.0, tile_grid_size=(18, 18)):
    image_norm = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_norm[:, :, 0] = clahe.apply(image[:, :, 0])
    image_norm[:, :, 1] = clahe.apply(image[:, :, 1])
    image_norm[:, :, 2] = clahe.apply(image[:, :, 2])

    # image_norm = cv2.addWeighted(image, 0.5, image_norm, 0.5, 0)
    return image_norm


class CropBlackRegions(A.ImageOnlyTransform):
    def __init__(self, tolerance=5, p=1.):
        super().__init__(p=p)
        self.tolerance = tolerance

    def apply(self, img, **params):
        return crop_black(img, self.tolerance)

    def get_transform_init_args_names(self):
        return ('tolerance',)


class UnsharpMask(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return unsharp_mask(img)

    def get_transform_init_args_names(self):
        return tuple()


class ChannelIndependentCLAHE(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return clahe_preprocessing(img)

    def get_transform_init_args_names(self):
        return tuple()


class IndependentRandomBrightnessContrast(A.ImageOnlyTransform):
    """ Change brightness & contrast independently per channels """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5):
        super(IndependentRandomBrightnessContrast, self).__init__(always_apply, p)
        self.brightness_limit = A.to_tuple(brightness_limit)
        self.contrast_limit = A.to_tuple(contrast_limit)

    def apply(self, img, **params):
        img = img.copy()
        for ch in range(img.shape[2]):
            alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
            beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
            img[..., ch] = brightness_contrast_adjust(img[..., ch], alpha, beta)

        return img


class ZeroTopAndBottom(A.ImageOnlyTransform):
    """
    Competition-specific augmentation erases top and bottom rows of the image.
    This makes from 'full eye' photos a 'rectangular' version.
    """

    def __init__(self, aspect_ratio=(1.0, 1.4), p=0.5):
        super().__init__(p=p)
        self.aspect_ratio = aspect_ratio

    def apply(self, img, aspect_ratio=1.0, **params):
        height, width = img.shape[:2]
        assert height == width
        new_height = int(width / aspect_ratio)

        h_pad_top = int((height - new_height) / 2.0)
        h_pad_bottom = height - new_height - h_pad_top

        img = img.copy()
        img[0:h_pad_top] = (0, 0, 0)
        img[height - h_pad_bottom:height] = (0, 0, 0)
        return img

    def get_params(self):
        return {'aspect_ratio': random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])}

    def get_transform_init_args_names(self):
        return ('max_aspect_ratio',)


class DestroyImage(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, **params):
        return 0

    def apply(self, img, blur_ksize=1.0, **params):
        img = gaussian_blur(img, ksize=blur_ksize)
        return img

    def get_params(self):
        return {'blur': random.uniform(8, 16) * 2 + 1}


def get_train_aug(image_size, augmentation=None, crop_black=True):
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
        CropBlackRegions() if crop_black else A.NoOp(always_apply=True),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),
        ChannelIndependentCLAHE(),

        A.Compose([
            A.CoarseDropout(max_height=32, max_width=32, min_height=8, min_width=8),
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
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1,
                               rotate_limit=45,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ElasticTransform(alpha_affine=5, border_mode=cv2.BORDER_CONSTANT,
                               value=0),
            A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0),
        ], p=float(augmentation == HARD)),

        A.OneOf([
            A.RandomBrightnessContrast(),
            IndependentRandomBrightnessContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(hue_shift_limit=5),
            A.CLAHE(),
            A.RGBShift(r_shift_limit=20, b_shift_limit=10, g_shift_limit=10)
        ], p=float(augmentation >= MEDIUM)),

        # Just flips
        A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip()
        ], p=float(augmentation == LIGHT)),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ], p=float(augmentation >= MEDIUM)),

        A.Normalize()
    ])


def get_test_aug(image_size, crop_black=True):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions() if crop_black else A.NoOp(always_apply=True),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),
        ChannelIndependentCLAHE(),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize()
    ])
