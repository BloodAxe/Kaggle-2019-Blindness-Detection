import albumentations as A
import cv2
import numpy as np
from skimage.measure import label
from skimage.morphology import remove_small_objects


def crop_black(image, tolerance=5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.threshold(gray, tolerance, 255, type=cv2.THRESH_BINARY, dst=gray)
    # cv2.threshold(gray, tolerance, 255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst=gray)
    cv2.medianBlur(gray, 7, gray)

    # Remove small objects that occupy less than 5% of an image
    min_size = 0.05 * int(image.shape[0] * image.shape[1])
    label_image = label(gray)
    label_image = remove_small_objects(label_image, min_size=min_size)
    gray = (label_image > 0).astype(np.uint8)

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


class CropBlackRegions(A.ImageOnlyTransform):
    def __init__(self, tolerance=5, p=1.):
        super().__init__(p=p)
        self.tolerance = tolerance

    def apply(self, img, **params):
        return crop_black(img, self.tolerance)

    def get_transform_init_args_names(self):
        return ('tolerance',)


def red_free(image):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4944099/
    The red-free version of this photo shows the new vessels at the optic disc more clearly.
    Altering the images, e.g. by using red-free, is a valuable tool for detecting retinopathy
    :param image:
    :return:
    """
    image = image.copy()
    image[..., 0] = 0
    return image


class RedFree(A.ImageOnlyTransform):
    def __init__(self, p=1):
        super().__init__(p=p)

    def apply(self, img, **params):
        return red_free(img)


def unsharp_mask(image, sigmaX=10):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


class UnsharpMask(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return unsharp_mask(img)

    def get_transform_init_args_names(self):
        return tuple()


def unsharp_mask_v2(image):
    filter = cv2.bilateralFilter(image,
                                 d=32,
                                 sigmaColor=75,
                                 sigmaSpace=15)

    multiplier = 6
    difference = cv2.addWeighted(image, multiplier, filter, -multiplier, 0, dtype=cv2.CV_32F)
    a_max = np.max(difference)
    a_min = np.min(difference)
    rng = max(a_max, -a_min, 1)
    scale = 127. / rng
    difference = difference * scale + 127
    difference = difference.astype(np.uint8)
    return difference


def unsharp_mask_v3(image):
    filter = cv2.bilateralFilter(image,
                                 d=32,
                                 sigmaColor=75,
                                 sigmaSpace=15)

    multiplier = 6
    difference = cv2.addWeighted(image, multiplier, filter, -multiplier, 0, dtype=cv2.CV_32F)
    difference = np.abs(difference).astype(np.uint8) * 2
    return difference


class UnsharpMaskV2(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return unsharp_mask_v3(img)

    def get_transform_init_args_names(self):
        return tuple()


def clahe_preprocessing(image, clip_limit=4.0, tile_grid_size=(18, 18)):
    image_norm = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_norm[:, :, 0] = clahe.apply(image[:, :, 0])
    image_norm[:, :, 1] = clahe.apply(image[:, :, 1])
    image_norm[:, :, 2] = clahe.apply(image[:, :, 2])

    # image_norm = cv2.addWeighted(image, 0.5, image_norm, 0.5, 0)
    return image_norm


class ChannelIndependentCLAHE(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, img, **params):
        return clahe_preprocessing(img)

    def get_transform_init_args_names(self):
        return tuple()


def get_preprocessing_transform(preprocessing: str) -> A.ImageOnlyTransform:
    assert preprocessing in {None, 'unsharp', 'unsharpv2', 'iclahe', 'clahe', 'redfree'}

    if preprocessing is None:
        return A.NoOp()

    if preprocessing == 'unsharp':
        return UnsharpMask(p=1)

    if preprocessing == 'unsharpv2':
        return UnsharpMaskV2(p=1)

    if preprocessing == 'iclahe':
        return ChannelIndependentCLAHE(p=1)

    if preprocessing == 'clahe':
        return A.CLAHE(p=1)

    if preprocessing == 'redfree':
        return RedFree(p=1)

    raise KeyError(f'Unsupported preprocessing method {preprocessing}')
