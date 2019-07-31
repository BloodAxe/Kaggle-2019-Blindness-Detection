import random

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.functional import brightness_contrast_adjust, gaussian_blur, elastic_transform


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


class RedFree(A.ImageOnlyTransform):
    def __init__(self, p=1):
        super().__init__(p=p)

    def apply(self, img, **params):
        return red_free(img)


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

    def apply(self, img, blur_ksize=3, **params):
        img = gaussian_blur(img, ksize=blur_ksize)
        return img

    def get_params(self):
        return {'blur_ksize': int(random.uniform(8, 16)) * 2 + 1}


def fancy_pca(img, alpha=0.1):
    '''
    INPUTS:
    img:  numpy array with (h, w, rgb) shape, as ints between 0-255)
    alpha_std:  how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1
    RETURNS:
    numpy image-like array as float range(0, 1)
    NOTE: Depending on what is originating the image data and what is receiving
    the image data returning the values in the expected form is very important
    in having this work correctly. If you receive the image values as UINT 0-255
    then it's probably best to return in the same format. (As this
    implementation does). If the image comes in as float values ranging from
    0.0 to 1.0 then this function should be modified to return the same.
    Otherwise this can lead to very frustrating and difficult to troubleshoot
    problems in the image processing pipeline.
    This is 'Fancy PCA' from:
    # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    #######################
    #### FROM THE PAPER ###
    #######################
    "The second form of data augmentation consists of altering the intensities
    of the RGB channels in training images. Specifically, we perform PCA on the
    set of RGB pixel values throughout the ImageNet training set. To each
    training image, we add multiples of the found principal components, with
    magnitudes proportional to the corresponding eigenvalues times a random
    variable drawn from a Gaussian with mean zero and standard deviation 0.1.
    Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
    we add the following quantity:
    [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
    Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
    matrix of RGB pixel values, respectively, and αi is the aforementioned
    random variable. Each αi is drawn only once for all the pixels of a
    particular training image until that image is used for training again, at
    which point it is re-drawn. This scheme approximately captures an important
    property of natural images, namely, that object identity is invariant to
    change."
    ### END ###############
    Other useful resources for getting this working:
    # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
    # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2
    '''

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
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

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    # about 100x faster after vectorizing the numpy, it will be even faster later
    # since currently it's working on full size images and not small, square
    # images that will be fed in later as part of the post processing before being
    # sent into the model
    #     print("elapsed time: {:2.2f}".format(time.time() - start_time), "\n")

    return orig_img


class FancyPCA(A.ImageOnlyTransform):
    """
    https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
    https://pixelatedbrian.github.io/2018-04-29-fancy_pca/
    """

    def __init__(self, alpha_std=0.1, p=0.5):
        super().__init__(p=p)
        self.alpha_std = alpha_std

    def apply(self, img, alpha=0.1, **params):
        img = fancy_pca(img, alpha)
        return img

    def get_params(self):
        return {'alpha': random.gauss(0, self.alpha_std)}


def create_microaneurisms(image, location=(256, 256), radius=140, num=5, aneurism_radius=(1, 3), alpha=0.2):
    # mask = image.copy()
    aneurism_mask = np.zeros_like(image)
    for i in range(num):
        x = int(random.gauss(location[0], radius))
        y = int(random.gauss(location[0], radius))
        r = int(random.uniform(aneurism_radius[0], aneurism_radius[1]))
        cv2.circle(aneurism_mask, (x, y), r, (255, 255, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        # cv2.circle(mask, (x, y), r, (0, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    aneurism_mask = elastic_transform(aneurism_mask, alpha=5, sigma=2, alpha_affine=0)
    cv2.GaussianBlur(aneurism_mask, ksize=(5, 5), sigmaX=0, dst=aneurism_mask)
    aneurism_mask = 1.0 - aneurism_mask / 255.

    overlay = cv2.addWeighted(image, (1 - alpha),
                              image * aneurism_mask, alpha, 0, dtype=cv2.CV_8U)

    return overlay


class AddMildDR(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, apply=False, new_diagnosis=0, **params):
        if apply:
            return new_diagnosis
        return diagnosis

    def apply(self, img, apply=False, new_diagnosis=0, microaneurisms_count=0, alpha=0.2, **params):
        if apply:
            img = create_microaneurisms(img,
                                        location=[img.shape[1] // 2, img.shape[0] // 2],
                                        radius=min(img.shape[1], img.shape[0]) // 2 - 10,
                                        num=microaneurisms_count,
                                        aneurism_radius=(1, 4),
                                        alpha=alpha)
        return img

    def update_params(self, params, image=None, diagnosis=0, **kwargs):
        if diagnosis == 0:
            params['apply'] = True
            params['new_diagnosis'] = 1
            params['microaneurisms_count'] = int(random.uniform(3, 10))
            params['alpha'] = random.uniform(0.1, 0.3)

        return params

    def get_params(self):
        return {}


def get_preprocessing_transform(preprocessing):
    assert preprocessing in {None, 'unsharp', 'clahe'}

    if preprocessing is None:
        return A.NoOp()

    if preprocessing == 'unsharp':
        return UnsharpMask(p=1)

    if preprocessing == 'clahe':
        return ChannelIndependentCLAHE(p=1)

    raise KeyError(f'Unsupported preprocessing method {preprocessing}')


def get_train_transform(image_size, augmentation=None, preprocessing=None, crop_black=True):
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

        # Fake decease generation
        # AddMildDR(p=0.3),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),

        A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, value=0),

                A.NoOp()
            ])
        ], p=float(augmentation == LIGHT)),

        A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, value=0),
                A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    value=0),
                A.NoOp()
            ])
        ], p=float(augmentation == MEDIUM)),

        A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=45,
                                   border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ElasticTransform(alpha_affine=0,
                                   alpha=35,
                                   sigma=5,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   value=0),
                A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                    border_mode=cv2.BORDER_CONSTANT,
                                    value=0),
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT,
                                 value=0),
                A.NoOp()
            ])
        ], p=float(augmentation == HARD)),

        A.Compose([
            ZeroTopAndBottom(p=0.3),
        ], p=float(augmentation >= MEDIUM)),

        A.OneOf([
            A.ISONoise(),
            A.NoOp()
        ], p=float(augmentation == HARD)),

        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                contrast_limit=0.24),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ], p=float(augmentation >= LIGHT)),

        # Color augmentations
        A.OneOf([
            FancyPCA(alpha_std=4),
            A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=5),
            A.NoOp()
        ], p=float(augmentation == MEDIUM)),

        A.OneOf([
            FancyPCA(alpha_std=6),
            A.RGBShift(r_shift_limit=40, b_shift_limit=30, g_shift_limit=30),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=10),
            A.NoOp()
        ], p=float(augmentation == HARD)),

        # Just flips
        A.Compose([
            A.HorizontalFlip(p=0.5),
        ], p=float(augmentation == LIGHT)),

        A.Compose([
            A.VerticalFlip(p=0.5)
        ], p=float(augmentation == MEDIUM)),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ], p=float(augmentation == HARD)),

        get_preprocessing_transform(preprocessing),
        A.Normalize()
    ])


def get_test_transform(image_size, preprocessing: str = None, crop_black=True):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions() if crop_black else A.NoOp(always_apply=True),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),

        get_preprocessing_transform(preprocessing),
        A.Normalize()
    ])
