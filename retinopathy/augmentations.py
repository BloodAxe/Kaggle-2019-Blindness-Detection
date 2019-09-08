import random
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.functional import brightness_contrast_adjust, elastic_transform

from retinopathy.preprocessing import CropBlackRegions, get_preprocessing_transform, ChannelIndependentCLAHE


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


class BrightnessContrastDestroy(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, **params):
        return 0

    def apply(self, img, alpha=0, beta=0, **params):
        from albumentations.augmentations.functional import brightness_contrast_adjust
        img = brightness_contrast_adjust(img, alpha=alpha, beta=beta)
        return img

    def get_params(self):
        return {'alpha': random.uniform(0.05, 0.25),
                'beta': random.uniform(-0.5, -0.75)}


class MakeTooBlurry(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, **params):
        return 0

    def apply(self, img, blur_ksize=3, **params):
        img = cv2.boxFilter(img, ddepth=cv2.CV_8U, ksize=(blur_ksize, blur_ksize))
        return img

    def get_params(self):
        return {'blur_ksize': int(random.uniform(21, 32)) * 2 + 1}


class MakeTooBlurryMedian(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, **params):
        return 0

    def apply(self, img, blur_ksize=3, **params):
        img = cv2.medianBlur(img, ksize=blur_ksize)
        return img

    def get_params(self):
        return {'blur_ksize': int(random.uniform(8, 16)) * 2 + 1}


class DiagnosisNoise(A.ImageOnlyTransform):
    def __init__(self, p=0.5, sigma=0.02):
        super().__init__(p=p)
        self.sigma = sigma

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, offset=0, **params):
        return float(diagnosis + offset)

    def apply(self, img, **params):
        return img

    def get_params(self):
        return {'offset': random.gauss(0, self.sigma)}


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


def create_microaneurisms(image, center=(256, 256),
                          radius=140,
                          color=(100, 255, 0),
                          num=5,
                          aneurism_radius=(1, 3),
                          alpha=0.2):
    # mask = image.copy()
    aneurism_mask = np.zeros_like(image)
    for i in range(num):
        x = int(random.gauss(center[0], radius))
        y = int(random.gauss(center[0], radius))

        x = min(max(x, center[0] - radius), center[1] + radius)
        y = min(max(y, center[1] - radius), center[1] + radius)

        r = int(random.uniform(aneurism_radius[0], aneurism_radius[1]))
        cv2.circle(aneurism_mask, (x, y), r, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

        # cv2.circle(mask, (x, y), r, (0, 0, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    aneurism_mask = elastic_transform(aneurism_mask, alpha=5, sigma=2, alpha_affine=0)
    cv2.GaussianBlur(aneurism_mask, ksize=(5, 5), sigmaX=0, dst=aneurism_mask)
    aneurism_mask = 1.0 - aneurism_mask / 255.

    overlay = cv2.addWeighted(image, (1 - alpha),
                              image * aneurism_mask, alpha, 0, dtype=cv2.CV_8U)

    return overlay


def create_cotton_wool_mask(size=128):
    iterations = 25
    num_circles = 10
    acc = np.zeros((size, size), dtype=np.float32)

    for i in range(iterations):
        img = np.zeros_like(acc)
        for j in range(num_circles):
            r = int(random.uniform(5, 25))
            r2 = int(random.uniform(5, 25))

            a = random.uniform(0, 180)

            x = int(random.gauss(size // 2, size // 8))
            y = int(random.gauss(size // 2, size // 8))

            pt = int(x), int(y)
            cv2.ellipse(img, pt, (r, r2), a, 0, 360, color=1, thickness=cv2.FILLED)
            # cv2.circle(img, pt, r, color=1, thickness=cv2.FILLED)

        acc += img

    acc /= iterations

    acc = (acc * 255).astype(np.uint8)
    acc_blur = cv2.GaussianBlur(acc, ksize=(9, 9), sigmaX=5, borderType=cv2.BORDER_CONSTANT)
    acc_sharp = cv2.addWeighted(acc, 1.2, acc_blur, -0.2, 0, dtype=cv2.CV_8U)
    acc = acc_sharp

    acc = elastic_transform(acc, alpha=15, sigma=12, alpha_affine=50, border_mode=cv2.BORDER_CONSTANT)

    # cv2.imshow('acc', acc)
    # cv2.waitKey(30)

    return acc


def create_cotton_wool_spots(image, center=(256, 256),
                             radius=140,
                             num=5,
                             spot_radius=List[int],
                             spot_color=List[Tuple],
                             alpha=0.75):
    # mask = image.copy()
    # mask = np.zeros_like(image)

    image2 = image.copy().astype(np.float32)

    for i, sr, sc in zip(range(num), spot_radius, spot_color):
        spot = create_cotton_wool_mask()
        spot = cv2.resize(spot, (2 * sr, 2 * sr))

        repeat = True
        while repeat:
            x = int(random.gauss(center[0], radius))
            y = int(random.gauss(center[1], radius))
            x = max(min(x, center[0] + radius - sr), center[0] - radius + sr)
            y = max(min(y, center[1] + radius - sr), center[1] - radius + sr)
            repeat = bool((x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2)

        blend = (np.expand_dims(spot / 255., -1) * np.array(sc))

        image2[y - sr:y + sr, x - sr:x + sr] += blend
        # mask[y - sr:y + sr, x - sr:x + sr] += np.expand_dims(spot, -1)

    image3 = cv2.addWeighted(image, (1 - alpha),
                             image2, alpha, 0, dtype=cv2.CV_8U)

    return image3


class AddMicroaneurisms(A.ImageOnlyTransform):
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
                                        center=[img.shape[1] // 2, img.shape[0] // 2],
                                        radius=min(img.shape[1], img.shape[0]) // 2 - 10,
                                        num=microaneurisms_count,
                                        aneurism_radius=(1, 4),
                                        alpha=alpha)
        return img

    def update_params(self, params, image=None, diagnosis=0, **kwargs):
        count = int(random.uniform(1, 10))
        new_diag = 2 if count > 6 else 1

        params['apply'] = True
        params['new_diagnosis'] = max(new_diag, diagnosis)
        params['microaneurisms_count'] = count
        params['alpha'] = random.uniform(0.1, 0.3)
        return params

    def get_params(self):
        return {}


class AddCottonWools(A.ImageOnlyTransform):
    def __init__(self,
                 count=(1, 6),
                 radius=(4, 32),
                 color_range=((20, 65, 65), (30, 85, 85)),
                 p=0.5):
        super().__init__(p=p)
        self.count = count
        self.radius = radius
        self.color_range = color_range

    @property
    def targets(self):
        return {'image': self.apply, 'diagnosis': self.apply_to_diagnosis}

    def apply_to_diagnosis(self, diagnosis, apply=False, new_diagnosis=0, **params):
        if apply:
            return new_diagnosis
        return diagnosis

    def apply(self, img, apply=False, new_diagnosis=0, count=0, spot_color=None, spot_radius=None, alpha=0.2, **params):
        if apply:
            img = create_cotton_wool_spots(img,
                                           center=[img.shape[1] // 2, img.shape[0] // 2],
                                           radius=min(img.shape[1], img.shape[0]) // 2 - 10,
                                           num=count,
                                           spot_radius=spot_radius,
                                           spot_color=spot_color,
                                           alpha=alpha)
        return img

    def update_params(self, params, image=None, diagnosis=0, **kwargs):

        count = int(random.uniform(self.count[0], self.count[1]))
        radius = [int(random.uniform(self.radius[0], self.radius[1])) for _ in range(count)]
        color = [(random.uniform(self.color_range[0][0], self.color_range[1][0]),
                  random.uniform(self.color_range[0][1], self.color_range[1][1]),
                  random.uniform(self.color_range[0][2], self.color_range[1][2])) for _ in range(count)]

        params['apply'] = True
        params['new_diagnosis'] = max(1, diagnosis)
        params['count'] = count
        params['spot_radius'] = radius
        params['spot_color'] = color
        params['alpha'] = random.uniform(0.35, 0.55)
        return params

    def get_params(self):
        return {}


def get_none_augmentations(image_size):
    return A.NoOp()


def get_light_augmentations(image_size):
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.85), image_size[0]),
                          height=image_size[0],
                          width=image_size[1], p=0.3),
        ZeroTopAndBottom(p=0.3),
        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25,
                                       contrast_limit=0.2),
            IndependentRandomBrightnessContrast(brightness_limit=0.1,
                                                contrast_limit=0.1),
            A.RandomGamma(gamma_limit=(75, 125)),
            A.NoOp()
        ]),
        A.OneOf([
            ChannelIndependentCLAHE(p=0.5),
            A.CLAHE(),
            A.NoOp()
        ]),
        A.HorizontalFlip(p=0.5),
    ])


def get_medium_augmentations(image_size):
    return A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0),
            A.NoOp()
        ]),
        ZeroTopAndBottom(p=0.3),
        A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                          height=image_size[0],
                          width=image_size[1], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                contrast_limit=0.24),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),
        A.OneOf([
            FancyPCA(alpha_std=4),
            A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=5),
            A.NoOp()
        ]),
        A.OneOf([
            ChannelIndependentCLAHE(p=0.5),
            A.CLAHE(),
            A.NoOp()
        ]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])


def get_hard_augmentations(image_size):
    return A.Compose([
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
        ]),

        A.OneOf([
            ZeroTopAndBottom(p=0.3),

            A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                              height=image_size[0],
                              width=image_size[1], p=0.3),
            A.NoOp()
        ]),

        A.ISONoise(p=0.5),

        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                contrast_limit=0.24),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),

        A.OneOf([
            FancyPCA(alpha_std=6),
            A.RGBShift(r_shift_limit=40, b_shift_limit=30, g_shift_limit=30),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=10),
            A.ToGray(p=0.2),
            A.NoOp()
        ]),

        # Intentionally destroy image quality and assign 0 class in this case
        # A.Compose([
        #     BrightnessContrastDestroy(p=0.1),
        #     A.OneOf([
        #         MakeTooBlurry(),
        #         MakeTooBlurryMedian(),
        #         A.NoOp()
        #     ], p=0.1),
        # ]),

        # Add preprocessing method as an augmentation
        ChannelIndependentCLAHE(p=0.5),

        A.ChannelDropout(),
        A.RandomGridShuffle(p=0.3),
        DiagnosisNoise(p=0.2),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ])
    ])


def get_hard_augmentations_v2(image_size):
    return A.Compose([
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
        ]),

        A.OneOf([
            ZeroTopAndBottom(p=0.3),

            A.RandomSizedCrop(min_max_height=(int(image_size[0] * 0.75), image_size[0]),
                              height=image_size[0],
                              width=image_size[1], p=0.3),
            A.NoOp()
        ]),

        A.ISONoise(p=0.5),
        A.JpegCompression(p=0.3, quality_lower=75),

        # Brightness/contrast augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            IndependentRandomBrightnessContrast(brightness_limit=0.25,
                                                contrast_limit=0.24),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),

        A.OneOf([
            FancyPCA(alpha_std=6),
            A.RGBShift(r_shift_limit=40, b_shift_limit=30, g_shift_limit=30),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=10),
            A.ToGray(p=0.2),
            A.NoOp()
        ]),

        # Intentionally destroy image quality and assign 0 class in this case
        # A.Compose([
        #     BrightnessContrastDestroy(p=0.1),
        #     A.OneOf([
        #         MakeTooBlurry(),
        #         MakeTooBlurryMedian(),
        #         A.NoOp()
        #     ], p=0.1),
        # ]),

        # Add preprocessing method as an augmentation
        ChannelIndependentCLAHE(p=0.5),

        A.OneOf([
            A.ChannelDropout(p=0.2),
            A.CoarseDropout(p=0.1, max_holes=2, max_width=256, max_height=256, min_height=16, min_width=16),
            A.NoOp()
        ]),

        A.RandomGridShuffle(p=0.3),
        DiagnosisNoise(p=0.2),

        # D4
        A.Compose([
            A.RandomRotate90(),
            A.Transpose()
        ])
    ])


def get_train_transform(image_size, augmentation=None, preprocessing=None, crop_black=True):
    if augmentation is None:
        augmentation = 'none'

    artificial = augmentation.endswith('-art')
    if artificial:
        augmentation = augmentation.replace('-art', '')
        print('Using Artifical decease sings generation')

    LEVELS = {
        'none': get_none_augmentations,
        'light': get_light_augmentations,
        'medium': get_medium_augmentations,
        'hard': get_hard_augmentations,
        'hard2': get_hard_augmentations_v2
    }

    assert augmentation in LEVELS.keys()
    augmentation = LEVELS[augmentation](image_size)

    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions(tolerance=5) if crop_black else A.NoOp(always_apply=True),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),

        # Fake decease generation
        A.Compose([
            AddMicroaneurisms(),
            AddCottonWools()
        ], p=float(artificial)),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),

        augmentation,
        get_preprocessing_transform(preprocessing),
        A.Normalize()
    ])


def get_test_transform(image_size, preprocessing: str = None, crop_black=True):
    longest_size = max(image_size[0], image_size[1])
    return A.Compose([
        CropBlackRegions(tolerance=5) if crop_black else A.NoOp(always_apply=True),
        A.LongestMaxSize(longest_size, interpolation=cv2.INTER_CUBIC),

        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),

        get_preprocessing_transform(preprocessing),
        A.Normalize()
    ])
