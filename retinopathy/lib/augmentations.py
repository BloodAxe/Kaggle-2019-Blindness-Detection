import albumentations as A
import cv2


def crop_black(image, tolerance=10):
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


def contrast_enchance(image, sigmaX=10):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


class CropBlackRegions(A.ImageOnlyTransform):
    def __init__(self, tolerance=15):
        super().__init__(always_apply=True, p=1)
        self.tolerance = tolerance

    def apply(self, img, **params):
        return crop_black(img, self.tolerance)

    def get_transform_init_args_names(self):
        return ('tolerance',)


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
        A.PadIfNeeded(image_size[0], image_size[1],
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize()
    ])
