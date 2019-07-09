import albumentations as A
import cv2
import math
import numpy as np
from pytorch_toolbelt.utils.fs import id_from_fname
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


def get_class_names():
    CLASS_NAMES = [
        'No DR',
        'Mild',
        'Moderate',
        'Severe',
        'Proliferative DR'
    ]
    return CLASS_NAMES


class RetinopathyDataset(Dataset):
    def __init__(self, images, targets, transform: A.Compose, target_as_array=False, dtype=int):
        self.images = np.array(images)
        self.targets = np.array(targets) if targets is not None else None
        self.transform = transform
        self.target_as_array = target_as_array
        self.dtype = dtype

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])  # Read with OpenCV instead PIL. It's faster
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]

        log_height = math.log(height)
        log_width = math.log(width)
        meta_features = np.array([
            log_height,
            log_width
        ])

        image = self.transform(image=image)['image']
        data = {'image': tensor_from_rgb_image(image),
                'image_id': id_from_fname(self.images[item]),
                'meta_features': meta_features}

        if self.targets is not None:
            target = self.dtype(self.targets[item])
            if self.target_as_array:
                data['targets'] = np.array([target])
            else:
                data['targets'] = target

        return data
