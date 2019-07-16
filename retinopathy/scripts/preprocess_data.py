import os
from functools import partial
from multiprocessing.pool import Pool

import cv2
from albumentations.augmentations.functional import longest_max_size
from pytorch_toolbelt.utils import fs
from tqdm import tqdm

from retinopathy.lib.augmentations import crop_black


def preprocess(image_fname, output_dir):
    image = cv2.imread(image_fname)
    image = crop_black(image, tolerance=5)
    image = longest_max_size(image, max_size=768, interpolation=cv2.INTER_CUBIC)

    image_id = fs.id_from_fname(image_fname)
    dst_fname = os.path.join(output_dir, image_id + '.png')
    cv2.imwrite(dst_fname, image)
    return


def convert_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = fs.find_images_in_dir(input_dir)

    processing_fn = partial(preprocess, output_dir=output_dir)

    with Pool(32) as wp:
        for image_id in tqdm(wp.imap_unordered(processing_fn, images), total=len(images)):
            pass


def main():
    convert_dir('data/aptos-2019/train_images', 'data/aptos-2019/train_images_768')
    convert_dir('data/aptos-2019/test_images', 'data/aptos-2019/test_images_768')

    convert_dir('data/aptos-2015/train_images', 'data/aptos-2015/train_images_768')
    convert_dir('data/aptos-2015/test_images', 'data/aptos-2015/test_images_768')

    convert_dir('data/idrid/train_images', 'data/idrid/train_images_768')
    convert_dir('data/idrid/test_images', 'data/idrid/test_images_768')


if __name__ == '__main__':
    main()
