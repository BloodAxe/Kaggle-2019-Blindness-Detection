import argparse
import os

import pandas as pd
import torch
from pytorch_toolbelt.utils import fs

from retinopathy.dataset import get_datasets
from retinopathy.inference import run_model_inference_via_dataset, run_model_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--need-features', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=None, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-w', '--workers', type=int, default=4, help='')

    args = parser.parse_args()

    need_features = args.need_features
    batch_size = args.batch_size
    num_workers = args.workers

    checkpoints = args.input
    for i, checkpoint_fname in enumerate(checkpoints):
        print(i, checkpoint_fname)

        # Make OOF predictions
        checkpoint = torch.load(checkpoint_fname)
        params = checkpoint['checkpoint_data']['cmd_args']
        image_size = params['size']
        data_dir = params['data_dir']

        # train_ds, valid_ds, train_sizes = get_datasets(data_dir=params['data_dir'],
        #                                                use_aptos2019=params['use_aptos2019'],
        #                                                use_aptos2015=params['use_aptos2015'],
        #                                                use_idrid=params['use_idrid'],
        #                                                use_messidor=params['use_messidor'],
        #                                                use_unsupervised=False,
        #                                                image_size=(image_size, image_size),
        #                                                augmentation=params['augmentations'],
        #                                                preprocessing=params['preprocessing'],
        #                                                target_dtype=int,
        #                                                coarse_grading=params.get('coarse', False),
        #                                                fold=i,
        #                                                folds=4)
        # print(len(valid_ds))
        # oof_predictions = run_model_inference_via_dataset(checkpoint_fname,
        #                                                   valid_ds,
        #                                                   apply_softmax=True,
        #                                                   need_features=need_features,
        #                                                   batch_size=batch_size,
        #                                                   workers=num_workers)

        # dst_fname = fs.change_extension(checkpoint_fname, '_oof_predictions.pkl')
        # oof_predictions.to_pickle(dst_fname)

        # Now run inference on holdout IDRID Test dataset
        idrid_test = run_model_inference(model_checkpoint=checkpoint_fname,
                                         apply_softmax=True,
                                         need_features=need_features,
                                         test_csv=pd.read_csv(os.path.join(data_dir, 'idrid', 'test_labels.csv')),
                                         data_dir=os.path.join(data_dir, 'idrid'),
                                         images_dir='test_images_768',
                                         batch_size=batch_size,
                                         tta='fliplr',
                                         workers=num_workers,
                                         crop_black=True)
        idrid_test.to_pickle(fs.change_extension(checkpoint_fname, '_idrid_test_predictions.pkl'))

        # Now run inference on Messidor 2 Test dataset
        messidor2_train = run_model_inference(model_checkpoint=checkpoint_fname,
                                              apply_softmax=True,
                                              need_features=need_features,
                                              test_csv=pd.read_csv(os.path.join(data_dir, 'messidor_2', 'train_labels.csv')),
                                              data_dir=os.path.join(data_dir, 'messidor_2'),
                                              images_dir='train_images_768',
                                              batch_size=batch_size,
                                              tta='fliplr',
                                              workers=num_workers,
                                              crop_black=True)
        messidor2_train.to_pickle(fs.change_extension(checkpoint_fname, '_messidor2_train_predictions.pkl'))

        # Now run inference on Aptos2019 public test
        aptos2019_test = run_model_inference(model_checkpoint=checkpoint_fname,
                                             apply_softmax=True,
                                             need_features=need_features,
                                             test_csv=pd.read_csv(os.path.join(data_dir, 'aptos-2019', 'test.csv')),
                                             data_dir=os.path.join(data_dir, 'aptos-2019'),
                                             images_dir='test_images_768',
                                             batch_size=batch_size,
                                             tta='fliplr',
                                             workers=num_workers,
                                             crop_black=True)
        aptos2019_test.to_pickle(fs.change_extension(checkpoint_fname, '_aptos2019_test_predictions.pkl'))

        # Now run inference on Aptos2015 private test
        if True:
            aptos2015_df = pd.read_csv(os.path.join(data_dir, 'aptos-2015', 'test_labels.csv'))
            aptos2015_df = aptos2015_df[aptos2015_df['Usage'] == 'Private']
            aptos2015_test = run_model_inference(model_checkpoint=checkpoint_fname,
                                                 apply_softmax=True,
                                                 need_features=need_features,
                                                 test_csv=aptos2015_df,
                                                 data_dir=os.path.join(data_dir, 'aptos-2015'),
                                                 images_dir='test_images_768',
                                                 batch_size=batch_size,
                                                 tta='fliplr',
                                                 workers=num_workers,
                                                 crop_black=True)
            aptos2015_test.to_pickle(fs.change_extension(checkpoint_fname, '_aptos2015_test_private_predictions.pkl'))

        if False:
            aptos2015_df = pd.read_csv(os.path.join(data_dir, 'aptos-2015', 'test_labels.csv'))
            aptos2015_df = aptos2015_df[aptos2015_df['Usage'] == 'Public']
            aptos2015_test = run_model_inference(model_checkpoint=checkpoint_fname,
                                                 apply_softmax=True,
                                                 need_features=need_features,
                                                 test_csv=aptos2015_df,
                                                 data_dir=os.path.join(data_dir, 'aptos-2015'),
                                                 images_dir='test_images_768',
                                                 batch_size=batch_size,
                                                 tta='fliplr',
                                                 workers=num_workers,
                                                 crop_black=True)
            aptos2015_test.to_pickle(fs.change_extension(checkpoint_fname, '_aptos2015_test_public_predictions.pkl'))

        if False:
            aptos2015_df = pd.read_csv(os.path.join(data_dir, 'aptos-2015', 'train_labels.csv'))
            aptos2015_test = run_model_inference(model_checkpoint=checkpoint_fname,
                                                 apply_softmax=True,
                                                 need_features=need_features,
                                                 test_csv=aptos2015_df,
                                                 data_dir=os.path.join(data_dir, 'aptos-2015'),
                                                 images_dir='train_images_768',
                                                 batch_size=batch_size,
                                                 tta='fliplr',
                                                 workers=num_workers,
                                                 crop_black=True)
            aptos2015_test.to_pickle(fs.change_extension(checkpoint_fname, '_aptos2015_train_predictions.pkl'))


if __name__ == '__main__':
    main()
