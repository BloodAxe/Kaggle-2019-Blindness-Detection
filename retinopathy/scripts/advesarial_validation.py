from __future__ import absolute_import

import argparse
import collections
import multiprocessing
import os
from datetime import datetime
from functools import partial

import pandas as pd
import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, AUCCallback
from catalyst.dl.callbacks import F1ScoreCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import ShowPolarBatchesCallback
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters, to_numpy, set_trainable
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from retinopathy.dataset import RetinopathyDataset
from retinopathy.factory import get_model, get_loss, get_optimizer, get_optimizable_parameters, get_train_aug, get_test_aug
from retinopathy.visualization import draw_classification_predictions


def get_dataloaders(data_dir, batch_size, num_workers,
                    image_size, augmentation, fast):
    train_csv = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train_csv['id_code'] = train_csv['id_code'].apply(lambda x: os.path.join(data_dir, 'train_images', f'{x}.png'))
    train_csv['is_test'] = 0

    test_csv = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    test_csv['id_code'] = test_csv['id_code'].apply(lambda x: os.path.join(data_dir, 'test_images', f'{x}.png'))
    test_csv['is_test'] = 1

    dataset = pd.concat((train_csv[['id_code', 'is_test']], test_csv[['id_code', 'is_test']]))
    x = dataset['id_code']
    y = dataset['is_test']

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, random_state=42, test_size=0.1, shuffle=True, stratify=y)
    if fast:
        train_x = train_x[:32]
        train_y = train_y[:32]

        valid_x = valid_x[:32]
        valid_y = valid_y[:32]

        num_workers = 0

    train_ds = RetinopathyDataset(train_x, train_y, transform=get_train_aug(image_size, augmentation), target_as_array=True)
    valid_ds = RetinopathyDataset(valid_x, valid_y, transform=get_test_aug(image_size), target_as_array=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=num_workers)

    return train_dl, valid_dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Data directory for INRIA sattelite dataset')
    parser.add_argument('-m', '--model', type=str, default='cls_resnet18', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Epoch to run')
    parser.add_argument('-es', '--early-stopping', type=int, default=None, help='Maximum number of epochs without improvement')
    parser.add_argument('-fe', '--freeze-encoder', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-l', '--criterion', type=str, default='bce', help='Criterion')
    parser.add_argument('-o', '--optimizer', default='Adam', help='Name of the optimizer')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Checkpoint filename to use as initial model weights')
    parser.add_argument('-w', '--workers', default=multiprocessing.cpu_count(), type=int, help='Num workers')
    parser.add_argument('-a', '--augmentations', default='hard', type=str, help='')
    parser.add_argument('-tta', '--tta', default=None, type=str, help='Type of TTA to use [fliplr, d4]')
    parser.add_argument('-tm', '--train-mode', default='random', type=str, help='')
    parser.add_argument('-rm', '--run-mode', default='fit_predict', type=str, help='')
    parser.add_argument('--transfer', default=None, type=str, help='')
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()
    set_manual_seed(args.seed)

    data_dir = args.data_dir
    num_workers = args.workers
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    early_stopping = args.early_stopping
    model_name = args.model
    optimizer_name = args.optimizer
    image_size = (512, 512)
    fast = args.fast
    augmentations = args.augmentations
    train_mode = args.train_mode
    run_mode = args.run_mode
    log_dir = None
    fp16 = args.fp16
    freeze_encoder = args.freeze_encoder

    run_train = run_mode == 'fit_predict' or run_mode == 'fit'
    run_predict = run_mode == 'fit_predict' or run_mode == 'predict'

    model = maybe_cuda(get_model(model_name, num_classes=1))

    if args.transfer:
        transfer_checkpoint = fs.auto_file(args.transfer)
        print("Transfering weights from model checkpoint", transfer_checkpoint)
        checkpoint = load_checkpoint(transfer_checkpoint)
        pretrained_dict = checkpoint['model_state_dict']

        for name, value in pretrained_dict.items():
            try:
                model.load_state_dict(collections.OrderedDict([(name, value)]), strict=False)
            except Exception as e:
                print(e)

    checkpoint = None
    if args.checkpoint:
        checkpoint = load_checkpoint(fs.auto_file(args.checkpoint))
        unpack_checkpoint(checkpoint, model=model)

        checkpoint_epoch = checkpoint['epoch']
        print('Loaded model weights from:', args.checkpoint)
        print('Epoch                    :', checkpoint_epoch)
        print('Metrics (Train):',
              'f1  :', checkpoint['epoch_metrics']['train']['f1_score'],
              'loss:', checkpoint['epoch_metrics']['train']['loss'])
        print('Metrics (Valid):',
              'f1  :', checkpoint['epoch_metrics']['valid']['f1_score'],
              'loss:', checkpoint['epoch_metrics']['valid']['loss'])

        log_dir = os.path.dirname(os.path.dirname(fs.auto_file(args.checkpoint)))

    if run_train:

        if freeze_encoder:
            set_trainable(model.encoder, trainable=False, freeze_bn=True)

        criterion = get_loss(args.criterion)
        parameters = get_optimizable_parameters(model)
        optimizer = get_optimizer(optimizer_name, parameters, learning_rate)

        if checkpoint is not None:
            try:
                unpack_checkpoint(checkpoint, optimizer=optimizer)
                print('Restored optimizer state from checkpoint')
            except Exception as e:
                print('Failed to restore optimizer state from checkpoint', e)

        train_loader, valid_loader = get_dataloaders(data_dir=data_dir,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     image_size=image_size,
                                                     augmentation=augmentations,
                                                     fast=fast)

        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

        current_time = datetime.now().strftime('%b%d_%H_%M')
        prefix = f'adversarial/{args.model}/{current_time}_{args.criterion}'

        if fp16:
            prefix += '_fp16'

        if fast:
            prefix += '_fast'

        log_dir = os.path.join('runs', prefix)
        os.makedirs(log_dir, exist_ok=False)

        scheduler = MultiStepLR(optimizer,
                                milestones=[10, 30, 50, 70, 90], gamma=0.5)

        print('Train session    :', prefix)
        print('\tFP16 mode      :', fp16)
        print('\tFast mode      :', args.fast)
        print('\tTrain mode     :', train_mode)
        print('\tEpochs         :', num_epochs)
        print('\tEarly stopping :', early_stopping)
        print('\tWorkers        :', num_workers)
        print('\tData dir       :', data_dir)
        print('\tLog dir        :', log_dir)
        print('\tAugmentations  :', augmentations)
        print('\tTrain size     :', len(train_loader), len(train_loader.dataset))
        print('\tValid size     :', len(valid_loader), len(valid_loader.dataset))
        print('Model            :', model_name)
        print('\tParameters     :', count_parameters(model))
        print('\tImage size     :', image_size)
        print('\tFreeze encoder :', freeze_encoder)
        print('Optimizer        :', optimizer_name)
        print('\tLearning rate  :', learning_rate)
        print('\tBatch size     :', batch_size)
        print('\tCriterion      :', args.criterion)

        # model training
        visualization_fn = partial(draw_classification_predictions, class_names=['Train', 'Test'])

        callbacks = [
            F1ScoreCallback(),
            AUCCallback(),
            ShowPolarBatchesCallback(visualization_fn, metric='f1_score', minimize=False),
        ]

        if early_stopping:
            callbacks += [EarlyStoppingCallback(early_stopping, metric='auc', minimize=False)]

        runner = SupervisedRunner(input_key='image')
        runner.train(
            fp16=fp16,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loaders=loaders,
            logdir=log_dir,
            num_epochs=num_epochs,
            verbose=True,
            main_metric='auc',
            minimize_metric=False,
            state_kwargs={"cmd_args": vars(args)}
        )

    if run_predict and not fast:
        # Training is finished. Let's run predictions using best checkpoint weights
        best_checkpoint = load_checkpoint(fs.auto_file('best.pth', where=log_dir))
        unpack_checkpoint(best_checkpoint, model=model)

        model.eval()
        torch.no_grad()

        train_csv = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        train_csv['id_code'] = train_csv['id_code'].apply(lambda x: os.path.join(data_dir, 'train_images', f'{x}.png'))
        test_ds = RetinopathyDataset(train_csv['id_code'], None, get_test_aug(image_size), target_as_array=True)
        test_dl = DataLoader(test_ds, batch_size, pin_memory=True, num_workers=num_workers)

        test_ids = []
        test_preds = []

        for batch in tqdm(test_dl, desc='Inference'):
            input = batch['image'].cuda()
            outputs = model(input)
            predictions = to_numpy(outputs['logits'].sigmoid().squeeze(1))
            test_ids.extend(batch['image_id'])
            test_preds.extend(predictions)

        df = pd.DataFrame.from_dict({'id_code': test_ids, 'is_test': test_preds})
        df.to_csv(os.path.join(log_dir, 'test_in_train.csv'), index=None)


if __name__ == '__main__':
    main()
