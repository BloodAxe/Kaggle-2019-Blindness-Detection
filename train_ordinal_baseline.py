from __future__ import absolute_import

import argparse
import collections
import multiprocessing
import os
from datetime import datetime
from functools import partial

from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, AccuracyCallback
from catalyst.dl.callbacks import MixupCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.catalyst import ShowPolarBatchesCallback, ConfusionMatrixCallback
from pytorch_toolbelt.utils.random import set_manual_seed
from pytorch_toolbelt.utils.torch_utils import maybe_cuda, count_parameters, \
    set_trainable

from retinopathy.lib.callbacks import AscensionCallback, CappaScoreCallback
from retinopathy.lib.dataset import get_class_names, \
    get_datasets, get_dataloaders
from retinopathy.lib.factory import get_model, get_loss, get_optimizer, \
    get_optimizable_parameters, get_scheduler
from retinopathy.lib.visualization import draw_classification_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-acc', '--accumulation-steps', type=int, default=1, help='Number of batches to process')
    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('-m', '--model', type=str, default='ord_resnext50_gap', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Epoch to run')
    parser.add_argument('-es', '--early-stopping', type=int, default=None,
                        help='Maximum number of epochs without improvement')
    parser.add_argument('-f', '--fold', action='append', type=int, default=None)
    parser.add_argument('-fe', '--freeze-encoder', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-l', '--criterion', type=str, default='link', help='Criterion')
    parser.add_argument('-o', '--optimizer', default='Adam', help='Name of the optimizer')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Checkpoint filename to use as initial model weights')
    parser.add_argument('-w', '--workers', default=multiprocessing.cpu_count(), type=int, help='Num workers')
    parser.add_argument('-a', '--augmentations', default='medium', type=str, help='')
    parser.add_argument('-tta', '--tta', default=None, type=str, help='Type of TTA to use [fliplr, d4]')
    parser.add_argument('--transfer', default=None, type=str, help='')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-s', '--scheduler', default='multistep', type=str, help='')
    parser.add_argument('-wd', '--weight-decay', default=0, type=float, help='L2 weight decay')

    args = parser.parse_args()

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
    fp16 = args.fp16
    freeze_encoder = args.freeze_encoder
    criterion_name = args.criterion
    folds = args.fold
    mixup = args.mixup
    balance = args.balance
    use_swa = args.swa
    show_batches = args.show
    scheduler_name = args.scheduler
    verbose = args.verbose
    weight_decay = args.weight_decay

    if folds is None or len(folds) == 0:
        folds = [None]

    for fold in folds:

        set_manual_seed(args.seed)
        model = maybe_cuda(
            get_model(model_name, num_classes=len(get_class_names())))

        if args.transfer:
            transfer_checkpoint = fs.auto_file(args.transfer)
            print("Transfering weights from model checkpoint",
                  transfer_checkpoint)
            checkpoint = load_checkpoint(transfer_checkpoint)
            pretrained_dict = checkpoint['model_state_dict']

            for name, value in pretrained_dict.items():
                try:
                    model.load_state_dict(
                        collections.OrderedDict([(name, value)]), strict=False)
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
                  'cappa:',
                  checkpoint['epoch_metrics']['train']['kappa_score'],
                  'accuracy:',
                  checkpoint['epoch_metrics']['train']['accuracy'],
                  'loss:', checkpoint['epoch_metrics']['train']['loss'])
            print('Metrics (Valid):',
                  'cappa:',
                  checkpoint['epoch_metrics']['valid']['kappa_score'],
                  'accuracy:',
                  checkpoint['epoch_metrics']['valid']['accuracy'],
                  'loss:', checkpoint['epoch_metrics']['valid']['loss'])

        if freeze_encoder:
            set_trainable(model.encoder, trainable=False, freeze_bn=True)

        criterion = get_loss(criterion_name)
        parameters = get_optimizable_parameters(model)
        optimizer = get_optimizer(optimizer_name, parameters,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay)

        if checkpoint is not None:
            try:
                unpack_checkpoint(checkpoint, optimizer=optimizer)
                print('Restored optimizer state from checkpoint')
            except Exception as e:
                print('Failed to restore optimizer state from checkpoint', e)

        train_ds, valid_ds = get_datasets(data_dir=data_dir,
                                          use_aptos2015=not fast,
                                          use_aptos2019=True,
                                          use_idrid=not fast,
                                          use_messidor=not fast,
                                          image_size=image_size,
                                          augmentation=augmentations,
                                          target_dtype=int,
                                          fold=fold,
                                          folds=4)

        train_loader, valid_loader = get_dataloaders(train_ds, valid_ds,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     oversample_factor=2 if fast else 1,
                                                     balance=balance)

        if use_swa:
            from torchcontrib.optim import SWA
            optimizer = SWA(optimizer,
                            swa_start=len(train_loader),
                            swa_freq=512)

        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

        current_time = datetime.now().strftime('%b%d_%H_%M')
        prefix = f'ordinal/{model_name}/fold_{fold}/{current_time}_{criterion_name}'

        if fp16:
            prefix += '_fp16'

        if fast:
            prefix += '_fast'

        log_dir = os.path.join('runs', prefix)
        os.makedirs(log_dir, exist_ok=False)

        scheduler = get_scheduler(scheduler_name, optimizer,
                                  lr=learning_rate,
                                  num_epochs=num_epochs,
                                  batches_in_epoch=len(train_loader))

        print('Train session    :', prefix)
        print('\tFP16 mode      :', fp16)
        print('\tFast mode      :', fast)
        print('\tMixup          :', mixup)
        print('\tBalance        :', balance)
        print('\tEpochs         :', num_epochs)
        print('\tEarly stopping :', early_stopping)
        print('\tWorkers        :', num_workers)
        print('\tData dir       :', data_dir)
        print('\tFold           :', fold)
        print('\tLog dir        :', log_dir)
        print('\tAugmentations  :', augmentations)
        print('\tTrain size     :', len(train_loader),
              len(train_loader.dataset))
        print('\tValid size     :', len(valid_loader),
              len(valid_loader.dataset))
        print('Model            :', model_name)
        print('\tParameters     :', count_parameters(model))
        print('\tImage size     :', image_size)
        print('\tFreeze encoder :', freeze_encoder)
        print('Optimizer        :', optimizer_name)
        print('\tLearning rate  :', learning_rate)
        print('\tBatch size     :', batch_size)
        print('\tCriterion      :', criterion_name)
        print('\tScheduler      :', scheduler_name)

        # model training
        visualization_fn = partial(draw_classification_predictions,
                                   class_names=get_class_names())

        callbacks = [
            AccuracyCallback(),
            CappaScoreCallback(),
            ConfusionMatrixCallback(class_names=get_class_names()),
            AscensionCallback(model)
        ]

        if mixup:
            callbacks += [MixupCallback(fields=['image'])]

        if early_stopping:
            callbacks += [
                EarlyStoppingCallback(early_stopping,
                                      metric='kappa_score', minimize=False)]

        if show_batches:
            callbacks += [
                ShowPolarBatchesCallback(visualization_fn,
                                         metric='accuracy', minimize=False)]

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
            verbose=verbose,
            main_metric='kappa_score',
            minimize_metric=False,
            checkpoint_data={"cmd_args": vars(args)}
        )

        del runner, callbacks, loaders, optimizer, model, criterion, scheduler

        # if fold is not None:
        #     dataset_fname = os.path.join(data_dir, 'train_with_folds.csv')
        #     dataset = pd.read_csv(dataset_fname)
        #     oof_csv = dataset[dataset['fold'] == fold]
        #
        #     model_checkpoint = os.path.join(log_dir, 'checkpoints', 'best.pth')
        #     oof_predictions = run_model_inference(
        #         model_checkpoint=model_checkpoint,
        #         test_csv=oof_csv,
        #         images_dir='train_images',
        #         data_dir=data_dir,
        #         batch_size=batch_size,
        #         tta=None,
        #         apply_softmax=False)
        #
        #     checkpoint = load_checkpoint(model_checkpoint)
        #     del checkpoint['criterion_state_dict']
        #     del checkpoint['optimizer_state_dict']
        #     del checkpoint['scheduler_state_dict']
        #     checkpoint['oof_predictions'] = oof_predictions.to_dict()
        #     torch.save(checkpoint, os.path.join(log_dir, 'checkpoints',
        #                                         f'{model_name}_fold{fold}' + '.pth'))


if __name__ == '__main__':
    main()
