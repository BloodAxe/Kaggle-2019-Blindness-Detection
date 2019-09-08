from __future__ import absolute_import

import argparse
import collections
import json
import multiprocessing
import os
from datetime import datetime

import torch
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint
from pytorch_toolbelt.utils import fs
from pytorch_toolbelt.utils.random import set_manual_seed, get_random_name
from pytorch_toolbelt.utils.torch_utils import count_parameters, \
    set_trainable

from retinopathy.callbacks import LPRegularizationCallback, \
    CustomOptimizerCallback
from retinopathy.dataset import get_class_names, \
    get_datasets, get_dataloaders
from retinopathy.factory import get_model, get_optimizer, \
    get_optimizable_parameters, get_scheduler, SEResnetEncoder
from retinopathy.scripts.clean_checkpoint import clean_checkpoint
from retinopathy.train_utils import report_checkpoint, get_reg_callbacks, get_ord_callbacks, get_cls_callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--balance-datasets', action='store_true')
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--use-idrid', action='store_true')
    parser.add_argument('--use-messidor', action='store_true')
    parser.add_argument('--use-messidor2-pl1', action='store_true')
    parser.add_argument('--use-aptos2015', action='store_true')
    parser.add_argument('--use-aptos2015-pl1', action='store_true')
    parser.add_argument('--use-aptos2019', action='store_true')
    parser.add_argument('--use-aptos2019-test-pl1', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--coarse', action='store_true')
    parser.add_argument('-acc', '--accumulation-steps', type=int, default=1, help='Number of batches to process')
    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('-m', '--model', type=str, default='resnet18_gap', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Epoch to run')
    parser.add_argument('-es', '--early-stopping', type=int, default=None,
                        help='Maximum number of epochs without improvement')
    parser.add_argument('-f', '--fold', action='append', type=int, default=None)
    parser.add_argument('-ft', '--fine-tune', default=0, type=int)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--criterion-reg', type=str, default=None, nargs='+', help='Criterion')
    parser.add_argument('--criterion-ord', type=str, default=['mse'], nargs='+', help='Criterion')
    parser.add_argument('--criterion-cls', type=str, default=None, nargs='+', help='Criterion')
    parser.add_argument('-l1', type=float, default=0, help='L1 regularization loss')
    parser.add_argument('-l2', type=float, default=0, help='L2 regularization loss')
    parser.add_argument('-o', '--optimizer', default='Adam', help='Name of the optimizer')
    parser.add_argument('-p', '--preprocessing', default=None, help='Preprocessing method')
    parser.add_argument('-c', '--checkpoint', type=str, default=None,
                        help='Checkpoint filename to use as initial model weights')
    parser.add_argument('-w', '--workers', default=multiprocessing.cpu_count(), type=int, help='Num workers')
    parser.add_argument('-a', '--augmentations', default='medium', type=str, help='')
    parser.add_argument('-tta', '--tta', default=None, type=str, help='Type of TTA to use [fliplr, d4]')
    parser.add_argument('-t', '--transfer', default=None, type=str, help='')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-s', '--scheduler', default='multistep', type=str, help='')
    parser.add_argument('--size', default=512, type=int, help='Image size for training & inference')
    parser.add_argument('-wd', '--weight-decay', default=0, type=float, help='L2 weight decay')
    parser.add_argument('-wds', '--weight-decay-step', default=None, type=float,
                        help='L2 weight decay step to add after each epoch')
    parser.add_argument('-d', '--dropout', default=0.0, type=float, help='Dropout before head layer')
    parser.add_argument('--warmup', default=0, type=int,
                        help='Number of warmup epochs with 0.1 of the initial LR and frozed encoder')
    parser.add_argument('-x', '--experiment', default=None, type=str, help='Dropout before head layer')

    args = parser.parse_args()

    data_dir = args.data_dir
    num_workers = args.workers
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    l1 = args.l1
    l2 = args.l2
    early_stopping = args.early_stopping
    model_name = args.model
    optimizer_name = args.optimizer
    image_size = (args.size, args.size)
    fast = args.fast
    augmentations = args.augmentations
    fp16 = args.fp16
    fine_tune = args.fine_tune
    criterion_reg_name = args.criterion_reg
    criterion_cls_name = args.criterion_cls
    criterion_ord_name = args.criterion_ord
    folds = args.fold
    mixup = args.mixup
    balance = args.balance
    balance_datasets = args.balance_datasets
    use_swa = args.swa
    show_batches = args.show
    scheduler_name = args.scheduler
    verbose = args.verbose
    weight_decay = args.weight_decay
    use_idrid = args.use_idrid
    use_messidor = args.use_messidor
    use_aptos2015 = args.use_aptos2015
    use_aptos2019 = args.use_aptos2019
    use_aptos2019_test_pl1 = args.use_aptos2019_test_pl1
    use_aptos2015_pl1 = args.use_aptos2015_pl1
    use_messidor2_pl1 = args.use_messidor2_pl1
    warmup = args.warmup
    dropout = args.dropout
    use_unsupervised = False
    experiment = args.experiment
    preprocessing = args.preprocessing
    weight_decay_step = args.weight_decay_step
    coarse_grading = args.coarse
    class_names = get_class_names(coarse_grading)
    accumulation_steps = args.accumulation_steps

    assert use_aptos2015 or use_aptos2019 or use_idrid or use_messidor

    current_time = datetime.now().strftime('%b%d_%H_%M')
    random_name = get_random_name()

    if folds is None or len(folds) == 0:
        folds = [None]

    for fold in folds:
        torch.cuda.empty_cache()
        checkpoint_prefix = f'{model_name}_{args.size}_{augmentations}'

        if preprocessing is not None:
            checkpoint_prefix += f'_{preprocessing}'
        if use_aptos2019:
            checkpoint_prefix += '_aptos2019'
        if use_aptos2015:
            checkpoint_prefix += '_aptos2015'
        if use_messidor:
            checkpoint_prefix += '_messidor'
        if use_idrid:
            checkpoint_prefix += '_idrid'
        if use_aptos2019_test_pl1 or use_aptos2015_pl1:
            checkpoint_prefix += '_pl1'
        if coarse_grading:
            checkpoint_prefix += '_coarse'

        if fold is not None:
            checkpoint_prefix += f'_fold{fold}'

        checkpoint_prefix += f'_{random_name}'

        if experiment is not None:
            checkpoint_prefix = experiment

        directory_prefix = f'{current_time}/{checkpoint_prefix}'
        log_dir = os.path.join('runs', directory_prefix)
        os.makedirs(log_dir, exist_ok=False)

        config_fname = os.path.join(log_dir, f'{checkpoint_prefix}.json')
        with open(config_fname, 'w') as f:
            train_session_args = vars(args)
            f.write(json.dumps(train_session_args, indent=2))

        set_manual_seed(args.seed)
        num_classes = len(class_names)
        model = get_model(model_name, num_classes=num_classes, dropout=dropout)

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

            report_checkpoint(checkpoint)
            del checkpoint

        if args.checkpoint:
            checkpoint = load_checkpoint(fs.auto_file(args.checkpoint))
            unpack_checkpoint(checkpoint, model=model)
            report_checkpoint(checkpoint)
            del checkpoint

        model = model.cuda()

        train_ds, valid_ds, train_sizes = get_datasets(data_dir=data_dir,
                                                       use_aptos2019=use_aptos2019,
                                                       use_aptos2019_test_pl1=use_aptos2019_test_pl1,
                                                       use_aptos2015_pl1=use_aptos2015_pl1,
                                                       use_aptos2015=use_aptos2015,
                                                       use_idrid=use_idrid,
                                                       use_messidor=use_messidor,
                                                       use_messidor2_pl1=use_messidor2_pl1,
                                                       use_unsupervised=False,
                                                       coarse_grading=coarse_grading,
                                                       image_size=image_size,
                                                       augmentation=augmentations,
                                                       preprocessing=preprocessing,
                                                       target_dtype=int,
                                                       fold=fold,
                                                       folds=4)

        train_loader, valid_loader = get_dataloaders(train_ds, valid_ds,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     train_sizes=train_sizes,
                                                     balance=balance,
                                                     balance_datasets=balance_datasets,
                                                     balance_unlabeled=False)

        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

        print('Datasets         :', data_dir)
        print('  Train size     :', len(train_loader), len(train_loader.dataset))
        print('  Valid size     :', len(valid_loader), len(valid_loader.dataset))
        print('  Aptos 2019     :', use_aptos2019)
        print('  Aptos 2019 PL1 :', use_aptos2019_test_pl1)
        print('  Aptos 2015     :', use_aptos2015)
        print('  IDRID          :', use_idrid)
        print('  Messidor       :', use_messidor)
        print('Train session    :', directory_prefix)
        print('  FP16 mode      :', fp16)
        print('  Fast mode      :', fast)
        print('  Mixup          :', mixup)
        print('  Balance cls.   :', balance)
        print('  Balance ds.    :', balance_datasets)
        print('  Warmup epoch   :', warmup)
        print('  Train epochs   :', num_epochs)
        print('  Fine-tune ephs :', fine_tune)
        print('  Workers        :', num_workers)
        print('  Fold           :', fold)
        print('  Log dir        :', log_dir)
        print('  Augmentations  :', augmentations)
        print('Model            :', model_name)
        print('  Parameters     :', count_parameters(model))
        print('  Image size     :', image_size)
        print('  Dropout        :', dropout)
        print('  Classes        :', class_names, num_classes)
        print('Optimizer        :', optimizer_name)
        print('  Learning rate  :', learning_rate)
        print('  Batch size     :', batch_size)
        print('  Criterion (cls):', criterion_cls_name)
        print('  Criterion (reg):', criterion_reg_name)
        print('  Criterion (ord):', criterion_ord_name)
        print('  Scheduler      :', scheduler_name)
        print('  Weight decay   :', weight_decay, weight_decay_step)
        print('  L1 reg.        :', l1)
        print('  L2 reg.        :', l2)
        print('  Early stopping :', early_stopping)

        # model training
        callbacks = []
        criterions = {}

        main_metric = 'ord/kappa'
        if criterion_reg_name is not None:
            cb, crits = get_reg_callbacks(criterion_reg_name, class_names=class_names, show=show_batches)
            callbacks += cb
            criterions.update(crits)

        if criterion_ord_name is not None:
            cb, crits = get_ord_callbacks(criterion_ord_name, class_names=class_names, show=show_batches)
            callbacks += cb
            criterions.update(crits)

        if criterion_cls_name is not None:
            cb, crits = get_cls_callbacks(criterion_cls_name,
                                          num_classes=num_classes,
                                          num_epochs=num_epochs, class_names=class_names, show=show_batches)
            callbacks += cb
            criterions.update(crits)

        if l1 > 0:
            callbacks += [LPRegularizationCallback(start_wd=l1, end_wd=l1, schedule=None, prefix='l1', p=1)]

        if l2 > 0:
            callbacks += [LPRegularizationCallback(start_wd=l2, end_wd=l2, schedule=None, prefix='l2', p=2)]

        callbacks += [
            CustomOptimizerCallback(accumulation_steps=accumulation_steps)
        ]

        runner = SupervisedRunner(input_key='image')

        # Pretrain/warmup
        if warmup:
            set_trainable(model.encoder, False, False)
            optimizer = get_optimizer(optimizer_name, get_optimizable_parameters(model),
                                      learning_rate=learning_rate)

            runner.train(
                fp16=fp16,
                model=model,
                criterion=criterions,
                optimizer=optimizer,
                scheduler=None,
                callbacks=callbacks,
                loaders=loaders,
                logdir=os.path.join(log_dir, 'warmup'),
                num_epochs=warmup,
                verbose=verbose,
                main_metric=main_metric,
                minimize_metric=False,
                checkpoint_data={"cmd_args": vars(args)}
            )

            del optimizer

        # Main train
        if num_epochs:
            set_trainable(model.encoder, True, False)

            optimizer = get_optimizer(optimizer_name, get_optimizable_parameters(model),
                                      learning_rate=learning_rate,
                                      weight_decay=weight_decay)

            if use_swa:
                from torchcontrib.optim import SWA
                optimizer = SWA(optimizer,
                                swa_start=len(train_loader),
                                swa_freq=512)

            scheduler = get_scheduler(scheduler_name, optimizer,
                                      lr=learning_rate,
                                      num_epochs=num_epochs,
                                      batches_in_epoch=len(train_loader))

            # Additional callbacks that specific to main stage only added here to copy of callbacks
            main_stage_callbacks = callbacks
            if early_stopping:
                es_callback = EarlyStoppingCallback(early_stopping,
                                                    min_delta=1e-4,
                                                    metric=main_metric, minimize=False)
                main_stage_callbacks = callbacks + [es_callback]

            runner.train(
                fp16=fp16,
                model=model,
                criterion=criterions,
                optimizer=optimizer,
                scheduler=scheduler,
                callbacks=main_stage_callbacks,
                loaders=loaders,
                logdir=os.path.join(log_dir, 'main'),
                num_epochs=num_epochs,
                verbose=verbose,
                main_metric=main_metric,
                minimize_metric=False,
                checkpoint_data={"cmd_args": vars(args)}
            )

            del optimizer, scheduler

            best_checkpoint = os.path.join(log_dir, 'main', 'checkpoints', 'best.pth')
            model_checkpoint = os.path.join(log_dir, 'main', 'checkpoints', f'{checkpoint_prefix}.pth')
            clean_checkpoint(best_checkpoint, model_checkpoint)

            # Restoring best model from checkpoint
            checkpoint = load_checkpoint(best_checkpoint)
            unpack_checkpoint(checkpoint, model=model)
            report_checkpoint(checkpoint)
            del checkpoint

        # Stage 3 - Fine tuning
        if fine_tune:
            del train_loader, valid_loader, loaders
            train_loader, valid_loader = get_dataloaders(train_ds, valid_ds,
                                                         batch_size=batch_size,
                                                         num_workers=num_workers,
                                                         train_sizes=train_sizes,
                                                         balance=balance,
                                                         balance_datasets=balance_datasets,
                                                         balance_unlabeled=False)

            loaders = collections.OrderedDict()
            loaders["train"] = train_loader
            loaders["valid"] = valid_loader

            set_trainable(model, True, False)
            set_trainable(model.encoder, False, False)
            if isinstance(model.encoder, SEResnetEncoder):
                # Allow training last layer
                set_trainable(model.encoder.layer4, True, False)

            optimizer = get_optimizer(optimizer_name, get_optimizable_parameters(model),
                                      learning_rate=learning_rate)
            scheduler = get_scheduler('multistep', optimizer,
                                      lr=learning_rate,
                                      num_epochs=fine_tune,
                                      batches_in_epoch=len(train_loader))

            runner.train(
                fp16=fp16,
                model=model,
                criterion=criterions,
                optimizer=optimizer,
                scheduler=scheduler,
                callbacks=callbacks,
                loaders=loaders,
                logdir=os.path.join(log_dir, 'finetune'),
                num_epochs=fine_tune,
                verbose=verbose,
                main_metric=main_metric,
                minimize_metric=False,
                checkpoint_data={"cmd_args": vars(args)}
            )

            best_checkpoint = os.path.join(log_dir, 'finetune', 'checkpoints', 'best.pth')
            model_checkpoint = os.path.join(log_dir, 'finetune', 'checkpoints', f'{checkpoint_prefix}.pth')
            clean_checkpoint(best_checkpoint, model_checkpoint)

        del callbacks, runner, criterions, loaders


if __name__ == '__main__':
    with torch.autograd.detect_anomaly():
        main()
