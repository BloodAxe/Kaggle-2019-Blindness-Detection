from functools import partial

from catalyst.dl import CriterionCallback
from pytorch_toolbelt.utils.catalyst import ConfusionMatrixCallback, ShowPolarBatchesCallback

from retinopathy.callbacks import CappaScoreCallback, CustomAccuracyCallback, NegativeMiningCallback, \
    TSACriterionCallback, UDACriterionCallback, UDARegressionCriterionCallback, ConfusionMatrixCallbackFromRegression, \
    FScoreCallback, RMSEMetric
from retinopathy.dataset import UNLABELED_CLASS
from retinopathy.factory import get_loss
from retinopathy.visualization import draw_regression_predictions, draw_classification_predictions


def report_checkpoint(checkpoint):
    print('Epoch          :', checkpoint['epoch'])
    print('Metrics (Train):', checkpoint['epoch_metrics']['train'])
    print('Metrics (Valid):', checkpoint['epoch_metrics']['valid'])


def get_cls_callbacks(loss_name, num_classes, num_epochs, class_names, tsa=None, uda=None, show=False):
    if len(loss_name) == 1:
        loss_name, multiplier = loss_name[0], 1.0
    elif len(loss_name) == 2:
        loss_name, multiplier = loss_name[0], float(loss_name[1])
    else:
        raise ValueError(loss_name)

    criterions = {'cls': get_loss(loss_name, ignore_index=UNLABELED_CLASS)}
    output_key = 'logits'

    if tsa:
        crit_callback = TSACriterionCallback(prefix='cls/tsa_loss', loss_key='cls',
                                             output_key=output_key,
                                             criterion_key='cls',
                                             multiplier=multiplier,
                                             num_classes=num_classes,
                                             num_epochs=num_epochs)
    else:
        crit_callback = CriterionCallback(prefix='cls/loss', loss_key='cls',
                                          output_key=output_key,
                                          criterion_key='cls',
                                          multiplier=multiplier)

    callbacks = [
        crit_callback,
        CappaScoreCallback(prefix='cls/kappa',
                           output_key=output_key,
                           ignore_index=UNLABELED_CLASS,
                           class_names=class_names),
        # Metrics
        CustomAccuracyCallback(output_key=output_key,
                               prefix='cls/accuracy',
                               ignore_index=UNLABELED_CLASS),
        # F1 scores
        FScoreCallback(
            prefix='cls/f1_macro',
            beta=1,
            average='macro',
            output_key=output_key,
            ignore_index=UNLABELED_CLASS),
        FScoreCallback(
            prefix='cls/f1_micro',
            beta=2,
            average='micro',
            output_key=output_key,
            ignore_index=UNLABELED_CLASS),
        # F2 scores
        FScoreCallback(
            prefix='cls/f2_macro',
            beta=2,
            average='macro',
            output_key=output_key,
            ignore_index=UNLABELED_CLASS),
        FScoreCallback(
            prefix='cls/f2_micro',
            beta=2,
            average='micro',
            output_key=output_key,
            ignore_index=UNLABELED_CLASS)
    ]

    if uda:
        callbacks += [
            UDACriterionCallback(prefix='cls/uda', output_key=output_key,
                                 unsupervised_label=UNLABELED_CLASS)
        ]
    else:
        callbacks += [
            ConfusionMatrixCallback(
                prefix='cls/confusion',
                output_key=output_key,
                class_names=class_names),
            NegativeMiningCallback(ignore_index=UNLABELED_CLASS),
        ]

    if show:
        visualization_fn = partial(draw_classification_predictions,
                                   class_names=class_names)
        callbacks += [
            ShowPolarBatchesCallback(visualization_fn, metric='cls/accuracy', minimize=False)]
    return callbacks, criterions


def get_reg_callbacks(loss_name, class_names, prefix='reg', output_key='regression', uda=None, show=False):
    if len(loss_name) == 1:
        loss_name, multiplier = loss_name[0], 1.0
    elif len(loss_name) == 2:
        loss_name, multiplier = loss_name[0], float(loss_name[1])
    else:
        raise ValueError(loss_name)

    criterions = {prefix: get_loss(loss_name, ignore_index=UNLABELED_CLASS)}
    callbacks = [
        # Loss
        CriterionCallback(prefix=f'{prefix}/loss', loss_key=prefix,
                          output_key=output_key,
                          criterion_key=prefix,
                          multiplier=multiplier),
        # Metrics
        RMSEMetric(prefix=f'{prefix}/rmse',
                   output_key=output_key),

        CappaScoreCallback(prefix=f'{prefix}/kappa',
                           output_key=output_key,
                           ignore_index=UNLABELED_CLASS,
                           class_names=class_names,
                           optimize_thresholds=False,
                           from_regression=True),
        CustomAccuracyCallback(
            prefix=f'{prefix}/accuracy',
            output_key=output_key,
            from_regression=True,
            ignore_index=UNLABELED_CLASS),
        ConfusionMatrixCallbackFromRegression(
            prefix=f'{prefix}/confusion',
            output_key=output_key,
            class_names=class_names,
            ignore_index=UNLABELED_CLASS),
        # F1 scores
        FScoreCallback(
            prefix=f'{prefix}/f1_macro',
            beta=1,
            average='macro',
            output_key=output_key,
            from_regression=True,
            ignore_index=UNLABELED_CLASS),
        FScoreCallback(
            prefix=f'{prefix}/f1_micro',
            beta=2,
            average='micro',
            output_key=output_key,
            from_regression=True,
            ignore_index=UNLABELED_CLASS),
        # F2 scores
        FScoreCallback(
            prefix=f'{prefix}/f2_macro',
            beta=2,
            average='macro',
            output_key=output_key,
            from_regression=True,
            ignore_index=UNLABELED_CLASS),
        FScoreCallback(
            prefix=f'{prefix}/f2_micro',
            beta=2,
            average='micro',
            output_key=output_key,
            from_regression=True,
            ignore_index=UNLABELED_CLASS)
    ]

    if uda:
        callbacks += [
            UDARegressionCriterionCallback(
                prefix=f'{prefix}/uda',
                output_key=output_key,
                unsupervised_label=UNLABELED_CLASS)
        ]

    if show:
        visualization_fn = partial(draw_regression_predictions,
                                   outputs_key=output_key,
                                   class_names=class_names,
                                   unsupervised_label=UNLABELED_CLASS)
        callbacks += [
            ShowPolarBatchesCallback(visualization_fn, metric=f'{prefix}/accuracy', minimize=False)]

    return callbacks, criterions


def get_ord_callbacks(loss_name, class_names, uda=False, show=False):
    return get_reg_callbacks(loss_name,
                             class_names=class_names,
                             prefix='ord',
                             output_key='ordinal',
                             uda=uda,
                             show=show)
