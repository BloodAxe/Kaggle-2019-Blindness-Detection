import os

import numpy as np
import torch
from catalyst.dl import MetricCallback, RunnerState, Callback, CriterionCallback
from catalyst.dl.callbacks import MixupCallback
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import plot_confusion_matrix, render_figure_to_tensor
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch import nn
from torch.nn import Module
import pandas as pd
from typing import List
import torch.nn.functional as F
from retinopathy.lib.models.ordinal import LogisticCumulativeLink
from retinopathy.lib.models.regression import regression_to_class


class CappaScoreCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "kappa_score"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.targets = []
        self.predictions = []

    def on_loader_start(self, state):
        self.targets = []
        self.predictions = []

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(state.output[self.output_key].detach())
        targets = to_numpy(state.input[self.input_key].detach())

        self.targets.extend(targets)
        self.predictions.extend(np.argmax(outputs, axis=1))

    def on_loader_end(self, state):
        score = cohen_kappa_score(self.predictions, self.targets, weights='quadratic')
        state.metrics.epoch_values[state.loader_name][self.prefix] = score


class CappaScoreCallbackFromRegression(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix: str = "kappa_score"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.targets = []
        self.predictions = []

    def on_loader_start(self, state):
        self.targets = []
        self.predictions = []

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(regression_to_class(state.output[self.output_key].detach()))
        targets = to_numpy(state.input[self.input_key].detach())

        self.targets.extend(targets)
        self.predictions.extend(outputs)

    def on_loader_end(self, state):
        score = cohen_kappa_score(self.predictions, self.targets, weights='quadratic')
        state.metrics.epoch_values[state.loader_name][self.prefix] = score


def accuracy_from_regression(outputs, targets):
    """
    Computes the accuracy@k for the specified values of k
    """
    batch_size = targets.size(0)

    outputs = regression_to_class(outputs.detach()).float()
    correct = outputs.eq(targets.detach())

    acc = correct.float().sum() / batch_size
    return acc


class AccuracyCallbackFromRegression(MetricCallback):
    """
    Accuracy metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "accuracy",
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`.
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=accuracy_from_regression,
            input_key=input_key,
            output_key=output_key
        )


class ConfusionMatrixCallbackFromRegression(Callback):
    """
    Compute and log confusion matrix to Tensorboard.
    For use with Multiclass classification/segmentation.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "confusion_matrix",
            class_names=None
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        self.prefix = prefix
        self.class_names = class_names
        self.output_key = output_key
        self.input_key = input_key
        self.outputs = []
        self.targets = []

    def on_loader_start(self, state):
        self.outputs = []
        self.targets = []

    def on_batch_end(self, state: RunnerState):
        outputs = to_numpy(regression_to_class(state.output[self.output_key]))
        targets = to_numpy(state.input[self.input_key])

        self.outputs.extend(outputs)
        self.targets.extend(targets)

    def on_loader_end(self, state):
        targets = np.array(self.targets)
        outputs = np.array(self.outputs)

        if self.class_names is None:
            class_names = [str(i) for i in range(targets.shape[1])]
        else:
            class_names = self.class_names

        num_classes = len(class_names)
        cm = confusion_matrix(outputs, targets, labels=range(num_classes))

        fig = plot_confusion_matrix(cm,
                                    figsize=(6 + num_classes // 3, 6 + num_classes // 3),
                                    class_names=class_names,
                                    normalize=True,
                                    noshow=True)
        fig = render_figure_to_tensor(fig)

        logger = get_tensorboard_logger(state)
        logger.add_image(f'{self.prefix}/epoch', fig, global_step=state.step)


class SWACallback(Callback):
    """
    Callback for use :'torchcontrib.optim.SWA'
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def on_loader_end(self, state: RunnerState):
        from torchcontrib.optim.swa import SWA
        if state.loader_name == 'train':
            self.optimizer.swap_swa_sgd()
            SWA.bn_update(state.loaders, state.model, state.device)


class MixupSameLabelCallback(CriterionCallback):
    """
    Callback to do mixup augmentation.

    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """

    def __init__(
            self,
            fields: List[str] = ("features",),
            alpha=1.3,
            on_train_only=True,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.target_key = 'targets'
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        targets = state.input[self.target_key]

        for label_index in torch.arange(5):
            mask = targets == label_index
            lam = np.random.beta(self.alpha, self.alpha)

            index = torch.randperm(mask.shape[0])

            for f in self.fields:
                state.input[f][mask] = lam * state.input[f][mask] + \
                                       (1 - lam) * state.input[f][mask][index]

    def _compute_loss(self, state: RunnerState, criterion):
        # As we don't change target, compute basic loss
        return super()._compute_loss(state, criterion)


class MixupRegressionCallback(MixupCallback):
    """
    Callback to do mixup augmentation.
    It's modification compute recompute the target according to:
    ```
        y = max(y_a, y_b)
    ```
    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """

    def __init__(self, fields: List[str] = ("features",), alpha=1.5, on_train_only=True, **kwargs):
        """
        Note we set alpha 1.5 to enforce mixing
        :param fields:
        :param alpha:
        :param on_train_only:
        :param kwargs:
        """
        super().__init__(fields, alpha, on_train_only, **kwargs)

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        if self.lam < 0.3 or self.lam > 0.7:
            # Do not apply mixup on small lambdas
            return

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                             (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        if self.lam < 0.3 or self.lam > 0.7:
            # Do not apply mixup on small lambdas
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a: torch.Tensor = state.input[self.input_key]
        y_b: torch.Tensor = state.input[self.input_key][self.index]
        # y = max(y_a, y_b)

        # In case of regression, if we do mixup of images of DR of different stages, we assign the maximum stage as our target
        mask = y_b > y_a
        y = y_a.masked_scatter(mask, y_b[mask])

        loss = criterion(pred, y)
        return loss


class UnsupervisedCriterionCallback(CriterionCallback):
    """
    """

    def __init__(
            self,
            input_key='original',
            output_key='logits',
            target_key='targets',
            on_train_only=True,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.input_key=input_key,
        self.target_key = target_key
        self.output_key = output_key
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def on_batch_end(self, state: RunnerState):
        targets = state.input[self.target_key]
        if not (targets == -1).any():
            # If batch contains no unsupervised samples - quit
            return

        input = state.input[self.input_key]
        output = state.model(input)[self.output_key]

        augmented_log_prob = F.log_softmax(state.output[self.output_key], dim=1)
        original_prob = F.softmax(output, dim=1)

        loss = F.kl_div(augmented_log_prob, original_prob)

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)



class AscensionCallback(Callback):
    """
    Ensure that each cutpoint is ordered in ascending value.
    e.g.
    .. < cutpoint[i - 1] < cutpoint[i] < cutpoint[i + 1] < ...
    This is done by clipping the cutpoint values at the end of a batch gradient
    update. By no means is this an efficient way to do things, but it works out
    of the box with stochastic gradient descent.
    Parameters
    ----------
    margin : float, (default=0.0)
        The minimum value between any two adjacent cutpoints.
        e.g. enforce that cutpoint[i - 1] + margin < cutpoint[i]
    min_val : float, (default=-1e6)
        Minimum value that the smallest cutpoint may take.
    """

    def __init__(self, net: nn.Module, margin: float = 0.0, min_val: float = -1.0e6) -> None:
        super().__init__()
        self.net = net
        self.margin = margin
        self.min_val = min_val

    def clip(self, module: Module) -> None:
        # NOTE: Only works for LogisticCumulativeLink right now
        # We assume the cutpoints parameters are called `cutpoints`.
        if isinstance(module, LogisticCumulativeLink):
            cutpoints = module.cutpoints.data
            for i in range(cutpoints.shape[0] - 1):
                cutpoints[i].clamp_(self.min_val,
                                    cutpoints[i + 1] - self.margin)

    def on_batch_end(self, state: RunnerState):
        self.net.apply(self.clip)


class NegativeMiningCallback(Callback):

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            from_regression=False
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        self.output_key = output_key
        self.input_key = input_key
        self.from_regression = from_regression
        self.image_ids = []
        self.y_preds = []
        self.y_trues = []

    def on_loader_start(self, state: RunnerState):
        self.image_ids = []
        self.y_preds = []
        self.y_trues = []

    def on_loader_end(self, state: RunnerState):
        df = pd.DataFrame.from_dict({
            'image_id': self.image_ids,
            'y_true': self.y_trues,
            'y_pred': self.y_preds
        })

        fname = os.path.join(state.logdir, 'negatives', state.loader_name, f'epoch_{state.epoch}.csv')
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        df.to_csv(fname, index=None)

    def on_batch_end(self, state: RunnerState):
        y_true = state.input[self.input_key].detach()
        y_pred = state.output[self.output_key].detach()

        if self.from_regression:
            y_pred = regression_to_class(y_pred)
        else:
            y_pred = torch.argmax(y_pred, dim=1)

        y_pred = to_numpy(y_pred).astype(int)
        y_true = to_numpy(y_true).astype(int)
        negatives = y_true != y_pred
        image_ids = np.array(state.input['image_id'])

        self.image_ids.extend(image_ids[negatives])
        self.y_preds.extend(y_pred[negatives])
        self.y_trues.extend(y_true[negatives])


class WeightDecayCallback(Callback):
    def __init__(self, optimizer, start_wd=0, epoch_step=5e-6):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.epoch_step = epoch_step
        self.current_wd = start_wd

    def on_stage_end(self, state: RunnerState):
        self.current_wd += self.epoch_step
        for pg in self.optimizer.param_groups:
            pg["weight_decay"] = self.current_wd
