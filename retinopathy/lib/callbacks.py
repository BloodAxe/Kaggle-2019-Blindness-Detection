import os

import numpy as np
import torch
from catalyst.dl import MetricCallback, RunnerState, Callback
from catalyst.dl.callbacks import MixupCallback
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import plot_confusion_matrix, render_figure_to_tensor
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch import nn
from torch.nn import Module

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


class MixupRegressionCallback(MixupCallback):
    """
    Callback to do mixup augmentation.
    It's modification compute recompute the target according to:
    ```
        y = y_a * self.lam + y_b * (1 - self.lam)
    ```
    Paper: https://arxiv.org/abs/1710.09412

    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]
        y = y_a * self.lam + y_b * (1 - self.lam)
        loss = criterion(pred, y)
        return loss


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


import pandas as pd


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

        y_pred = to_numpy(y_pred)
        y_true = to_numpy(y_true)
        negatives = y_true != y_pred
        image_ids = np.array(state.input['image_id'])

        self.image_ids.extend(image_ids[negatives])
        self.y_preds.extend(y_pred[negatives])
        self.y_trues.extend(y_true[negatives])
