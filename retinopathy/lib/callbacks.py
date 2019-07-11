import torch
from catalyst.dl import MetricCallback, RunnerState, Callback, MultiMetricCallback
from typing import Callable, List
import numpy as np
from catalyst.dl.callbacks import MixupCallback
from pytorch_toolbelt.utils.catalyst import get_tensorboard_logger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from pytorch_toolbelt.utils.visualization import plot_confusion_matrix, render_figure_to_tensor
from sklearn.metrics import cohen_kappa_score, confusion_matrix

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
