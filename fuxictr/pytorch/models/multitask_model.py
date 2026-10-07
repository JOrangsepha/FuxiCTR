# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import numpy as np
import torch
import os, sys
import logging
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from tqdm import tqdm
from collections import defaultdict


# ``NORM_FLOOR`` is a local guard, not a result. No Ali-CCP run used it.
NORM_FLOOR_EXPERIMENTALLY_EVALUATED = False
NORM_FLOOR_MIN = 1e-2


def combine_task_losses(losses, loss_weight="EQ"):
    """Combine per-task scalar losses.

    ``EQ`` is the historical default: the unnormalized sum of the per-task
    mean losses. It does not divide by the number of tasks and it does not
    rescale by the loss magnitude. On Ali-CCP the click BCE is much larger
    than the conversion BCE, so the sum is dominated by click. Training
    configs keep ``EQ``.

    ``NORM`` divides each task loss by its detached absolute value, then
    sums, so each task contributes about 1 when every task loss is nonzero.
    The multiplier on task ``k`` is ``1 / max(|L_k|, 1e-12)``. On Ali-CCP
    about 16–17% of 8192-row batches contain no conversion. There ``L_conv``
    underflows toward 0 (or equals ``mean(p)``), and the conversion gradient
    is multiplied by up to ``1e12``. Two stopped ``g6_mean`` NORM runs
    (seeds 2025 and 2026, epoch-1 validation) landed at conversion AUC
    0.5007 and 0.5327, with epoch-mean train loss 1.843 and 1.842 instead
    of 2.0. See ``benchmarks/rankmixer/results/followup/NORM_collapse_evidence.md``.
    ``NORM`` stays selectable. It is not stable on this data, and it is not
    a training default.

    ``NORM_FLOOR`` is the same formula with the denominator clamped to
    ``NORM_FLOOR_MIN`` (1e-2), so the multiplier is at most 100.
    ``NORM_FLOOR_EXPERIMENTALLY_EVALUATED`` is False. No follow-up job used it.

    A sequence of floats is a manual weight vector applied as
    ``sum_k w_k * L_k``. The recorded alternative to per-batch NORM was
    ``[1.0, 10.0]``.

    Args:
        losses (list): One scalar tensor per task.
        loss_weight (str or list): ``"EQ"``, ``"NORM"``, ``"NORM_FLOOR"``,
            or one weight per task.

    Returns:
        torch.Tensor: Scalar combined loss.

    Raises:
        ValueError: If ``loss_weight`` is not a supported mode.
    """
    stacked = torch.stack(list(losses))
    if isinstance(loss_weight, str):
        mode = loss_weight.upper()
        if mode == "EQ":
            return torch.sum(stacked)
        if mode == "NORM":
            # Floor of 1e-12 only avoids division by exact zero. It does not
            # stop the 1/L blow-up on an empty conversion batch.
            scale = stacked.detach().abs().clamp_min(1e-12)
            return torch.sum(stacked / scale)
        if mode == "NORM_FLOOR":
            # NOT experimentally evaluated. See NORM_FLOOR_EXPERIMENTALLY_EVALUATED.
            scale = stacked.detach().abs().clamp_min(NORM_FLOOR_MIN)
            return torch.sum(stacked / scale)
        raise ValueError(
            "loss_weight must be 'EQ', 'NORM', 'NORM_FLOOR', or a sequence of floats. "
            "Got {!r}.".format(loss_weight))
    if isinstance(loss_weight, (list, tuple)):
        if len(loss_weight) != int(stacked.shape[0]):
            raise ValueError(
                "loss_weight length {} does not match {} tasks.".format(len(loss_weight), int(stacked.shape[0])))
        weights = stacked.new_tensor([float(value) for value in loss_weight])
        return torch.sum(stacked * weights)
    raise ValueError(
        "loss_weight must be 'EQ', 'NORM', 'NORM_FLOOR', or a sequence of floats. "
        "Got {!r}.".format(loss_weight))


class MultiTaskModel(BaseModel):
    """Base class for multi-task learning models in PyTorch.

    Extends ``BaseModel`` to support multiple prediction tasks with separate
    output activations and loss functions per task.

    Args:
        feature_map (FeatureMap): Feature map object.
        model_id (str): Model identifier. Default: ``"MultiTaskModel"``.
        task (list or str): Task type(s) for each task. Default: ``["binary_classification"]``.
        num_tasks (int): Number of tasks. Default: ``1``.
        loss_weight (str or list): ``"EQ"`` (unnormalized sum, the default),
            ``"NORM"`` (each loss divided by its detached magnitude; unstable
            when a task loss is ~0), ``"NORM_FLOOR"`` (same idea, denominator
            clamped at 1e-2, not experimentally evaluated), or a list of
            per-task weights. Default: ``"EQ"``.
        gpu (int): GPU device ID, -1 for CPU. Default: ``-1``.
        monitor (str): Metric to monitor for early stopping. Default: ``"AUC"``.
        save_best_only (bool): Whether to save only the best model. Default: ``True``.
        monitor_mode (str): ``max`` or ``min`` for the monitored metric. Default: ``"max"``.
        early_stop_patience (int): Patience for early stopping. Default: ``2``.
        eval_steps (int or None): Evaluation frequency in steps. Default: ``None``.
        embedding_regularizer (str or None): Regularizer for embeddings. Default: ``None``.
        net_regularizer (str or None): Regularizer for network weights. Default: ``None``.
        reduce_lr_on_plateau (bool): Whether to reduce LR on plateau. Default: ``True``.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self,
                 feature_map,
                 model_id="MultiTaskModel",
                 task=["binary_classification"],
                 num_tasks=1,
                 loss_weight='EQ',
                 gpu=-1,
                 monitor="AUC",
                 save_best_only=True,
                 monitor_mode="max",
                 early_stop_patience=2,
                 eval_steps=None,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(MultiTaskModel, self).__init__(feature_map=feature_map,
                                             model_id=model_id,
                                             task="binary_classification",
                                             gpu=gpu,
                                             loss_weight=loss_weight,
                                             monitor=monitor,
                                             save_best_only=save_best_only,
                                             monitor_mode=monitor_mode,
                                             early_stop_patience=early_stop_patience,
                                             eval_steps=eval_steps,
                                             embedding_regularizer=embedding_regularizer,
                                             net_regularizer=net_regularizer,
                                             reduce_lr_on_plateau=reduce_lr_on_plateau,
                                             **kwargs)
        self.device = get_device(gpu)
        self.num_tasks = num_tasks
        self.loss_weight = loss_weight
        if isinstance(task, list):
            assert len(task) == num_tasks, "the number of tasks must equal the length of \"task\""
            self.output_activation = nn.ModuleList([self.get_output_activation(str(t)) for t in task])
        else:
            self.output_activation = nn.ModuleList(
                [self.get_output_activation(task) for _ in range(num_tasks)]
            )

    def compile(self, optimizer, loss, lr):
        """Configure the optimizer and loss functions for multi-task training.

        Args:
            optimizer (str): Optimizer name.
            loss (str or list): Loss function name(s). If a list, each task uses
                the corresponding loss; otherwise the same loss is used for all tasks.
            lr (float): Learning rate.
        """
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        if isinstance(loss, list):
            self.loss_fn = [get_loss(l) for l in loss]
        else:
            self.loss_fn = [get_loss(loss) for _ in range(self.num_tasks)]

    def get_labels(self, inputs):
        """Override get_labels() to use multiple labels.

        Args:
            inputs (dict): Dictionary of input tensors keyed by feature name.

        Returns:
            list: List of label tensors for each task.
        """
        labels = self.feature_map.labels
        y = [inputs[labels[i]].to(self.device).float().view(-1, 1)
             for i in range(len(labels))]
        return y

    def regularization_loss(self):
        """Compute the combined embedding and network regularization loss.

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        reg_loss = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_loss

    def add_loss(self, return_dict, y_true):
        """Compute the task losses without regularization.

        Args:
            return_dict (dict): Model forward outputs containing per-task predictions.
            y_true (list): List of ground-truth tensors for each task.

        Returns:
            torch.Tensor: Combined task loss.
        """
        labels = self.feature_map.labels
        loss = [self.loss_fn[i](return_dict["{}_pred".format(labels[i])], y_true[i], reduction='mean')
                for i in range(len(labels))]
        # EQ remains the unnormalized sum. NORM and a weight list are opt-in.
        return combine_task_losses(loss, self.loss_weight)

    def compute_loss(self, return_dict, y_true):
        """Compute the total loss including regularization.

        Args:
            return_dict (dict): Model forward outputs.
            y_true (list): List of ground-truth tensors for each task.

        Returns:
            torch.Tensor: Total loss value.
        """
        loss = self.add_loss(return_dict, y_true) + self.regularization_loss()
        return loss

    def evaluate(self, data_generator, metrics=None):
        """Evaluate the model on a validation data generator.

        Args:
            data_generator: Data generator yielding batches.
            metrics (list, optional): List of metric names to compute.

        Returns:
            dict: Mapping of per-task and mean metric names to values.
        """
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            y_true_all = defaultdict(list)
            labels = self.feature_map.labels
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                batch_y_true = self.get_labels(batch_data)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["{}_pred".format(labels[i])].data.cpu().numpy().reshape(-1))
                    y_true_all[labels[i]].extend(batch_y_true[i].data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            all_val_logs = {}
            mean_val_logs = defaultdict(list)
            group_id = np.array(group_id) if len(group_id) > 0 else None

            for i in range(len(labels)):
                y_pred = np.array(y_pred_all[labels[i]], np.float64)
                y_true = np.array(y_true_all[labels[i]], np.float64)
                if metrics is not None:
                    val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
                else:
                    val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
                logging.info('[Task: {}][Metrics] '.format(labels[i]) + ' - '.join(
                    '{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
                for k, v in val_logs.items():
                    all_val_logs['{}_{}'.format(labels[i], k)] = v
                    mean_val_logs[k].append(v)
            for k, v in mean_val_logs.items():
                mean_val_logs[k] = np.mean(v)
            all_val_logs.update(mean_val_logs)
            return all_val_logs

    def predict(self, data_generator):
        """Generate predictions on a data generator.

        Args:
            data_generator: Data generator yielding batches.

        Returns:
            dict: Mapping of label names to predicted numpy arrays.
        """
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred_all = defaultdict(list)
            labels = self.feature_map.labels
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                for i in range(len(labels)):
                    y_pred_all[labels[i]].extend(
                        return_dict["{}_pred".format(labels[i])].data.cpu().numpy().reshape(-1))
        return y_pred_all
