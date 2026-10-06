# =========================================================================
# Copyright (C) 2026. The FuxiCTR Library. All rights reserved.
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
"""MT-RankMixer: a multi-task extension of RankMixer.

The shared trunk is RankMixer (Zhu et al., CIKM 2025, arXiv:2507.15551).
Per-task token gates and towers are original to this implementation and are
not in the paper. The paper mean-pools tokens once for every task head.
``task_pooling: mean`` (or ``gate_type: mean``) restores that shared pool so
the per-task gate can be ablated without changing the trunk or the towers.
"""

from torch import nn
from fuxictr.pytorch.models import MultiTaskModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, TaskTokenGate, build_rankmixer_stack


def resolve_task_pooling(task_pooling="gate", gate_type="softmax"):
    """Resolve the pooling switch without changing the historical default.

    ``task_pooling="gate"`` with ``gate_type`` ``softmax`` or ``sigmoid`` is the
    default per-task gate. ``task_pooling="mean"`` or ``gate_type="mean"``
    shares one token mean-pool across tasks. Setting either switch to mean
    selects the shared pool.

    Args:
        task_pooling (str): ``"gate"`` or ``"mean"``. Default: ``"gate"``.
        gate_type (str): ``"softmax"``, ``"sigmoid"``, or ``"mean"``.

    Returns:
        tuple: ``(task_pooling, gate_type)`` after the alias is applied.

    Raises:
        ValueError: If the combination is not one of the supported switches.
    """
    if task_pooling not in ("gate", "mean"):
        raise ValueError("task_pooling must be 'gate' or 'mean'.")
    if gate_type not in ("softmax", "sigmoid", "mean"):
        raise ValueError("gate_type must be 'softmax', 'sigmoid', or 'mean'.")
    if task_pooling == "mean" or gate_type == "mean":
        return "mean", "mean"
    return "gate", gate_type


class MTRankMixer(MultiTaskModel):
    """Shared RankMixer backbone with per-task token gates and towers.

    Each task learns a gate over the output tokens (softmax, or sigmoid followed
    by L1 normalization) and a private MLP tower. ``task_pooling="mean"`` skips
    that gate and feeds every tower the same mean-pooled token vector.
    Token grouping is the same switch as single-task RankMixer: sequential
    chunks by default, or explicit semantic groups from the config.

    Args:
        feature_map (FeatureMap): Feature specifications.
        task (list): Per-task task types. Default: two binary tasks.
        num_tasks (int): Number of tasks. Default: ``2``.
        model_id (str): Model name. Default: ``"MTRankMixer"``.
        gpu (int): Device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate. Default: ``1e-3``.
        embedding_dim (int): Field embedding size. Default: ``16``.
        num_tokens (int or None): Token count ``T``. Default: ``8``.
        token_dim (int): Token hidden size ``D``. Default: ``64``.
        num_layers (int): Number of RankMixer blocks. Default: ``2``.
        ffn_multiplier (float): Per-token FFN expansion ratio ``k``. Default: ``4``.
        token_grouping (str): ``"sequential"`` or ``"semantic"``.
        feature_groups (list or None): Semantic groups.
        gate_type (str): ``"softmax"``, ``"sigmoid"``, or ``"mean"``.
            ``"mean"`` is an alias of ``task_pooling="mean"``. Default: ``"softmax"``.
        task_pooling (str): ``"gate"`` (per-task token gate) or ``"mean"``
            (one shared mean-pool for every task tower). Default: ``"gate"``.
        tower_hidden_units (list): Hidden units of each task tower. Default: ``[64]``.
        tower_hidden_activations (str): Tower activation. Default: ``"relu"``.
        tower_batch_norm (bool): BatchNorm in the towers. Default: ``False``.
        use_sparse_moe (bool): Sparse-MoE per-token FFNs in the shared trunk.
        num_experts (int): Experts per token. Default: ``2``.
        moe_lambda (float): L1 penalty weight. Default: ``1e-3``.
        net_dropout (float): Dropout. Default: ``0``.
        use_residual (bool): Residual connections. Default: ``True``.
        use_layer_norm (bool): LayerNorm. Default: ``True``.
        embedding_regularizer (str or None): Embedding regularizer.
        net_regularizer (str or None): Network regularizer.
        **kwargs: Optimizer, loss, loss_weight, and base-model settings.
    """
    def __init__(self,
                 feature_map,
                 task=["binary_classification", "binary_classification"],
                 num_tasks=2,
                 model_id="MTRankMixer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 num_tokens=8,
                 token_dim=64,
                 num_layers=2,
                 ffn_multiplier=4,
                 token_grouping="sequential",
                 feature_groups=None,
                 gate_type="softmax",
                 task_pooling="gate",
                 tower_hidden_units=[64],
                 tower_hidden_activations="relu",
                 tower_batch_norm=False,
                 use_sparse_moe=False,
                 num_experts=2,
                 moe_lambda=1.0e-3,
                 net_dropout=0,
                 use_residual=True,
                 use_layer_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(MTRankMixer, self).__init__(feature_map,
                                          task=task,
                                          num_tasks=num_tasks,
                                          model_id=model_id,
                                          gpu=gpu,
                                          embedding_regularizer=embedding_regularizer,
                                          net_regularizer=net_regularizer,
                                          **kwargs)
        self.moe_lambda = moe_lambda
        self.use_sparse_moe = use_sparse_moe
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.tokenizer, self.encoder, num_tokens = build_rankmixer_stack(
            feature_map,
            embedding_dim=embedding_dim,
            num_tokens=num_tokens,
            token_dim=token_dim,
            num_layers=num_layers,
            ffn_multiplier=ffn_multiplier,
            token_grouping=token_grouping,
            feature_groups=feature_groups,
            use_sparse_moe=use_sparse_moe,
            num_experts=num_experts,
            net_dropout=net_dropout,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm)
        self.num_tokens = num_tokens
        self.task_pooling, self.gate_type = resolve_task_pooling(task_pooling, gate_type)
        if self.task_pooling == "gate":
            self.task_gate = TaskTokenGate(num_tasks=self.num_tasks,
                                           num_tokens=num_tokens,
                                           token_dim=token_dim,
                                           gate_type=self.gate_type)
        else:
            # Shared mean-pool ablation: no per-task gate parameters.
            self.task_gate = None
        self.tower = nn.ModuleList([
            MLP_Block(input_dim=token_dim,
                      output_dim=1,
                      hidden_units=tower_hidden_units,
                      hidden_activations=tower_hidden_activations,
                      output_activation=None,
                      dropout_rates=net_dropout,
                      batch_norm=tower_batch_norm)
            for _ in range(self.num_tasks)
        ])
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def mix_tokens(self, tokens):
        """Pool RankMixer tokens for each task.

        In gate mode each task has its own weights over tokens. In mean mode
        every task receives the same mean-pooled vector and the gate list is
        ``None``.

        Args:
            tokens (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            tuple: A list of task vectors ``(batch, token_dim)``, and either a
            list of gate tensors ``(batch, num_tokens)`` or ``None``.
        """
        if self.task_pooling == "mean":
            shared = tokens.mean(dim=1)
            return [shared for _ in range(self.num_tasks)], None
        return self.task_gate(tokens)

    def forward(self, inputs):
        """Encode tokens once, then pool and predict each task.

        Args:
            inputs (dict): Batch dictionary from the data loader.

        Returns:
            dict: ``{label}_pred`` for each task, plus ``moe_reg`` when Sparse-MoE is on.
        """
        X = self.get_inputs(inputs)
        tokens = self.tokenizer(self.embedding_layer(X))
        tokens, moe_reg = self.encoder(tokens)
        mixed, _ = self.mix_tokens(tokens)
        labels = self.feature_map.labels
        return_dict = {}
        for i in range(self.num_tasks):
            logit = self.tower[i](mixed[i])
            return_dict["{}_pred".format(labels[i])] = self.output_activation[i](logit)
        if moe_reg is not None:
            return_dict["moe_reg"] = moe_reg
        return return_dict

    def add_loss(self, return_dict, y_true):
        """Sum of per-task losses plus the optional Sparse-MoE penalty.

        Args:
            return_dict (dict): Forward outputs.
            y_true (list): Per-task labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        loss = super(MTRankMixer, self).add_loss(return_dict, y_true)
        moe_reg = return_dict.get("moe_reg")
        if moe_reg is not None:
            loss = loss + self.moe_lambda * moe_reg
        return loss
