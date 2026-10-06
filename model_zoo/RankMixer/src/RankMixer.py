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
"""RankMixer: Scaling Up Ranking Models in Industrial Recommenders.

Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, et al. CIKM 2025.
arXiv: https://arxiv.org/abs/2507.15551
"""

from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, build_rankmixer_stack


class RankMixer(BaseModel):
    """Single-task RankMixer ranking model.

    Field embeddings are tokenized into ``T`` tokens, mixed by ``L`` RankMixer
    blocks (parameter-free multi-head token mixing and a per-token FFN), then
    mean-pooled into the prediction MLP. The paper's final head is unspecified;
    the head here is a configurable MLP on the pooled vector, defaulting to a
    single linear layer.

    Args:
        feature_map (FeatureMap): Feature specifications.
        model_id (str): Model name. Default: ``"RankMixer"``.
        gpu (int): Device index, ``-1`` for CPU. Default: ``-1``.
        learning_rate (float): Learning rate. Default: ``1e-3``.
        embedding_dim (int): Field embedding size. Default: ``16``.
        num_tokens (int): Number of tokens ``T``. Must divide ``token_dim``. Default: ``8``.
        token_dim (int): Token hidden size ``D``. Default: ``64``.
        num_layers (int): Number of RankMixer blocks ``L``. Default: ``2``.
        ffn_multiplier (float): Per-token FFN expansion ratio ``k``. Default: ``4``.
        token_grouping (str): ``"sequential"`` (paper default) or ``"semantic"``.
        feature_groups (list or None): Semantic groups. Used only when
            ``token_grouping="semantic"``.
        use_sparse_moe (bool): Replace each per-token FFN with ReLU-routed Sparse-MoE.
        num_experts (int): Experts per token when Sparse-MoE is on. Default: ``2``.
        moe_lambda (float): Weight of the ReLU-gate L1 penalty. Default: ``1e-3``.
        mlp_hidden_units (list): Hidden units of the prediction MLP. Default: ``[]``.
        mlp_hidden_activations (str): Activation of the prediction MLP. Default: ``"relu"``.
        mlp_batch_norm (bool): BatchNorm in the prediction MLP. Default: ``False``.
        net_dropout (float): Dropout on RankMixer sublayers. Default: ``0``.
        use_residual (bool): Residual connections inside each block. Default: ``True``.
        use_layer_norm (bool): LayerNorm inside each block. Default: ``True``.
        embedding_regularizer (str or None): Embedding regularizer. Default: ``None``.
        net_regularizer (str or None): Network regularizer. Default: ``None``.
        **kwargs: Optimizer, loss, and base-model settings.
    """
    def __init__(self,
                 feature_map,
                 model_id="RankMixer",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 num_tokens=8,
                 token_dim=64,
                 num_layers=2,
                 ffn_multiplier=4,
                 token_grouping="sequential",
                 feature_groups=None,
                 use_sparse_moe=False,
                 num_experts=2,
                 moe_lambda=1.0e-3,
                 mlp_hidden_units=[],
                 mlp_hidden_activations="relu",
                 mlp_batch_norm=False,
                 net_dropout=0,
                 use_residual=True,
                 use_layer_norm=True,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(RankMixer, self).__init__(feature_map,
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
        self.fc = MLP_Block(input_dim=token_dim,
                            output_dim=1,
                            hidden_units=mlp_hidden_units,
                            hidden_activations=mlp_hidden_activations,
                            output_activation=self.output_activation,
                            batch_norm=mlp_batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """Embed, tokenize, encode, mean-pool, and predict.

        Args:
            inputs (dict): Batch dictionary from the data loader.

        Returns:
            dict: ``y_pred`` and, when Sparse-MoE is enabled, ``moe_reg``.
        """
        X = self.get_inputs(inputs)
        tokens = self.tokenizer(self.embedding_layer(X))
        tokens, moe_reg = self.encoder(tokens)
        y_pred = self.fc(tokens.mean(dim=1))
        return_dict = {"y_pred": y_pred}
        if moe_reg is not None:
            return_dict["moe_reg"] = moe_reg
        return return_dict

    def add_loss(self, return_dict, y_true):
        """Task loss plus the optional Sparse-MoE L1 penalty.

        Args:
            return_dict (dict): Forward outputs.
            y_true (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        loss = super(RankMixer, self).add_loss(return_dict, y_true)
        moe_reg = return_dict.get("moe_reg")
        if moe_reg is not None:
            loss = loss + self.moe_lambda * moe_reg
        return loss
