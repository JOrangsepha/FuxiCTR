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
"""RankMixer layers from Zhu et al., CIKM 2025 / arXiv:2507.15551.

The block follows Eq. (1) of the paper:

    S = LN(TokenMixing(X) + X)
    X' = LN(PFFN(S) + S)

Token mixing is parameter-free and uses H = T heads so the residual shape is
unchanged (Eq. 3-5). Per-token FFNs do not share parameters across tokens
(Eq. 6-9). The optional Sparse-MoE path replaces each per-token FFN with
ReLU-routed experts (Eq. 10-11).
"""

import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    "MultiHeadTokenMixing",
    "PerTokenFFN",
    "PerTokenSparseMoE",
    "RankMixerBlock",
    "RankMixerEncoder",
    "FeatureTokenizer",
    "TaskTokenGate",
    "resolve_feature_groups",
    "collect_input_features",
    "split_sequential_chunks",
    "build_feature_tokenizer",
    "build_rankmixer_stack",
]


def collect_input_features(feature_map):
    """List non-meta features in ``FeatureEmbedding`` order.

    Args:
        feature_map (FeatureMap): Feature map whose ``features`` dict is ordered.

    Returns:
        tuple: ``(names, sources)`` aligned with the field axis of the embedding tensor.
    """
    names, sources = [], []
    for name, spec in feature_map.features.items():
        if spec.get("type") == "meta":
            continue
        names.append(str(name))
        sources.append(spec.get("source", "") or "")
    return names, sources


def resolve_feature_groups(feature_names, feature_sources, feature_groups, num_tokens=None):
    """Parse semantic token groups into field indices.

    Each entry of ``feature_groups`` is either a list of feature names or a string.
    A string is treated as a feature ``source`` when at least one feature has that
    source, and otherwise as a single feature name. Every input feature must belong
    to exactly one group. Feature names are coerced with ``str`` so YAML integers
    such as ``101`` match string column names.

    Args:
        feature_names (list): Feature names in embedding order.
        feature_sources (list): Source tag of each feature, same length as ``feature_names``.
        feature_groups (list): User-specified groups from the config.
        num_tokens (int or None): If set, must equal the number of groups.

    Returns:
        list: One list of field indices per token.

    Raises:
        ValueError: If the config is empty, names a missing feature, repeats a
            feature, leaves a feature out, or disagrees with ``num_tokens``.
    """
    if not feature_groups:
        raise ValueError("feature_groups is required when token_grouping='semantic'.")
    if not isinstance(feature_groups, (list, tuple)):
        raise ValueError("feature_groups must be a list of groups.")
    if len(feature_names) != len(feature_sources):
        raise ValueError("feature_names and feature_sources must have the same length.")

    name_to_idx = {str(name): i for i, name in enumerate(feature_names)}
    source_to_idxs = {}
    for i, source in enumerate(feature_sources):
        source_to_idxs.setdefault(str(source), []).append(i)

    groups = []
    used = set()
    for group_id, group in enumerate(feature_groups):
        if isinstance(group, str) or not isinstance(group, (list, tuple)):
            key = str(group)
            if key in source_to_idxs and source_to_idxs[key]:
                idxs = list(source_to_idxs[key])
            elif key in name_to_idx:
                idxs = [name_to_idx[key]]
            else:
                raise ValueError(
                    "Unknown feature group {!r}. Expected a feature name or a feature source."
                    .format(group))
        else:
            idxs = []
            for name in group:
                key = str(name)
                if key not in name_to_idx:
                    raise ValueError("Unknown feature in feature_groups: {!r}".format(name))
                idxs.append(name_to_idx[key])
        if len(idxs) == 0:
            raise ValueError("feature group {} is empty.".format(group_id))
        if len(idxs) != len(set(idxs)):
            raise ValueError("feature group {} contains duplicate features.".format(group_id))
        for idx in idxs:
            if idx in used:
                raise ValueError(
                    "feature {!r} is assigned to more than one group.".format(feature_names[idx]))
            used.add(idx)
        groups.append(idxs)

    missing = [feature_names[i] for i in range(len(feature_names)) if i not in used]
    if missing:
        preview = ", ".join(missing[:20])
        extra = "" if len(missing) <= 20 else ", ... ({} more)".format(len(missing) - 20)
        raise ValueError(
            "semantic grouping must cover every feature; missing: {}{}".format(preview, extra))
    if num_tokens is not None and int(num_tokens) != len(groups):
        raise ValueError(
            "num_tokens={} does not match the number of feature groups ({})."
            .format(num_tokens, len(groups)))
    return groups


def split_sequential_chunks(flat, num_tokens):
    """Pad and split a flattened embedding into ``num_tokens`` equal chunks.

    This is the paper's sequential partition of ``e_input`` (Eq. 2) before ``Proj``.
    The last chunk is zero-padded when ``F * E`` is not divisible by ``T``.

    Args:
        flat (torch.Tensor): Tensor of shape ``(batch, total_dim)``.
        num_tokens (int): Number of tokens ``T``.

    Returns:
        torch.Tensor: Tensor of shape ``(batch, num_tokens, chunk_dim)``.
    """
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive.")
    total = flat.size(-1)
    if num_tokens > total:
        raise ValueError(
            "num_tokens ({}) cannot exceed the flattened embedding dim ({}).".format(
                num_tokens, total))
    chunk_dim = (total + num_tokens - 1) // num_tokens
    pad = chunk_dim * num_tokens - total
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat.view(flat.size(0), num_tokens, chunk_dim)


class FeatureTokenizer(nn.Module):
    """Map a field-embedding tensor to ``T`` tokens of dimension ``D``.

    Sequential mode flattens fields in feature-map order and applies one shared
    projection, matching ``Proj`` in Eq. (2). Semantic mode projects each
    user-specified group with its own matrix because group widths differ.

    Args:
        num_fields (int): Number of input fields.
        embedding_dim (int): Per-field embedding size. Field dims must be uniform,
            which is what ``FeatureEmbedding`` stacks.
        num_tokens (int): Number of output tokens.
        token_dim (int): Token hidden size ``D``.
        token_grouping (str): ``"sequential"`` or ``"semantic"``. Default: ``"sequential"``.
        group_indices (list or None): Field indices per token. Required for semantic mode.
    """
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 num_tokens,
                 token_dim,
                 token_grouping="sequential",
                 group_indices=None):
        super(FeatureTokenizer, self).__init__()
        if num_fields < 1:
            raise ValueError("num_fields must be positive.")
        if embedding_dim < 1 or token_dim < 1:
            raise ValueError("embedding_dim and token_dim must be positive.")
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.token_grouping = token_grouping
        if token_grouping == "sequential":
            total_dim = num_fields * embedding_dim
            self.chunk_dim = (total_dim + num_tokens - 1) // num_tokens
            self.proj = nn.Linear(self.chunk_dim, token_dim)
            self.group_indices = None
        elif token_grouping == "semantic":
            if not group_indices:
                raise ValueError("group_indices is required for semantic tokenization.")
            self.group_indices = [list(idxs) for idxs in group_indices]
            self.num_tokens = len(self.group_indices)
            self.projs = nn.ModuleList([
                nn.Linear(len(idxs) * embedding_dim, token_dim) for idxs in self.group_indices
            ])
        else:
            raise ValueError("token_grouping must be 'sequential' or 'semantic'.")

    def forward(self, feature_emb):
        """Tokenize field embeddings.

        Args:
            feature_emb (torch.Tensor): Shape ``(batch, num_fields, embedding_dim)``.

        Returns:
            torch.Tensor: Shape ``(batch, num_tokens, token_dim)``.
        """
        if feature_emb.dim() != 3:
            raise ValueError(
                "FeatureTokenizer expects field embeddings of shape (batch, fields, dim). "
                "Sequence features must be pooled to one vector per field before RankMixer.")
        if self.token_grouping == "sequential":
            chunks = split_sequential_chunks(feature_emb.flatten(start_dim=1), self.num_tokens)
            return self.proj(chunks)
        tokens = []
        for proj, idxs in zip(self.projs, self.group_indices):
            gathered = feature_emb[:, idxs, :].flatten(start_dim=1)
            tokens.append(proj(gathered))
        return torch.stack(tokens, dim=1)


def build_feature_tokenizer(feature_map,
                            embedding_dim,
                            num_tokens,
                            token_dim,
                            token_grouping="sequential",
                            feature_groups=None):
    """Build a tokenizer and return it together with the resolved token count.

    Args:
        feature_map (FeatureMap): Dataset feature map.
        embedding_dim (int): Embedding size used by ``FeatureEmbedding``.
        num_tokens (int or None): Requested token count. Required for sequential mode.
            In semantic mode it is checked against the number of groups when set.
        token_dim (int): Token hidden size.
        token_grouping (str): ``"sequential"`` or ``"semantic"``.
        feature_groups (list or None): Semantic groups. Rejected for sequential mode.

    Returns:
        tuple: ``(FeatureTokenizer, num_tokens)``.
    """
    names, sources = collect_input_features(feature_map)
    grouping = (token_grouping or "sequential").lower()
    if grouping == "sequential":
        if feature_groups not in (None, [], ""):
            raise ValueError("feature_groups is only used when token_grouping='semantic'.")
        if num_tokens is None:
            raise ValueError("num_tokens is required when token_grouping='sequential'.")
        num_tokens = int(num_tokens)
        tokenizer = FeatureTokenizer(num_fields=len(names),
                                     embedding_dim=embedding_dim,
                                     num_tokens=num_tokens,
                                     token_dim=token_dim,
                                     token_grouping="sequential")
        return tokenizer, num_tokens
    if grouping == "semantic":
        groups = resolve_feature_groups(names, sources, feature_groups, num_tokens)
        tokenizer = FeatureTokenizer(num_fields=len(names),
                                     embedding_dim=embedding_dim,
                                     num_tokens=len(groups),
                                     token_dim=token_dim,
                                     token_grouping="semantic",
                                     group_indices=groups)
        return tokenizer, len(groups)
    raise ValueError("token_grouping must be 'sequential' or 'semantic'.")


class MultiHeadTokenMixing(nn.Module):
    """Parameter-free multi-head token mixing (Eq. 3-5).

    Each token is split into ``H`` heads. The ``h``-th mixed token is the
    concatenation of the ``h``-th head of every input token. The paper sets
    ``H = T`` so each mixed token has dimension ``D`` and the residual connection
    is shape-aligned. ``D`` must be divisible by ``T``.

    Args:
        num_tokens (int): Number of tokens ``T``, also the number of heads.
        token_dim (int): Token dimension ``D``.
    """
    def __init__(self, num_tokens, token_dim):
        super(MultiHeadTokenMixing, self).__init__()
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive.")
        if token_dim % num_tokens != 0:
            raise ValueError(
                "token_dim ({}) must be divisible by num_tokens ({}) because RankMixer "
                "uses H = T mixing heads.".format(token_dim, num_tokens))
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_heads = num_tokens
        self.head_dim = token_dim // num_tokens

    def forward(self, x):
        """Mix tokens.

        Args:
            x (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            torch.Tensor: Shape ``(batch, num_tokens, token_dim)``.
        """
        batch, num_tokens, token_dim = x.shape
        heads = x.view(batch, num_tokens, self.num_heads, self.head_dim)
        # (batch, head, token, head_dim) -> concat tokens inside each head.
        mixed = heads.permute(0, 2, 1, 3).contiguous()
        return mixed.view(batch, self.num_heads, num_tokens * self.head_dim)


class PerTokenFFN(nn.Module):
    """Per-token two-layer FFN with untied parameters (Eq. 6-7).

    Token ``t`` uses its own ``W^{t,1}, b^{t,1}, W^{t,2}, b^{t,2}`` and GELU.
    The hidden width is ``k * D`` where ``k`` is ``ffn_multiplier``.

    Args:
        num_tokens (int): Number of tokens.
        token_dim (int): Input and output dimension ``D``.
        hidden_dim (int): Inner dimension ``kD``.
        dropout (float): Dropout between the two linear layers. The paper uses 0.
    """
    def __init__(self, num_tokens, token_dim, hidden_dim, dropout=0.0):
        super(PerTokenFFN, self).__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        self.num_tokens = num_tokens
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, token_dim),
            ) for _ in range(num_tokens)
        ])

    def forward(self, x):
        """Apply an independent FFN on each token.

        Args:
            x (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            torch.Tensor: Shape ``(batch, num_tokens, token_dim)``.
        """
        outputs = [self.ffns[t](x[:, t, :]) for t in range(self.num_tokens)]
        return torch.stack(outputs, dim=1)


class PerTokenSparseMoE(nn.Module):
    """Per-token Sparse-MoE FFN with ReLU routing (Eq. 10-11).

    Each token owns ``num_experts`` FFNs. Two routers implement a practical form
    of the paper's dense-training / sparse-inference (DTSI) scheme:

    * ``router_train`` is a softmax gate used only while training, so every expert
      receives a task gradient.
    * ``router_infer`` is a ReLU gate. Its raw activations are the L1 penalty
      ``L_reg``. At eval time the forward pass uses this gate only.

    During training the residual stream mixes the two gates equally. The sparse
    half is normalized so its scale matches softmax; the penalty still uses the
    raw ReLU values. ``L_reg`` is averaged over the batch so ``moe_lambda`` does
    not grow with batch size. The paper writes an unnormalized sum.

    Args:
        num_tokens (int): Number of tokens.
        token_dim (int): Token dimension.
        hidden_dim (int): Expert inner dimension.
        num_experts (int): Experts per token.
        dropout (float): Dropout inside each expert. Default: ``0``.
    """
    def __init__(self, num_tokens, token_dim, hidden_dim, num_experts, dropout=0.0):
        super(PerTokenSparseMoE, self).__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive.")
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, token_dim),
                ) for _ in range(num_experts)
            ]) for _ in range(num_tokens)
        ])
        self.router_train = nn.ModuleList(
            [nn.Linear(token_dim, num_experts) for _ in range(num_tokens)])
        self.router_infer = nn.ModuleList(
            [nn.Linear(token_dim, num_experts) for _ in range(num_tokens)])
        self.init_weights()

    def init_weights(self):
        """Start the inference router fully open, then let the L1 penalty sparsify it."""
        for router in self.router_infer:
            nn.init.zeros_(router.weight)
            nn.init.constant_(router.bias, 1.0)

    def forward(self, x):
        """Route each token through its own experts.

        Args:
            x (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            tuple: Mixed tokens ``(batch, num_tokens, token_dim)`` and a scalar
            regularization term.
        """
        mixed = []
        reg_per_token = []
        for t in range(self.num_tokens):
            token = x[:, t, :]
            expert_out = torch.stack(
                [self.experts[t][e](token) for e in range(self.num_experts)], dim=1)
            g_train = torch.softmax(self.router_train[t](token), dim=-1)
            g_raw = F.relu(self.router_infer[t](token))
            reg_per_token.append(g_raw.sum(dim=-1))
            if self.training:
                g_sparse = g_raw / g_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                gates = 0.5 * g_train + 0.5 * g_sparse
            else:
                gates = g_raw
            mixed.append(torch.einsum("be,bed->bd", gates, expert_out))
        # Mean over the batch of the paper's sum over tokens and experts.
        reg = torch.stack(reg_per_token, dim=1).sum(dim=1).mean()
        return torch.stack(mixed, dim=1), reg


class RankMixerBlock(nn.Module):
    """One RankMixer block: token mixing, per-token FFN, residual, LayerNorm.

    Args:
        num_tokens (int): Number of tokens.
        token_dim (int): Token dimension.
        hidden_dim (int): FFN inner dimension.
        use_sparse_moe (bool): Replace the dense per-token FFN with Sparse-MoE.
        num_experts (int): Experts per token when ``use_sparse_moe`` is set.
        net_dropout (float): Dropout on each sublayer branch before the residual.
        use_residual (bool): Add the residual connection from Eq. (1).
        use_layer_norm (bool): Apply LayerNorm after the residual, as in Eq. (1).
        ffn_dropout (float): Dropout inside the FFN. Default: ``0``.
    """
    def __init__(self,
                 num_tokens,
                 token_dim,
                 hidden_dim,
                 use_sparse_moe=False,
                 num_experts=2,
                 net_dropout=0.0,
                 use_residual=True,
                 use_layer_norm=True,
                 ffn_dropout=0.0):
        super(RankMixerBlock, self).__init__()
        self.use_residual = use_residual
        self.token_mixing = MultiHeadTokenMixing(num_tokens, token_dim)
        self.norm1 = nn.LayerNorm(token_dim) if use_layer_norm else None
        self.norm2 = nn.LayerNorm(token_dim) if use_layer_norm else None
        self.dropout = nn.Dropout(net_dropout) if net_dropout and net_dropout > 0 else nn.Identity()
        if use_sparse_moe:
            self.pffn = PerTokenSparseMoE(num_tokens, token_dim, hidden_dim, num_experts, ffn_dropout)
        else:
            self.pffn = PerTokenFFN(num_tokens, token_dim, hidden_dim, ffn_dropout)

    def _merge(self, transformed, residual, norm):
        out = transformed + residual if self.use_residual else transformed
        if norm is not None:
            out = norm(out)
        return out

    def forward(self, x):
        """Run token mixing and the per-token FFN.

        Args:
            x (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            tuple: Updated tokens and an optional MoE penalty (``None`` for the dense FFN).
        """
        mixed = self._merge(self.dropout(self.token_mixing(x)), x, self.norm1)
        pffn_out = self.pffn(mixed)
        if isinstance(pffn_out, tuple):
            transformed, reg = pffn_out
        else:
            transformed, reg = pffn_out, None
        out = self._merge(self.dropout(transformed), mixed, self.norm2)
        return out, reg


class RankMixerEncoder(nn.Module):
    """Stack of ``L`` RankMixer blocks.

    Args:
        num_tokens (int): Number of tokens.
        token_dim (int): Token dimension.
        num_layers (int): Depth ``L``.
        ffn_multiplier (float): Hidden-width ratio ``k``. Inner dim is ``k * D``.
        use_sparse_moe (bool): Use Sparse-MoE FFNs.
        num_experts (int): Experts per token.
        net_dropout (float): Sublayer dropout.
        use_residual (bool): Enable residual connections.
        use_layer_norm (bool): Enable LayerNorm.
    """
    def __init__(self,
                 num_tokens,
                 token_dim,
                 num_layers,
                 ffn_multiplier=4,
                 use_sparse_moe=False,
                 num_experts=2,
                 net_dropout=0.0,
                 use_residual=True,
                 use_layer_norm=True):
        super(RankMixerEncoder, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        hidden_dim = int(ffn_multiplier * token_dim)
        if hidden_dim < 1:
            raise ValueError("ffn_multiplier * token_dim must be at least 1.")
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens=num_tokens,
                           token_dim=token_dim,
                           hidden_dim=hidden_dim,
                           use_sparse_moe=use_sparse_moe,
                           num_experts=num_experts,
                           net_dropout=net_dropout,
                           use_residual=use_residual,
                           use_layer_norm=use_layer_norm)
            for _ in range(num_layers)
        ])

    def forward(self, tokens):
        """Encode tokens.

        Args:
            tokens (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            tuple: Encoded tokens and the summed MoE penalty, or ``None``.
        """
        reg = None
        for block in self.blocks:
            tokens, block_reg = block(tokens)
            if block_reg is not None:
                reg = block_reg if reg is None else reg + block_reg
        return tokens, reg


def build_rankmixer_stack(feature_map,
                          embedding_dim,
                          num_tokens,
                          token_dim,
                          num_layers,
                          ffn_multiplier=4,
                          token_grouping="sequential",
                          feature_groups=None,
                          use_sparse_moe=False,
                          num_experts=2,
                          net_dropout=0.0,
                          use_residual=True,
                          use_layer_norm=True):
    """Build the tokenizer and the RankMixer stack from a feature map.

    Args:
        feature_map (FeatureMap): Dataset feature map.
        embedding_dim (int): Field embedding size.
        num_tokens (int or None): Token count.
        token_dim (int): Token dimension ``D``.
        num_layers (int): Number of blocks.
        ffn_multiplier (float): Per-token FFN expansion ratio ``k``.
        token_grouping (str): ``"sequential"`` or ``"semantic"``.
        feature_groups (list or None): Semantic groups.
        use_sparse_moe (bool): Enable the Sparse-MoE FFN.
        num_experts (int): Experts per token.
        net_dropout (float): Sublayer dropout.
        use_residual (bool): Residual connections.
        use_layer_norm (bool): LayerNorm.

    Returns:
        tuple: ``(tokenizer, encoder, num_tokens)``.
    """
    tokenizer, num_tokens = build_feature_tokenizer(
        feature_map, embedding_dim, num_tokens, token_dim, token_grouping, feature_groups)
    encoder = RankMixerEncoder(num_tokens=num_tokens,
                               token_dim=token_dim,
                               num_layers=num_layers,
                               ffn_multiplier=ffn_multiplier,
                               use_sparse_moe=use_sparse_moe,
                               num_experts=num_experts,
                               net_dropout=net_dropout,
                               use_residual=use_residual,
                               use_layer_norm=use_layer_norm)
    return tokenizer, encoder, num_tokens


class TaskTokenGate(nn.Module):
    """Per-task softmax or sigmoid gate over RankMixer output tokens.

    This module is not part of the RankMixer paper. The paper mean-pools tokens
    once and shares that vector across tasks. Here task ``k`` scores each token
    with its own linear map and mixes the tokens:

        alpha_{k,t} = softmax_t(w_k^T x_t + b_k)
        h_k = sum_t alpha_{k,t} x_t

    ``gate_type="sigmoid"`` applies a sigmoid and then L1-normalizes over tokens
    so the mixture stays on the same scale and the gate still sums to one.

    Args:
        num_tasks (int): Number of tasks.
        num_tokens (int): Number of tokens.
        token_dim (int): Token dimension.
        gate_type (str): ``"softmax"`` or ``"sigmoid"``. Default: ``"softmax"``.
    """
    def __init__(self, num_tasks, num_tokens, token_dim, gate_type="softmax"):
        super(TaskTokenGate, self).__init__()
        if gate_type not in ("softmax", "sigmoid"):
            raise ValueError("gate_type must be 'softmax' or 'sigmoid'.")
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive.")
        self.num_tasks = num_tasks
        self.num_tokens = num_tokens
        self.gate_type = gate_type
        self.score = nn.ModuleList([nn.Linear(token_dim, 1) for _ in range(num_tasks)])

    def forward(self, tokens):
        """Mix tokens with a separate gate for every task.

        Args:
            tokens (torch.Tensor): Shape ``(batch, num_tokens, token_dim)``.

        Returns:
            tuple: A list of task vectors ``(batch, token_dim)`` and a list of
            gate tensors ``(batch, num_tokens)``.
        """
        mixed, gates = [], []
        for scorer in self.score:
            logits = scorer(tokens).squeeze(-1)
            if self.gate_type == "softmax":
                gate = torch.softmax(logits, dim=-1)
            else:
                gate = torch.sigmoid(logits)
                gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            gates.append(gate)
            mixed.append(torch.einsum("bt,btd->bd", gate, tokens))
        return mixed, gates
