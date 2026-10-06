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

import importlib.util
import os
import sys
import unittest
from collections import OrderedDict

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fuxictr.features import FeatureMap
from fuxictr.pytorch.layers.interactions.rankmixer import (
    FeatureTokenizer,
    MultiHeadTokenMixing,
    PerTokenFFN,
    PerTokenSparseMoE,
    RankMixerBlock,
    TaskTokenGate,
    build_feature_tokenizer,
    resolve_feature_groups,
    split_sequential_chunks,
)


def _load_class(module_name, path, class_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


RankMixer = _load_class(
    "rankmixer_model",
    os.path.join(REPO_ROOT, "model_zoo", "RankMixer", "src", "RankMixer.py"),
    "RankMixer")
MTRankMixer = _load_class(
    "mt_rankmixer_model",
    os.path.join(REPO_ROOT, "model_zoo", "multitask", "MT_RankMixer", "src", "MTRankMixer.py"),
    "MTRankMixer")


def reference_token_mixing(x):
    """Literal reading of Eq. (3)-(4): concat the h-th head across tokens."""
    batch, num_tokens, token_dim = x.shape
    head_dim = token_dim // num_tokens
    heads = x.view(batch, num_tokens, num_tokens, head_dim)
    mixed = []
    for h in range(num_tokens):
        mixed.append(torch.cat([heads[:, t, h, :] for t in range(num_tokens)], dim=-1))
    return torch.stack(mixed, dim=1)


def make_feature_map(labels, features):
    feature_map = FeatureMap("toy", "/tmp")
    feature_map.labels = list(labels)
    feature_map.features = OrderedDict(features)
    feature_map.num_fields = feature_map.get_num_fields()
    feature_map.default_emb_dim = 4
    return feature_map


TOY_FEATURES = OrderedDict([
    ("user_id", {"source": "user", "type": "categorical", "vocab_size": 10, "padding_idx": 0}),
    ("age", {"source": "user", "type": "categorical", "vocab_size": 5, "padding_idx": 0}),
    ("item_id", {"source": "item", "type": "categorical", "vocab_size": 12, "padding_idx": 0}),
    ("price", {"source": "item", "type": "numeric"}),
])


class TestTokenMixingAndFFN(unittest.TestCase):
    def test_token_mixing_matches_formula_and_has_no_parameters(self):
        mixing = MultiHeadTokenMixing(num_tokens=2, token_dim=4)
        self.assertEqual(sum(p.numel() for p in mixing.parameters()), 0)
        x = torch.tensor([[[1., 2., 3., 4.],
                           [5., 6., 7., 8.]]])
        out = mixing(x)
        self.assertEqual(tuple(out.shape), (1, 2, 4))
        self.assertTrue(torch.equal(out, reference_token_mixing(x)))
        # s^0 = [x0_h0, x1_h0] = [1, 2, 5, 6]; s^1 = [3, 4, 7, 8]
        self.assertTrue(torch.equal(out[0, 0], torch.tensor([1., 2., 5., 6.])))
        self.assertTrue(torch.equal(out[0, 1], torch.tensor([3., 4., 7., 8.])))

    def test_token_dim_must_be_divisible_by_num_tokens(self):
        with self.assertRaises(ValueError):
            MultiHeadTokenMixing(num_tokens=3, token_dim=8)

    def test_per_token_ffn_parameters_are_independent(self):
        ffn = PerTokenFFN(num_tokens=2, token_dim=4, hidden_dim=8)
        weight_0 = ffn.ffns[0][0].weight
        weight_1 = ffn.ffns[1][0].weight
        self.assertNotEqual(weight_0.data_ptr(), weight_1.data_ptr())
        self.assertFalse(torch.allclose(weight_0, weight_1))
        saved = weight_1.detach().clone()
        with torch.no_grad():
            weight_0.fill_(0)
        self.assertTrue(torch.equal(weight_1, saved))
        x = torch.ones(3, 2, 4)
        y = ffn(x)
        self.assertEqual(tuple(y.shape), (3, 2, 4))
        self.assertFalse(torch.allclose(y[:, 0, :], y[:, 1, :]))

    def test_block_output_shape(self):
        block = RankMixerBlock(num_tokens=2, token_dim=4, hidden_dim=8)
        y, reg = block(torch.randn(5, 2, 4))
        self.assertEqual(tuple(y.shape), (5, 2, 4))
        self.assertIsNone(reg)


class TestSparseMoE(unittest.TestCase):
    def test_experts_are_untied_and_reg_backprops(self):
        moe = PerTokenSparseMoE(num_tokens=2, token_dim=4, hidden_dim=8, num_experts=2)
        e0 = moe.experts[0][0][0].weight
        e1 = moe.experts[1][0][0].weight
        self.assertNotEqual(e0.data_ptr(), e1.data_ptr())
        moe.train()
        out, reg = moe(torch.randn(4, 2, 4))
        self.assertEqual(tuple(out.shape), (4, 2, 4))
        self.assertEqual(tuple(reg.shape), ())
        (out.sum() + reg).backward()
        self.assertIsNotNone(moe.router_infer[0].bias.grad)
        self.assertIsNotNone(moe.router_train[0].weight.grad)
        self.assertIsNotNone(moe.experts[0][0][0].weight.grad)
        self.assertIsNotNone(moe.experts[1][1][0].weight.grad)


class TestGrouping(unittest.TestCase):
    def test_sequential_chunk_padding_keeps_prefix(self):
        flat = torch.arange(5, dtype=torch.float32).view(1, 5)
        chunks = split_sequential_chunks(flat, num_tokens=2)
        self.assertEqual(tuple(chunks.shape), (1, 2, 3))
        self.assertTrue(torch.equal(chunks[0, 0], torch.tensor([0., 1., 2.])))
        self.assertTrue(torch.equal(chunks[0, 1], torch.tensor([3., 4., 0.])))

    def test_semantic_tokenizer_groups_do_not_leak(self):
        groups = [[0, 1], [2, 3]]
        tokenizer = FeatureTokenizer(num_fields=4, embedding_dim=2, num_tokens=2, token_dim=4,
                                     token_grouping="semantic", group_indices=groups)
        with torch.no_grad():
            for proj in tokenizer.projs:
                proj.bias.zero_()
                proj.weight.fill_(0.1)
        emb = torch.zeros(2, 4, 2)
        emb[:, 0, :] = 1
        token_a = tokenizer(emb).detach().clone()
        emb[:, 3, :] = 5
        token_b = tokenizer(emb)
        self.assertTrue(torch.allclose(token_a[:, 0, :], token_b[:, 0, :]))
        self.assertFalse(torch.allclose(token_a[:, 1, :], token_b[:, 1, :]))

    def test_resolve_feature_groups_from_yaml_like_config(self):
        parsed = yaml.safe_load("""
        feature_groups:
          - [user_id, age]
          - item
          - [101]
        """)
        names = ["user_id", "age", "item_id", "101"]
        sources = ["user", "user", "item", "context"]
        groups = resolve_feature_groups(names, sources, parsed["feature_groups"], num_tokens=3)
        self.assertEqual(groups, [[0, 1], [2], [3]])
        # YAML may load an id as an int. str() must still match the column name.
        groups = resolve_feature_groups(["101", "205"], ["user", "item"], [[101], ["205"]])
        self.assertEqual(groups, [[0], [1]])

    def test_source_groups_and_rejections(self):
        names = ["user_id", "age", "item_id"]
        sources = ["user", "user", "item"]
        self.assertEqual(
            resolve_feature_groups(names, sources, ["user", "item"]),
            [[0, 1], [2]])
        with self.assertRaises(ValueError):
            resolve_feature_groups(names, sources, [["missing"]])
        with self.assertRaises(ValueError):
            resolve_feature_groups(names, sources, [["user_id", "user_id"], ["age"], ["item_id"]])
        with self.assertRaises(ValueError):
            resolve_feature_groups(names, sources, [["user_id"], ["user_id", "age", "item_id"]])
        with self.assertRaises(ValueError):
            resolve_feature_groups(names, sources, [["user_id", "age"], ["item_id"]], num_tokens=3)
        with self.assertRaises(ValueError):
            resolve_feature_groups(names, sources, [["user_id", "age"]])

    def test_build_feature_tokenizer_rejects_groups_in_sequential_mode(self):
        feature_map = make_feature_map(["clk"], TOY_FEATURES)
        with self.assertRaises(ValueError):
            build_feature_tokenizer(feature_map, 4, 2, 8, "sequential", feature_groups=[["user_id"]])
        tokenizer, num_tokens = build_feature_tokenizer(
            feature_map, 4, None, 8, "semantic", feature_groups=["user", "item"])
        self.assertEqual(num_tokens, 2)
        out = tokenizer(torch.randn(3, 4, 4))
        self.assertEqual(tuple(out.shape), (3, 2, 8))


class TestTaskGate(unittest.TestCase):
    def test_gate_weights_sum_to_one(self):
        tokens = torch.randn(6, 4, 8)
        for gate_type in ("softmax", "sigmoid"):
            gate = TaskTokenGate(num_tasks=2, num_tokens=4, token_dim=8, gate_type=gate_type)
            mixed, gates = gate(tokens)
            self.assertEqual(len(mixed), 2)
            self.assertEqual(tuple(mixed[0].shape), (6, 8))
            for weights in gates:
                self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(6), atol=1e-5))
                self.assertTrue(torch.all(weights >= 0))


class TestModels(unittest.TestCase):
    def _batch(self):
        return {
            "user_id": torch.randint(1, 10, (3,)),
            "age": torch.randint(1, 5, (3,)),
            "item_id": torch.randint(1, 12, (3,)),
            "price": torch.randn(3),
        }

    def test_rankmixer_output_shape_and_ffn_independence(self):
        feature_map = make_feature_map(["clk"], TOY_FEATURES)
        model = RankMixer(feature_map,
                          model_root="/tmp/rankmixer_unit",
                          metrics=["AUC"],
                          verbose=0,
                          optimizer="adam",
                          loss="binary_crossentropy",
                          learning_rate=1e-3,
                          gpu=-1,
                          embedding_dim=4,
                          num_tokens=2,
                          token_dim=8,
                          num_layers=1,
                          ffn_multiplier=2,
                          token_grouping="sequential")
        out = model(self._batch())
        self.assertEqual(tuple(out["y_pred"].shape), (3, 1))
        ffn = model.encoder.blocks[0].pffn
        self.assertNotEqual(ffn.ffns[0][0].weight.data_ptr(), ffn.ffns[1][0].weight.data_ptr())
        y = torch.randint(0, 2, (3, 1)).float()
        loss = model.compute_loss(out, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())

    def test_moe_rankmixer_adds_penalty(self):
        feature_map = make_feature_map(["clk"], TOY_FEATURES)
        model = RankMixer(feature_map,
                          model_root="/tmp/rankmixer_moe_unit",
                          metrics=["AUC"],
                          verbose=0,
                          optimizer="adam",
                          loss="binary_crossentropy",
                          learning_rate=1e-3,
                          gpu=-1,
                          embedding_dim=4,
                          num_tokens=2,
                          token_dim=8,
                          num_layers=1,
                          ffn_multiplier=2,
                          use_sparse_moe=True,
                          num_experts=2,
                          moe_lambda=1.0)
        model.train()
        batch = self._batch()
        out = model(batch)
        self.assertIn("moe_reg", out)
        y = torch.zeros(3, 1)
        total = model.compute_loss(out, y)
        task = model.loss_fn(out["y_pred"], y, reduction="mean")
        self.assertGreater(total.item(), task.item())
        total.backward()
        self.assertIsNotNone(model.encoder.blocks[0].pffn.router_infer[0].weight.grad)

    def test_mt_rankmixer_shapes_and_semantic_groups(self):
        feature_map = make_feature_map(["click", "conversion"], TOY_FEATURES)
        model = MTRankMixer(feature_map,
                            model_root="/tmp/mt_rankmixer_unit",
                            metrics=["AUC"],
                            verbose=0,
                            optimizer="adam",
                            loss=["binary_crossentropy", "binary_crossentropy"],
                            task=["binary_classification", "binary_classification"],
                            num_tasks=2,
                            learning_rate=1e-3,
                            gpu=-1,
                            embedding_dim=4,
                            num_tokens=2,
                            token_dim=8,
                            num_layers=1,
                            ffn_multiplier=2,
                            token_grouping="semantic",
                            feature_groups=["user", "item"],
                            gate_type="softmax",
                            tower_hidden_units=[4])
        out = model(self._batch())
        self.assertEqual(tuple(out["click_pred"].shape), (3, 1))
        self.assertEqual(tuple(out["conversion_pred"].shape), (3, 1))
        tokens = model.tokenizer(model.embedding_layer(model.get_inputs(self._batch())))
        _, gates = model.task_gate(tokens)
        for weights in gates:
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-5))
        y = [torch.randint(0, 2, (3, 1)).float(), torch.randint(0, 2, (3, 1)).float()]
        loss = model.compute_loss(out, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())

    def _mt_model(self, **overrides):
        feature_map = make_feature_map(["click", "conversion"], TOY_FEATURES)
        kwargs = dict(feature_map=feature_map,
                      model_root="/tmp/mt_rankmixer_unit",
                      metrics=["AUC"],
                      verbose=0,
                      optimizer="adam",
                      loss=["binary_crossentropy", "binary_crossentropy"],
                      task=["binary_classification", "binary_classification"],
                      num_tasks=2,
                      learning_rate=1e-3,
                      gpu=-1,
                      embedding_dim=4,
                      num_tokens=2,
                      token_dim=8,
                      num_layers=1,
                      ffn_multiplier=2,
                      token_grouping="semantic",
                      feature_groups=["user", "item"],
                      tower_hidden_units=[4])
        kwargs.update(overrides)
        return MTRankMixer(**kwargs)

    def test_shared_mean_pooling_drops_the_gate_and_shares_the_vector(self):
        for overrides in ({"task_pooling": "mean"}, {"gate_type": "mean"}):
            model = self._mt_model(**overrides)
            self.assertEqual(model.task_pooling, "mean")
            self.assertIsNone(model.task_gate)
            tokens = torch.randn(3, 2, 8)
            mixed, gates = model.mix_tokens(tokens)
            self.assertIsNone(gates)
            expected = tokens.mean(dim=1)
            self.assertTrue(torch.allclose(mixed[0], expected))
            self.assertTrue(torch.allclose(mixed[1], expected))
            self.assertEqual(mixed[0].data_ptr(), mixed[1].data_ptr())
            out = model(self._batch())
            self.assertEqual(tuple(out["click_pred"].shape), (3, 1))
            self.assertEqual(tuple(out["conversion_pred"].shape), (3, 1))
            y = [torch.zeros(3, 1), torch.ones(3, 1)]
            loss = model.compute_loss(out, y)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())

    def test_mean_pooling_removes_only_the_gate_parameters(self):
        gated = self._mt_model(gate_type="softmax", task_pooling="gate")
        shared = self._mt_model(task_pooling="mean", gate_type="softmax")
        n_gated = sum(p.numel() for p in gated.parameters())
        n_shared = sum(p.numel() for p in shared.parameters())
        # Two Linear(token_dim, 1) scorers, weight plus bias.
        self.assertEqual(n_gated - n_shared, 2 * (8 + 1))

    def test_task_pooling_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            self._mt_model(task_pooling="attn")
        with self.assertRaises(ValueError):
            self._mt_model(gate_type="relu")

    def test_gate_summary_slices_and_log_parser(self):
        import importlib.util
        import numpy as np

        def load(module_name, relative):
            path = os.path.join(REPO_ROOT, relative)
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        gates = load("analyze_gates", "benchmarks/rankmixer/analyze_gates.py")
        summary = load("summarize_multiseed", "benchmarks/rankmixer/summarize_multiseed.py")
        weights = np.array([
            [[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]],
            [[0.1, 0.2, 0.7], [0.4, 0.4, 0.2]],
        ])
        rows = gates.summarize_gates(
            weights, np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            ["user", "item", "context"], ["click", "conversion"])
        by_key = {(row["slice"], row["task"]): row for row in rows}
        self.assertAlmostEqual(by_key[("all", "click")]["user"]["mean"], 0.3)
        self.assertAlmostEqual(by_key[("click=1", "click")]["user"]["mean"], 0.5)
        self.assertIsNone(by_key[("click=1", "click")]["user"]["std"])
        self.assertEqual(by_key[("conversion=1", "conversion")]["n"], 1)
        self.assertAlmostEqual(by_key[("conversion=1", "conversion")]["context"]["mean"], 0.2)
        text = gates.render_markdown(rows, ["user", "item", "context"])
        self.assertIn("click=1", text)

        log = """
Evaluation @epoch 1 - batch 4:
Save best model: monitor(max)=0.610000
Evaluation @epoch 2 - batch 4:
Monitor(max)=0.600000 STOP!
******** Test evaluation ********
[Task: click][Metrics] logloss: 0.160000 - AUC: 0.620000
[Task: conversion][Metrics] logloss: 0.002000 - AUC: 0.640000
"""
        best_epoch, metrics = summary.parse_log(log)
        self.assertEqual(best_epoch, "1")
        self.assertAlmostEqual(metrics["click_auc"], 0.62)
        self.assertAlmostEqual(metrics["conversion_auc"], 0.64)


if __name__ == "__main__":
    unittest.main()
