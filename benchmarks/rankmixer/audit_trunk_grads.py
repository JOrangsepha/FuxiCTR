#!/usr/bin/env python3
"""Trunk gradient norms of click BCE and conversion BCE.

``loss_weight: EQ`` is the unnormalized sum of the two mean binary cross-entropies.
On Ali-CCP the click logloss is about 0.16 and the conversion logloss is about
0.002, so the click term can dominate the shared trunk update. This script
measures ``||∇_trunk L_click||`` and ``||∇_trunk L_conv||`` on a few batches.
The trunk is the shared embedding, tokenizer, and RankMixer encoder. Task towers,
the token gate, and residual λ are not part of the trunk.

Training defaults stay ``EQ``. ``MultiTaskModel`` also accepts a list of manual
weights, or ``NORM`` (each task loss divided by its detached absolute value, then
summed). Those hooks are not turned on by the rigor suite. This script does not
change weights unless ``--train_steps`` is positive.

The ``--demo`` path builds a tiny random model and labels the table synthetic.
Those numbers are not an Ali-CCP result.

Example, after a rigor checkpoint exists::

    python benchmarks/rankmixer/audit_trunk_grads.py --gpu 0 --stage valid --batches 4 \\
        --checkpoint "$CKPT" --config "$CFG" --expid MTR_rg_g6_mean_s2025 \\
        --output /root/autodl-tmp/rg_out/grad_audit.md
"""

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_TRUNK_PREFIXES = ("embedding_layer.", "tokenizer.", "encoder.")


def is_trunk_parameter(name):
    """Shared RankMixer trunk. Towers, gates, and λ are excluded."""
    return name.startswith(_TRUNK_PREFIXES)


def grad_norm_of(model, predicate):
    """L2 norm of ``param.grad`` for parameters selected by ``predicate``."""
    squares = []
    for name, param in model.named_parameters():
        if param.grad is None or not predicate(name):
            continue
        squares.append(param.grad.detach().float().pow(2).sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()))


def per_task_trunk_grad_norms(model, batch):
    """Separate trunk gradients of each task BCE, plus the EQ sum.

    Does not step the optimizer. The graph is evaluated in eval mode so dropout
    does not add noise. Ali-CCP configs set dropout to 0 anyway.

    Returns:
        dict: ``tasks`` is one record per label, and ``eq_loss`` / ``eq_trunk_grad_norm``
        describe the unnormalized sum that ``EQ`` actually differentiates.
    """
    was_training = model.training
    model.eval()
    model.zero_grad()
    output = model(batch)
    y_true = model.get_labels(batch)
    labels = list(model.feature_map.labels)
    losses = []
    for index, label in enumerate(labels):
        pred = output["{}_pred".format(label)]
        losses.append(model.loss_fn[index](pred, y_true[index], reduction="mean"))
    rows = []
    for index, label in enumerate(labels):
        model.zero_grad()
        losses[index].backward(retain_graph=True)
        rows.append({
            "task": label,
            "loss": float(losses[index].detach()),
            "trunk_grad_norm": grad_norm_of(model, is_trunk_parameter),
            "full_grad_norm": grad_norm_of(model, lambda name: True),
        })
    eq_loss = torch.stack(losses).sum()
    model.zero_grad()
    eq_loss.backward()
    report = {
        "tasks": rows,
        "eq_loss": float(eq_loss.detach()),
        "eq_trunk_grad_norm": grad_norm_of(model, is_trunk_parameter),
    }
    if was_training:
        model.train()
    return report


def _fmt(value):
    return "{:.6e}".format(float(value))


def render_grad_markdown(reports, stage, source_note, synthetic=False):
    """Markdown table. ``reports`` is a list of ``per_task_trunk_grad_norms`` dicts."""
    lines = [
        "# Trunk gradient norms",
        "",
    ]
    if synthetic:
        lines.append(
            "SYNTHETIC demo on a tiny random model. This is not an Ali-CCP result "
            "and must not be copied into the result tables.")
        lines.append("")
    lines.append(source_note)
    lines.append("")
    lines.append(
        "`EQ` is the unnormalized sum of per-task mean BCE. It does not divide by "
        "the number of tasks and it does not rescale by the loss magnitude. On "
        "Ali-CCP, click logloss is about 0.16 and conversion logloss is about 0.002, "
        "so click can dominate `∇_trunk`. The trunk is `embedding_layer`, `tokenizer`, "
        "and `encoder`. Towers, the token gate, and residual λ are outside it.")
    lines.append("")
    lines.append(
        "Optional hooks, not used by the rigor suite: `loss_weight` may be a list of "
        "manual weights, or `NORM` (each task loss divided by its detached absolute "
        "value, then summed). Training defaults stay `EQ`.")
    lines.append("")
    lines.append("Stage: `{}`.".format(stage))
    lines.append("")
    lines.append("| batch | task | mean BCE | trunk grad L2 | full grad L2 |")
    lines.append("| --- | --- | --- | --- | --- |")
    click_norms = []
    conv_norms = []
    for batch_index, report in enumerate(reports):
        for row in report["tasks"]:
            lines.append("| {} | {} | {} | {} | {} |".format(
                batch_index, row["task"], _fmt(row["loss"]),
                _fmt(row["trunk_grad_norm"]), _fmt(row["full_grad_norm"])))
            if row["task"] == "click":
                click_norms.append(row["trunk_grad_norm"])
            elif row["task"] == "conversion":
                conv_norms.append(row["trunk_grad_norm"])
        lines.append("| {} | EQ sum | {} | {} | |".format(
            batch_index, _fmt(report["eq_loss"]), _fmt(report["eq_trunk_grad_norm"])))
    if click_norms and conv_norms:
        click_mean = sum(click_norms) / len(click_norms)
        conv_mean = sum(conv_norms) / len(conv_norms)
        ratio = click_mean / conv_mean if conv_mean > 0 else float("inf")
        lines.extend([
            "",
            "Mean trunk grad L2 over {} batch(es): click {}, conversion {}, "
            "click/conversion ratio {:.4f}.".format(
                len(reports), _fmt(click_mean), _fmt(conv_mean), ratio),
        ])
    lines.append("")
    return "\n".join(lines)


def _take_batches(generator, count):
    batches = []
    for batch in generator:
        batches.append(batch)
        if len(batches) >= count:
            break
    return batches


def _demo_model_and_batch():
    """Tiny CPU model so the script can be smoked without Ali-CCP."""
    from collections import OrderedDict

    from fuxictr.features import FeatureMap
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec_path = repo_root / "model_zoo" / "multitask" / "MT_RankMixer" / "src" / "MTRankMixer.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("mt_rankmixer_audit_demo", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    feature_map = FeatureMap("toy", "/tmp")
    feature_map.labels = ["click", "conversion"]
    feature_map.features = OrderedDict([
        ("user_id", {"source": "user", "type": "categorical", "vocab_size": 10, "padding_idx": 0}),
        ("age", {"source": "user", "type": "categorical", "vocab_size": 5, "padding_idx": 0}),
        ("item_id", {"source": "item", "type": "categorical", "vocab_size": 12, "padding_idx": 0}),
        ("price", {"source": "item", "type": "numeric"}),
    ])
    feature_map.num_fields = feature_map.get_num_fields()
    model = module.MTRankMixer(
        feature_map,
        model_root="/tmp/mt_rankmixer_grad_audit",
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
    batch = {
        "user_id": torch.randint(1, 10, (8,)),
        "age": torch.randint(1, 5, (8,)),
        "item_id": torch.randint(1, 12, (8,)),
        "price": torch.randn(8),
        "click": torch.randint(0, 2, (8, 1)).float(),
        "conversion": torch.randint(0, 2, (8, 1)).float(),
    }
    return model, batch


def _load_trained(repo_root, args):
    import exp_loader
    exp_loader.bootstrap_workdir(repo_root, args.workdir)
    import src as model_zoo
    params, feature_map = exp_loader.prepare_params(args.config, args.expid, args.gpu)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    if args.checkpoint:
        model.load_weights(args.checkpoint)
    return model, params, feature_map


def ensure_trainable(model, params):
    """Give ``train_step`` the optimizer and clip budget that ``fit`` normally sets.

    ``--train_steps`` used to call ``train_step`` on a freshly built model.
    ``compile`` is what creates ``optimizer`` (and, on ``MultiTaskModel``, the
    per-task loss list). ``_max_gradient_norm`` is set inside ``fit``, so a
    call before ``fit`` raised ``AttributeError``.
    """
    if not hasattr(model, "optimizer"):
        model.compile(params["optimizer"], params["loss"], params["learning_rate"])
    if getattr(model, "_max_gradient_norm", None) in (None, 0):
        model._max_gradient_norm = float(params.get("max_gradient_norm", 10.0))
    return model


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Audit per-task trunk gradient norms.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--stage", choices=("valid", "test"), default="valid")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--expid", type=str, default="")
    parser.add_argument("--train_steps", type=int, default=0,
                        help="Optional short training on the train split before the audit.")
    parser.add_argument("--demo", action="store_true",
                        help="Tiny synthetic model. Not an Ali-CCP result.")
    parser.add_argument("--output", type=str,
                        default=str(repo_root / "benchmarks" / "rankmixer" / "grad_audit.md"))
    parser.add_argument("--workdir", type=str,
                        default=str(repo_root / "model_zoo" / "multitask" / "MT_RankMixer"))
    args = parser.parse_args()
    if args.stage == "test" and not args.demo:
        print("WARNING: --stage test reads the held-out test split. "
              "The gradient audit for model selection belongs on validation.", file=sys.stderr)

    if args.demo:
        model, batch = _demo_model_and_batch()
        reports = [per_task_trunk_grad_norms(model, batch)]
        body = render_grad_markdown(
            reports, "synthetic",
            "Source: `--demo` tiny MT-RankMixer, one random batch.",
            synthetic=True)
    else:
        if not args.config or not args.expid:
            raise SystemExit("Pass --config and --expid, or pass --demo for a synthetic smoke.")
        if not args.checkpoint and args.train_steps <= 0:
            raise SystemExit(
                "Pass --checkpoint of a trained model, or --train_steps > 0 to take a few "
                "optimizer steps before measuring. Refusing to invent Ali-CCP numbers.")
        model, params, feature_map = _load_trained(repo_root, args)
        import exp_loader
        if args.train_steps > 0:
            from fuxictr.pytorch.dataloaders import RankDataLoader
            ensure_trainable(model, params)
            train_gen, _valid_gen = RankDataLoader(feature_map, stage="train", **params)
            model.train()
            taken = 0
            for batch in train_gen:
                loss = model.train_step(batch)
                taken += 1
                if taken >= args.train_steps:
                    break
            source = "Loaded config `{}`. Took {} optimizer step(s) on the train split before measuring.".format(
                args.expid, taken)
        else:
            source = "Checkpoint `{}`. Weights were not updated.".format(args.checkpoint)
        data_gen = exp_loader.open_stage(feature_map, params, args.stage)
        batches = _take_batches(data_gen, args.batches)
        if not batches:
            raise SystemExit("{} split produced no batches.".format(args.stage))
        reports = [per_task_trunk_grad_norms(model, batch) for batch in batches]
        body = render_grad_markdown(reports, args.stage, source, synthetic=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)
    print(body)
    print("wrote {}".format(output))


if __name__ == "__main__":
    main()
