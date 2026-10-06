#!/usr/bin/env python3
"""Average per-task token gates of a trained semantic MT-RankMixer.

Loads a checkpoint, runs the test split (optionally capped, default 500k rows),
and reports the mean and sample standard deviation of the gate on the user,
item, and context tokens. Slices: all rows, click = 1, conversion = 1.

The shared mean-pool ablation has no gate. This script exits if the loaded
model was built with ``task_pooling: mean``. Residual pooling
(``task_pooling: residual``) still has a gate, so it is included, and the
learned per-task mix weight λ_k is written next to the table. Every run also
reports the mean gate entropy in nats.

Example (after run_multiseed.sh, from the repo root on the GPU machine)::

    python benchmarks/rankmixer/analyze_gates.py --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def summarize_gates(weights, click, conversion, token_names, task_names):
    """Mean and sample std of gate weights, overall and on positive slices.

    Args:
        weights (np.ndarray): Shape ``(n, num_tasks, num_tokens)``.
        click (np.ndarray): Shape ``(n,)``. Positive rows are ``click == 1``.
        conversion (np.ndarray): Shape ``(n,)``. Positive rows are ``conversion == 1``.
        token_names (list): Length ``num_tokens``.
        task_names (list): Length ``num_tasks``.

    Returns:
        list: One dict per slice and task, with ``n`` and per-token ``mean`` / ``std``.
        Std is the sample standard deviation. It is ``None`` when ``n < 2``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    click = np.asarray(click).reshape(-1)
    conversion = np.asarray(conversion).reshape(-1)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (n, tasks, tokens).")
    n, num_tasks, num_tokens = weights.shape
    if click.shape[0] != n or conversion.shape[0] != n:
        raise ValueError("click and conversion must have one value per row.")
    if len(token_names) != num_tokens or len(task_names) != num_tasks:
        raise ValueError("token_names and task_names must match weights.")

    slices = (
        ("all", np.ones(n, dtype=bool)),
        ("click=1", click == 1),
        ("conversion=1", conversion == 1),
    )
    rows = []
    for slice_name, mask in slices:
        chosen = weights[mask]
        count = int(chosen.shape[0])
        for task_index, task_name in enumerate(task_names):
            record = {"slice": slice_name, "task": task_name, "n": count}
            if count == 0:
                for name in token_names:
                    record[name] = {"mean": None, "std": None}
            else:
                task_weights = chosen[:, task_index, :]
                mean = task_weights.mean(axis=0)
                if count < 2:
                    std = np.full(num_tokens, np.nan)
                else:
                    std = task_weights.std(axis=0, ddof=1)
                for token_index, name in enumerate(token_names):
                    std_value = float(std[token_index])
                    record[name] = {
                        "mean": float(mean[token_index]),
                        "std": None if np.isnan(std_value) else std_value,
                    }
            rows.append(record)
    return rows


def render_markdown(rows, token_names):
    """Render ``summarize_gates`` rows as a markdown table."""
    header = ["slice", "task", "n"]
    for name in token_names:
        header.append("{} mean".format(name))
        header.append("{} std".format(name))
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for record in rows:
        cells = [record["slice"], record["task"], str(record["n"])]
        for name in token_names:
            stats = record[name]
            if stats["mean"] is None:
                cells.extend(["", ""])
            else:
                cells.append("{:.6f}".format(stats["mean"]))
                if stats["std"] is None:
                    cells.append("")
                else:
                    cells.append("{:.6f}".format(stats["std"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def mean_gate_entropy(weights):
    """Mean entropy, in nats, of each task gate.

    Args:
        weights (np.ndarray): Shape ``(n, num_tasks, num_tokens)``.

    Returns:
        np.ndarray: Shape ``(num_tasks,)``.
    """
    probs = np.clip(np.asarray(weights, dtype=np.float64), 1e-8, 1.0)
    return (-(probs * np.log(probs)).sum(axis=-1)).mean(axis=0)


def render_entropy(weights, click, conversion, task_names):
    """Markdown table of mean gate entropy on the same slices as the gate table."""
    weights = np.asarray(weights, dtype=np.float64)
    click = np.asarray(click).reshape(-1)
    conversion = np.asarray(conversion).reshape(-1)
    slices = (
        ("all", np.ones(weights.shape[0], dtype=bool)),
        ("click=1", click == 1),
        ("conversion=1", conversion == 1),
    )
    lines = [
        "## Mean gate entropy (nats)",
        "",
        "| slice | task | n | entropy |",
        "| --- | --- | --- | --- |",
    ]
    for slice_name, mask in slices:
        chosen = weights[mask]
        count = int(chosen.shape[0])
        if count == 0:
            entropy = [None] * len(task_names)
        else:
            entropy = mean_gate_entropy(chosen)
        for task_index, task_name in enumerate(task_names):
            cell = "" if entropy[task_index] is None else "{:.6f}".format(float(entropy[task_index]))
            lines.append("| {} | {} | {} | {} |".format(slice_name, task_name, count, cell))
    return "\n".join(lines) + "\n"


def render_lambdas(task_names, lambdas):
    """Markdown table of residual mix weights. ``lambdas`` is shape ``(num_tasks,)``."""
    lines = [
        "## Residual mix weight λ",
        "",
        "h = λ * mean(tokens) + (1 - λ) * gated mix. λ is a learned sigmoid scalar per task.",
        "",
        "| task | lambda |",
        "| --- | --- |",
    ]
    for task_name, value in zip(task_names, lambdas):
        lines.append("| {} | {:.6f} |".format(task_name, float(value)))
    return "\n".join(lines) + "\n"


def plot_gates(rows, token_names, figure_path):
    """Grouped bars of mean gate weight. One panel per slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices = []
    for record in rows:
        if record["slice"] not in slices:
            slices.append(record["slice"])
    tasks = []
    for record in rows:
        if record["task"] not in tasks:
            tasks.append(record["task"])
    figure, axes = plt.subplots(1, len(slices), figsize=(4.2 * len(slices), 4.0), sharey=True)
    if len(slices) == 1:
        axes = [axes]
    x = np.arange(len(token_names))
    width = 0.8 / max(len(tasks), 1)
    for axis, slice_name in zip(axes, slices):
        for task_index, task_name in enumerate(tasks):
            record = next(row for row in rows if row["slice"] == slice_name and row["task"] == task_name)
            means, stds = [], []
            for name in token_names:
                stats = record[name]
                means.append(0.0 if stats["mean"] is None else stats["mean"])
                stds.append(0.0 if not stats["std"] else stats["std"])
            offset = (task_index - (len(tasks) - 1) / 2.0) * width
            axis.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=task_name)
        axis.set_xticks(x)
        axis.set_xticklabels(token_names)
        axis.set_title("{} (n={})".format(slice_name, record["n"]))
        axis.set_ylabel("gate weight")
        axis.set_ylim(0.0, 1.0)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def _slice_batch(batch, take):
    import torch
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[:take]
        else:
            sliced[key] = value
    return sliced


def collect_gates(model, data_generator, max_samples):
    """Run the test generator and stack gate weights up to ``max_samples``."""
    import torch

    if model.task_gate is None:
        raise SystemExit(
            "This checkpoint uses shared mean pooling and has no per-task gate. "
            "Load a task_pooling=gate or task_pooling=residual checkpoint instead.")
    labels = model.feature_map.labels
    weight_parts = []
    click_parts = []
    conversion_parts = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch in data_generator:
            if seen >= max_samples:
                break
            remaining = max_samples - seen
            batch_size = int(batch[labels[0]].shape[0])
            take = min(remaining, batch_size)
            sliced = _slice_batch(batch, take)
            features = model.get_inputs(sliced)
            tokens = model.tokenizer(model.embedding_layer(features))
            tokens, _ = model.encoder(tokens)
            _, gates = model.mix_tokens(tokens)
            stacked = torch.stack(gates, dim=1).detach().cpu().numpy()
            weight_parts.append(stacked)
            click_parts.append(sliced[labels[0]].detach().cpu().numpy().reshape(-1))
            conversion_parts.append(sliced[labels[1]].detach().cpu().numpy().reshape(-1))
            seen += take
    if seen == 0:
        raise SystemExit("Test generator produced no rows.")
    return (np.concatenate(weight_parts, axis=0),
            np.concatenate(click_parts, axis=0),
            np.concatenate(conversion_parts, axis=0))


def prepare_params(config_dir, expid, gpu):
    """Load config the same way ``run_expid.py`` does, without training."""
    from fuxictr.utils import load_config
    from fuxictr.features import FeatureMap

    params = load_config(config_dir, expid)
    params["gpu"] = gpu
    if params["data_format"] == "csv":
        from fuxictr.preprocess import FeatureProcessor, build_dataset
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    return params, feature_map


def default_paths(repo_root):
    checkpoint = (repo_root / "model_zoo" / "multitask" / "MT_RankMixer" / "checkpoints"
                  / "AliCCP_x1" / "MTRankMixer_aliccp_semantic_es_s2025.model")
    config = repo_root / "benchmarks" / "rankmixer" / "multiseed_runs" / "configs" \
        / "MTRankMixer_aliccp_semantic_es_s2025"
    template = repo_root / "benchmarks" / "rankmixer" / "configs" / "multiseed"
    return checkpoint, config, template


def main():
    repo_root = Path(__file__).resolve().parents[2]
    default_checkpoint, default_config, template_config = default_paths(repo_root)
    parser = argparse.ArgumentParser(description="Summarize MT-RankMixer token gates on the test set.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=500000)
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--expid", type=str, default="")
    parser.add_argument("--token_names", type=str, default="")
    parser.add_argument("--output", type=str, default=str(repo_root / "benchmarks" / "rankmixer" / "gate_analysis.md"))
    parser.add_argument("--figure", type=str, default=str(repo_root / "benchmarks" / "rankmixer" / "gate_weights.png"))
    parser.add_argument("--workdir", type=str,
                        default=str(repo_root / "model_zoo" / "multitask" / "MT_RankMixer"))
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    figure = Path(args.figure).resolve()
    if args.config:
        config_dir = str(Path(args.config).resolve())
        expid = args.expid or "MTRankMixer_aliccp_semantic_es"
    elif default_config.is_dir():
        config_dir = str(default_config)
        expid = args.expid or "MTRankMixer_aliccp_semantic_es_s2025"
    else:
        config_dir = str(template_config)
        expid = args.expid or "MTRankMixer_aliccp_semantic_es"
    if not checkpoint.is_file():
        raise SystemExit("checkpoint not found: {}".format(checkpoint))

    os.chdir(args.workdir)
    if repo_root.as_posix() not in sys.path:
        sys.path.insert(0, repo_root.as_posix())
    sys.path.insert(0, args.workdir)

    from fuxictr.pytorch.dataloaders import RankDataLoader
    import src as model_zoo

    params, feature_map = prepare_params(config_dir, expid, args.gpu)
    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.load_weights(str(checkpoint))
    test_gen = RankDataLoader(feature_map, stage="test", **params)
    weights, click, conversion = collect_gates(model, test_gen, args.max_samples)

    num_tokens = weights.shape[-1]
    if args.token_names:
        token_names = [name.strip() for name in args.token_names.split(",") if name.strip()]
    elif num_tokens == 3:
        token_names = ["user", "item", "context"]
    else:
        token_names = ["token{}".format(index) for index in range(num_tokens)]
    task_names = list(feature_map.labels)
    rows = summarize_gates(weights, click, conversion, token_names, task_names)
    table = render_markdown(rows, token_names)
    entropy_table = render_entropy(weights, click, conversion, task_names)
    lambda_table = ""
    lambdas = model.mix_lambdas()
    if lambdas is not None:
        lambda_table = render_lambdas(task_names, lambdas.detach().cpu().numpy())
    note = (
        "# MT-RankMixer token gate weights\n\n"
        "Checkpoint: `{}`\n\n"
        "Rows used: {} (cap {}). task_pooling: `{}`. "
        "Sample standard deviation (n - 1). "
        "Gates sum to 1 over tokens inside each task.\n\n"
    ).format(checkpoint, int(weights.shape[0]), args.max_samples, model.task_pooling)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = note + table + "\n" + entropy_table + "\n" + lambda_table
    output.write_text(body)
    plot_gates(rows, token_names, figure)
    print(body)
    print("wrote {} and {}".format(output, figure))


if __name__ == "__main__":
    main()
