#!/usr/bin/env python3
"""Average per-task token gates of a trained MT-RankMixer.

Default stage is the validation split. Use validation for gate diagnosis and
for any decision about which variant to keep. ``--stage test`` reads the
held-out test split and is only for the frozen final report, or for
reproducing a development-study table that was already computed on test.
Do not pick a model from test-stage output.

The row cap defaults to 500000. Slices: all rows, click = 1, conversion = 1.
The shared mean-pool ablation has no gate. This script exits if the loaded
model was built with ``task_pooling: mean``. Residual pooling still has a
gate, and the learned per-task mix weight λ_k is written next to the table.
Every run also reports the mean gate entropy in nats.

Multi-seed reporting is ``aggregate_gates.py``. It averages the per-seed gate
means and prints mean ± sample standard deviation across seeds. That standard
deviation is not the within-seed row standard deviation this script prints
for a single checkpoint.

Example (validation, the new study)::

    python benchmarks/rankmixer/analyze_gates.py --gpu 0 --stage valid \\
        --checkpoint "$CKPT" --config "$CFG" --expid "$EXPID" --seed 2025

Example (frozen final report only)::

    python benchmarks/rankmixer/analyze_gates.py --gpu 0 --stage test \\
        --checkpoint "$CKPT" --config "$CFG" --expid "$EXPID"
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import exp_loader  # noqa: E402


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
    lines = [
        "## Mean gate entropy (nats)",
        "",
        "| slice | task | n | entropy |",
        "| --- | --- | --- | --- |",
    ]
    for record in entropy_rows(weights, click, conversion, task_names):
        cell = "" if record["entropy"] is None else "{:.6f}".format(record["entropy"])
        lines.append("| {} | {} | {} | {} |".format(
            record["slice"], record["task"], record["n"], cell))
    return "\n".join(lines) + "\n"


def resolve_stage(stage):
    """Return ``valid`` or ``test``. The default for a missing value is validation.

    Args:
        stage (str or None): Requested split.

    Returns:
        str: ``"valid"`` or ``"test"``.

    Raises:
        ValueError: If ``stage`` is not one of those two names.
    """
    chosen = "valid" if stage is None or str(stage).strip() == "" else str(stage).strip().lower()
    if chosen not in ("valid", "test"):
        raise ValueError("stage must be 'valid' or 'test'.")
    return chosen


def stage_note(stage):
    """One paragraph that says what this split is allowed to be used for."""
    if stage == "valid":
        return (
            "Stage: validation. Use these numbers for analysis and model selection. "
            "The test split stays unread until the design is frozen."
        )
    return (
        "Stage: test. This split is only for the frozen final report, or for "
        "reproducing a development-study table. Do not use these numbers to pick "
        "a model or to diagnose gates during development."
    )


def _mean_std(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def format_mean_std(mean, std):
    if mean is None:
        return ""
    if std is None:
        return "{:.6f}".format(mean)
    return "{:.6f} ± {:.6f}".format(mean, std)


def aggregate_seed_gates(payloads):
    """Mean ± sample std, across seeds, of per-seed gate means.

    Args:
        payloads (list): JSON objects written by this script, one checkpoint each.
            Each object has ``rows`` from ``summarize_gates`` and ``token_names``.

    Returns:
        dict: ``token_names``, ``n_seeds``, and ``rows`` keyed like ``summarize_gates``
        but each token maps to the across-seed ``mean`` and ``std``.
    """
    if not payloads:
        raise ValueError("payloads must contain at least one seed.")
    token_names = list(payloads[0]["token_names"])
    grouped = {}
    for payload in payloads:
        if list(payload["token_names"]) != token_names:
            raise ValueError("token_names differ across seed payloads.")
        for record in payload["rows"]:
            key = (record["slice"], record["task"])
            bucket = grouped.setdefault(key, {"n": [], "tokens": {name: [] for name in token_names}})
            bucket["n"].append(record["n"])
            for name in token_names:
                bucket["tokens"][name].append(record[name]["mean"])
    rows = []
    for (slice_name, task_name), bucket in grouped.items():
        record = {
            "slice": slice_name,
            "task": task_name,
            "n_seeds": len(payloads),
            "n_rows_mean": None if not bucket["n"] else float(sum(bucket["n"]) / len(bucket["n"])),
        }
        for name in token_names:
            mean, std = _mean_std(bucket["tokens"][name])
            record[name] = {"mean": mean, "std": std}
        rows.append(record)
    return {"token_names": token_names, "n_seeds": len(payloads), "rows": rows}


def render_aggregate_markdown(aggregated, token_names):
    """Markdown table of across-seed gate means."""
    header = ["slice", "task", "n_seeds"]
    for name in token_names:
        header.append("{} mean±std".format(name))
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for record in aggregated["rows"]:
        cells = [record["slice"], record["task"], str(record["n_seeds"])]
        for name in token_names:
            stats = record[name]
            cells.append(format_mean_std(stats["mean"], stats["std"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def entropy_rows(weights, click, conversion, task_names):
    """Per-slice mean gate entropy records. Entropy is ``None`` when the slice is empty."""
    weights = np.asarray(weights, dtype=np.float64)
    click = np.asarray(click).reshape(-1)
    conversion = np.asarray(conversion).reshape(-1)
    slices = (
        ("all", np.ones(weights.shape[0], dtype=bool)),
        ("click=1", click == 1),
        ("conversion=1", conversion == 1),
    )
    rows = []
    for slice_name, mask in slices:
        chosen = weights[mask]
        count = int(chosen.shape[0])
        if count == 0:
            entropy = [None] * len(task_names)
        else:
            entropy = mean_gate_entropy(chosen)
        for task_index, task_name in enumerate(task_names):
            value = entropy[task_index]
            rows.append({
                "slice": slice_name,
                "task": task_name,
                "n": count,
                "entropy": None if value is None else float(value),
            })
    return rows


def aggregate_entropy(payloads):
    """Across-seed mean ± std of the per-seed mean entropies."""
    grouped = {}
    for payload in payloads:
        for record in payload.get("entropy", []):
            key = (record["slice"], record["task"])
            grouped.setdefault(key, []).append(record.get("entropy"))
    rows = []
    for (slice_name, task_name), values in grouped.items():
        mean, std = _mean_std(values)
        rows.append({
            "slice": slice_name,
            "task": task_name,
            "n_seeds": len(payloads),
            "mean": mean,
            "std": std,
        })
    return rows


def aggregate_lambdas(payloads):
    """Across-seed mean ± std of residual λ. Empty if no payload has λ."""
    grouped = {}
    any_lambda = False
    for payload in payloads:
        lambdas = payload.get("lambdas") or []
        if lambdas:
            any_lambda = True
        for record in lambdas:
            grouped.setdefault(record["task"], []).append(record.get("lambda"))
    if not any_lambda:
        return []
    rows = []
    for task_name, values in grouped.items():
        mean, std = _mean_std(values)
        rows.append({"task": task_name, "n_seeds": len(payloads), "mean": mean, "std": std})
    return rows


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


def collect_gates(model, data_generator, max_samples, stage="valid"):
    """Run one split generator and stack gate weights up to ``max_samples``."""
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
        raise SystemExit("{} split produced no rows.".format(stage))
    return (np.concatenate(weight_parts, axis=0),
            np.concatenate(click_parts, axis=0),
            np.concatenate(conversion_parts, axis=0))


def prepare_params(config_dir, expid, gpu):
    """Load config the same way ``run_expid.py`` does, without training."""
    return exp_loader.prepare_params(config_dir, expid, gpu)


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
    parser = argparse.ArgumentParser(
        description="Summarize MT-RankMixer token gates. Default split is validation.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stage", choices=("valid", "test"), default="valid",
                        help="valid: analysis and selection (default). "
                             "test: frozen final report only.")
    parser.add_argument("--max_samples", type=int, default=500000)
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--expid", type=str, default="")
    parser.add_argument("--seed", type=str, default="",
                        help="Training seed, stored in the JSON sidecar for multi-seed aggregation.")
    parser.add_argument("--token_names", type=str, default="")
    parser.add_argument("--output", type=str, default=str(repo_root / "benchmarks" / "rankmixer" / "gate_analysis.md"))
    parser.add_argument("--figure", type=str, default=str(repo_root / "benchmarks" / "rankmixer" / "gate_weights.png"))
    parser.add_argument("--json", type=str, default="",
                        help="JSON sidecar. Default: the output path with a .json suffix.")
    parser.add_argument("--workdir", type=str,
                        default=str(repo_root / "model_zoo" / "multitask" / "MT_RankMixer"))
    args = parser.parse_args()
    stage = resolve_stage(args.stage)
    if stage == "test":
        print("WARNING: --stage test reads the held-out test split. "
              "Use it only for the frozen final report or to reproduce a development-study table. "
              "Model selection belongs on --stage valid.", file=sys.stderr)

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

    exp_loader.bootstrap_workdir(repo_root, args.workdir)
    import src as model_zoo

    params, feature_map = prepare_params(config_dir, expid, args.gpu)
    model_class = getattr(model_zoo, params["model"])
    model = model_class(feature_map, **params)
    model.load_weights(str(checkpoint))
    data_gen = exp_loader.open_stage(feature_map, params, stage)
    weights, click, conversion = collect_gates(model, data_gen, args.max_samples, stage)

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
    entropy = entropy_rows(weights, click, conversion, task_names)
    lambda_table = ""
    lambda_records = []
    lambdas = model.mix_lambdas()
    if lambdas is not None:
        lambda_values = lambdas.detach().cpu().numpy()
        lambda_table = render_lambdas(task_names, lambda_values)
        lambda_records = [
            {"task": task_name, "lambda": float(value)}
            for task_name, value in zip(task_names, lambda_values)
        ]
    note = (
        "# MT-RankMixer token gate weights\n\n"
        "{}\n\n"
        "Checkpoint: `{}`\n\n"
        "Rows used: {} (cap {}). task_pooling: `{}`. "
        "Sample standard deviation (n - 1) is across rows inside this checkpoint, "
        "not across seeds. "
        "Gates sum to 1 over tokens inside each task.\n\n"
    ).format(stage_note(stage), checkpoint, int(weights.shape[0]), args.max_samples, model.task_pooling)
    output.parent.mkdir(parents=True, exist_ok=True)
    body = note + table + "\n" + entropy_table + "\n" + lambda_table
    output.write_text(body)
    json_path = Path(args.json).resolve() if args.json else output.with_suffix(".json")
    payload = {
        "stage": stage,
        "seed": str(args.seed),
        "expid": expid,
        "checkpoint": str(checkpoint),
        "n_rows": int(weights.shape[0]),
        "task_pooling": model.task_pooling,
        "token_names": token_names,
        "task_names": task_names,
        "rows": rows,
        "entropy": entropy,
        "lambdas": lambda_records,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    plot_gates(rows, token_names, figure)
    print(body)
    print("wrote {} and {} and {}".format(output, figure, json_path))


if __name__ == "__main__":
    main()
