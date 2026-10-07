#!/usr/bin/env python3
"""Aggregate per-seed gate JSON into mean ± sample std across seeds.

The inputs are the JSON sidecars written by ``analyze_gates.py``. The standard
deviation is across seeds, using the per-seed mean gate weight (or entropy, or
λ) as the observation. It is not the within-seed row standard deviation.

Example::

    python benchmarks/rankmixer/aggregate_gates.py \\
        --inputs analysis/g6_gate_s2025.json analysis/g6_gate_s2026.json analysis/g6_gate_s2027.json \\
        --output analysis/g6_gate_seeds.md
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import analyze_gates  # noqa: E402


def load_payloads(paths):
    payloads = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if not payload.get("seed"):
            match_name = Path(path).stem
            payloads.append(payload)
            payload.setdefault("_source", match_name)
        else:
            payloads.append(payload)
    return payloads


def render_report(payloads):
    stages = sorted({payload.get("stage", "") for payload in payloads})
    seeds = [str(payload.get("seed", "")) for payload in payloads]
    aggregated = analyze_gates.aggregate_seed_gates(payloads)
    token_names = aggregated["token_names"]
    lines = [
        "# Gate weights across seeds",
        "",
        "Seeds: {}.".format(", ".join(seeds)),
        "Stage: {}.".format(", ".join(stages) if any(stages) else "unknown"),
        "",
        analyze_gates.stage_note(stages[0] if len(stages) == 1 and stages[0] else "valid"),
        "",
        "Each cell is the mean of the per-seed mean gate weights, ± the sample "
        "standard deviation across seeds (n - 1). It is not the standard deviation "
        "across rows inside one checkpoint.",
        "",
        analyze_gates.render_aggregate_markdown(aggregated, token_names).rstrip(),
        "",
        "## Mean gate entropy across seeds (nats)",
        "",
        "| slice | task | n_seeds | entropy mean±std |",
        "| --- | --- | --- | --- |",
    ]
    for record in analyze_gates.aggregate_entropy(payloads):
        lines.append("| {} | {} | {} | {} |".format(
            record["slice"], record["task"], record["n_seeds"],
            analyze_gates.format_mean_std(record["mean"], record["std"])))
    lambdas = analyze_gates.aggregate_lambdas(payloads)
    if lambdas:
        lines.extend([
            "",
            "## Residual mix weight λ across seeds",
            "",
            "| task | n_seeds | lambda mean±std |",
            "| --- | --- | --- |",
        ])
        for record in lambdas:
            lines.append("| {} | {} | {} |".format(
                record["task"], record["n_seeds"],
                analyze_gates.format_mean_std(record["mean"], record["std"])))
    if "test" in stages:
        lines.extend([
            "",
            "At least one input was computed on the test split. Do not use this "
            "table to choose a model unless the design was already frozen.",
        ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Mean ± std of gate weights across seeds.")
    parser.add_argument("--inputs", nargs="+", required=True, help="analyze_gates JSON sidecars.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = load_payloads(args.inputs)
    body = render_report(payloads)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)
    print(body)
    print("wrote {}".format(output))


if __name__ == "__main__":
    main()
