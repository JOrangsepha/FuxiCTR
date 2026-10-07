#!/usr/bin/env python3
"""Print mean ± sample standard deviation for multiseed_summary.csv.

Only rows with status ``ok`` are included. Standard deviation uses the sample
definition (divide by n - 1). A single successful seed prints std as blank.
"""

import argparse
import csv
import math
from collections import defaultdict


METRICS = (
    "valid_click_auc",
    "valid_conv_auc",
    "valid_mean_auc",
    "valid_click_logloss",
    "valid_conv_logloss",
    "valid_mean_logloss",
    "test_click_auc",
    "test_conv_auc",
    "test_mean_auc",
    "test_click_logloss",
    "test_conv_logloss",
    "test_mean_logloss",
)


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(var)


def format_pair(mean, std):
    if std is None:
        return "{:.6f}".format(mean)
    return "{:.6f} ± {:.6f}".format(mean, std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    grouped = defaultdict(lambda: defaultdict(list))
    with open(args.csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            for key in METRICS:
                text = row.get(key, "")
                if text == "":
                    continue
                grouped[row["model"]][key].append(float(text))

    if not grouped:
        print("No status=ok rows in {}".format(args.csv))
        return

    active = [
        key for key in METRICS
        if any(grouped[model][key] for model in grouped)
    ]
    header = ["model", "n"] + active
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for model in sorted(grouped):
        cells = [model]
        n = max((len(grouped[model][key]) for key in active), default=0)
        cells.append(str(n))
        for key in active:
            values = grouped[model][key]
            if not values:
                cells.append("")
            else:
                cells.append(format_pair(*mean_std(values)))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
