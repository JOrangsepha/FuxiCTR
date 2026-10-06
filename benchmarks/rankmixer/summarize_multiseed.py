#!/usr/bin/env python3
"""Build multiseed_summary.csv from the per-job logs of run_multiseed.sh."""

import argparse
import csv
import re
from pathlib import Path


TEST_HEADER = "******** Test evaluation ********"
TASK_LINE = re.compile(
    r"\[Task: (?P<task>\w+)\]\[Metrics\] logloss: (?P<logloss>[-+0-9.eE]+) - AUC: (?P<auc>[-+0-9.eE]+)"
)
EVAL_EPOCH = re.compile(r"Evaluation @epoch (?P<epoch>\d+)")
SAVE_BEST = "Save best model:"


def parse_log(text):
    """Return best epoch and test metrics from one FuxiCTR log.

    Test metrics are taken from the last test-evaluation block. The best
    epoch is the last ``Evaluation @epoch N`` that is followed by
    ``Save best model`` before the next evaluation.
    """
    best_epoch = ""
    current_epoch = None
    for line in text.splitlines():
        epoch_match = EVAL_EPOCH.search(line)
        if epoch_match:
            current_epoch = epoch_match.group("epoch")
        if SAVE_BEST in line and current_epoch is not None:
            best_epoch = current_epoch
    test_text = text.rsplit(TEST_HEADER, 1)[-1] if TEST_HEADER in text else ""
    metrics = {}
    for match in TASK_LINE.finditer(test_text):
        task = match.group("task")
        metrics["{}_logloss".format(task)] = float(match.group("logloss"))
        metrics["{}_auc".format(task)] = float(match.group("auc"))
    return best_epoch, metrics


def job_status(exit_code, metrics):
    if exit_code == 124:
        return "timeout"
    if exit_code != 0:
        return "failed"
    if "click_auc" not in metrics or "conversion_auc" not in metrics:
        return "failed"
    return "ok"


def read_int(path, default):
    if not path.is_file():
        return default
    text = path.read_text().strip()
    if text == "":
        return default
    return int(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="Directory of <expid>.log files.")
    parser.add_argument("--csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        expid = log_path.stem
        seed_match = re.search(r"_s(\d+)$", expid)
        seed = seed_match.group(1) if seed_match else ""
        if expid.startswith("MTRankMixer_aliccp_semantic_mean_es"):
            model = "MTRankMixer_semantic_mean"
        elif expid.startswith("MTRankMixer_aliccp_semantic_es"):
            model = "MTRankMixer_semantic"
        elif expid.startswith("PLE_aliccp_es"):
            model = "PLE"
        else:
            model = expid
        exit_code = read_int(log_dir / "{}.exit".format(expid), default=-1)
        seconds = read_int(log_dir / "{}.time".format(expid), default=-1)
        best_epoch, metrics = parse_log(log_path.read_text(errors="replace"))
        status = job_status(exit_code, metrics)
        click_auc = metrics.get("click_auc", "")
        click_logloss = metrics.get("click_logloss", "")
        conv_auc = metrics.get("conversion_auc", "")
        conv_logloss = metrics.get("conversion_logloss", "")
        if click_auc != "" and conv_auc != "":
            mean_auc = (click_auc + conv_auc) / 2.0
            mean_logloss = (click_logloss + conv_logloss) / 2.0
        else:
            mean_auc = ""
            mean_logloss = ""
        rows.append({
            "model": model,
            "seed": seed,
            "expid": expid,
            "status": status,
            "exit_code": exit_code,
            "seconds": seconds,
            "best_epoch": best_epoch,
            "test_click_auc": click_auc,
            "test_click_logloss": click_logloss,
            "test_conv_auc": conv_auc,
            "test_conv_logloss": conv_logloss,
            "test_mean_auc": mean_auc,
            "test_mean_logloss": mean_logloss,
            "log_path": str(log_path),
        })

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "seed", "expid", "status", "exit_code", "seconds", "best_epoch",
        "test_click_auc", "test_click_logloss", "test_conv_auc", "test_conv_logloss",
        "test_mean_auc", "test_mean_logloss", "log_path",
    ]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} rows to {}".format(len(rows), out))


if __name__ == "__main__":
    main()
