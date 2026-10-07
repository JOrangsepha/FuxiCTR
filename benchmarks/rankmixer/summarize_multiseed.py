#!/usr/bin/env python3
"""Build a summary CSV from the per-job logs of a multi-seed driver.

Only training logs from ``run_expid.py`` are parsed. Gate analysis logs
(``gate_*.log``, ``analyze_*.log``) and the trunk-gradient audit
(``grad_*.log``) are skipped. They are not failed training runs.

Selection metrics are the last ``****** Validation evaluation ******`` block,
which is the restored best checkpoint. Test metrics are recorded when that
block exists. The rigor suite passes ``--skip_test``, so a successful rigor
run has validation metrics and an empty test block. A run is ``ok`` when it
exits 0 and at least one of those two blocks has click and conversion AUC.
"""

import argparse
import csv
import re
from pathlib import Path


VALID_HEADER = "****** Validation evaluation ******"
TEST_HEADER = "******** Test evaluation ********"
TASK_LINE = re.compile(
    r"\[Task: (?P<task>\w+)\]\[Metrics\] logloss: (?P<logloss>[-+0-9.eE]+) - AUC: (?P<auc>[-+0-9.eE]+)"
)
EVAL_EPOCH = re.compile(r"Evaluation @epoch (?P<epoch>\d+)")
SAVE_BEST = "Save best model:"
_ANALYSIS_PREFIXES = ("gate_", "analyze_", "grad_")


def is_training_log(path, text=""):
    """True when ``path`` is a ``run_expid`` training log, not an analysis log.

    Filename prefixes ``gate_``, ``analyze_``, and ``grad_`` are analysis
    outputs and are never training runs. A log that contains ``Start training:``
    is a training run. A log that mentions the gate script and never started
    training is skipped. Anything else is kept, so a crashed training log that
    died before the start line is still reported as failed.
    """
    name = Path(path).stem
    for prefix in _ANALYSIS_PREFIXES:
        if name.startswith(prefix):
            return False
    if "Start training:" in text:
        return True
    if "token gate weights" in text or "analyze_gates" in text:
        return False
    return True


def _metrics_from_block(text):
    metrics = {}
    for match in TASK_LINE.finditer(text):
        task = match.group("task")
        metrics["{}_logloss".format(task)] = float(match.group("logloss"))
        metrics["{}_auc".format(task)] = float(match.group("auc"))
    return metrics


def parse_log(text):
    """Return best epoch, validation metrics, and test metrics.

    Validation metrics come from the last validation-evaluation block, cut
    off at the test header when both exist. Test metrics come from the last
    test-evaluation block. The best epoch is the last ``Evaluation @epoch N``
    that is followed by ``Save best model`` before the next evaluation.
    """
    best_epoch = ""
    current_epoch = None
    for line in text.splitlines():
        epoch_match = EVAL_EPOCH.search(line)
        if epoch_match:
            current_epoch = epoch_match.group("epoch")
        if SAVE_BEST in line and current_epoch is not None:
            best_epoch = current_epoch
    valid_metrics = {}
    if VALID_HEADER in text:
        after = text.rsplit(VALID_HEADER, 1)[-1]
        if TEST_HEADER in after:
            after = after.split(TEST_HEADER, 1)[0]
        valid_metrics = _metrics_from_block(after)
    test_metrics = {}
    if TEST_HEADER in text:
        test_metrics = _metrics_from_block(text.rsplit(TEST_HEADER, 1)[-1])
    return best_epoch, valid_metrics, test_metrics


def _has_task_aucs(metrics):
    return "click_auc" in metrics and "conversion_auc" in metrics


def job_status(exit_code, valid_metrics, test_metrics):
    if exit_code == 124:
        return "timeout"
    if exit_code != 0:
        return "failed"
    if not _has_task_aucs(valid_metrics) and not _has_task_aucs(test_metrics):
        return "failed"
    return "ok"


def model_name(expid):
    """Group id for one expid, with the trailing ``_s{seed}`` removed."""
    stem = re.sub(r"_s\d+$", "", expid)
    if stem.startswith("MTRankMixer_aliccp_semantic_mean"):
        return "MTRankMixer_semantic_mean"
    if stem.startswith("MTRankMixer_aliccp_semantic_residual"):
        return "MTRankMixer_semantic_residual"
    if stem.startswith("MTRankMixer_aliccp_semantic_entropy"):
        return "MTRankMixer_semantic_entropy"
    if stem.startswith("MTRankMixer_aliccp_semantic"):
        return "MTRankMixer_semantic"
    if stem.startswith("PLE_aliccp_es"):
        return "PLE"
    aliases = {
        "MTR_rg_g3_mean": "g3_mean",
        "MTR_rg_g6_mean": "g6_mean",
        "MTR_rg_g6_residual": "g6_residual",
        "MTR_rg_g6_gate": "g6_gate",
        "MTR_rg_g6_random": "g6_random",
        "MTR_rg_g6_sequential": "g6_sequential",
        "PLE_rg": "PLE",
    }
    return aliases.get(stem, stem)


def read_int(path, default):
    if not path.is_file():
        return default
    text = path.read_text().strip()
    if text == "":
        return default
    return int(text)


def _mean_pair(metrics):
    if "click_auc" not in metrics or "conversion_auc" not in metrics:
        return "", ""
    return (
        (metrics["click_auc"] + metrics["conversion_auc"]) / 2.0,
        (metrics["click_logloss"] + metrics["conversion_logloss"]) / 2.0,
    )


def select_training_logs(log_dir):
    """Yield ``(path, text)`` for training logs only, in filename order."""
    chosen = []
    for log_path in sorted(Path(log_dir).glob("*.log")):
        text = log_path.read_text(errors="replace")
        if is_training_log(log_path, text):
            chosen.append((log_path, text))
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="Directory of <expid>.log files.")
    parser.add_argument("--csv", required=True, help="Output CSV path.")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    rows = []
    for log_path, text in select_training_logs(log_dir):
        expid = log_path.stem
        seed_match = re.search(r"_s(\d+)$", expid)
        seed = seed_match.group(1) if seed_match else ""
        exit_code = read_int(log_dir / "{}.exit".format(expid), default=-1)
        seconds = read_int(log_dir / "{}.time".format(expid), default=-1)
        best_epoch, valid_metrics, test_metrics = parse_log(text)
        status = job_status(exit_code, valid_metrics, test_metrics)
        valid_mean_auc, valid_mean_logloss = _mean_pair(valid_metrics)
        test_mean_auc, test_mean_logloss = _mean_pair(test_metrics)
        rows.append({
            "model": model_name(expid),
            "seed": seed,
            "expid": expid,
            "status": status,
            "exit_code": exit_code,
            "seconds": seconds,
            "best_epoch": best_epoch,
            "valid_click_auc": valid_metrics.get("click_auc", ""),
            "valid_click_logloss": valid_metrics.get("click_logloss", ""),
            "valid_conv_auc": valid_metrics.get("conversion_auc", ""),
            "valid_conv_logloss": valid_metrics.get("conversion_logloss", ""),
            "valid_mean_auc": valid_mean_auc,
            "valid_mean_logloss": valid_mean_logloss,
            "test_click_auc": test_metrics.get("click_auc", ""),
            "test_click_logloss": test_metrics.get("click_logloss", ""),
            "test_conv_auc": test_metrics.get("conversion_auc", ""),
            "test_conv_logloss": test_metrics.get("conversion_logloss", ""),
            "test_mean_auc": test_mean_auc,
            "test_mean_logloss": test_mean_logloss,
            "log_path": str(log_path),
        })

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "seed", "expid", "status", "exit_code", "seconds", "best_epoch",
        "valid_click_auc", "valid_click_logloss", "valid_conv_auc", "valid_conv_logloss",
        "valid_mean_auc", "valid_mean_logloss",
        "test_click_auc", "test_click_logloss", "test_conv_auc", "test_conv_logloss",
        "test_mean_auc", "test_mean_logloss", "log_path",
    ]
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {} training rows to {} (analysis logs skipped)".format(len(rows), out))


if __name__ == "__main__":
    main()
