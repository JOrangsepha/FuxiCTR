#!/usr/bin/env python3
"""Post-hoc evaluation of one checkpoint, including clicked-only CVR AUC.

Ali-CCP's ``conversion`` label is impression-level (CTCVR): a row converts only
if it was clicked. The conversion AUC that ``run_expid.py`` logs is therefore
an impression-level CTCVR AUC, which is dominated by the click signal. The
classic CVR metric is the conversion AUC restricted to clicked rows
(``click == 1``). This script loads a saved checkpoint, does not train, and
writes one JSON with:

* ``click_auc``, ``click_logloss``                      all impressions
* ``conv_auc``, ``conv_logloss``                        all impressions (CTCVR)
* ``avg_auc``                                           mean of the two AUCs (the training monitor)
* ``cvr_clicked_auc``, ``cvr_clicked_logloss``          conversion head, rows with click == 1
* ``cvr_clicked_auc_ratio``                             same rows, ranked by p_conv / p_click
* row counts, including conversions without a click (should be 0)

``--stage valid`` is the default. ``--stage test`` additionally needs
``--allow_test`` and ``--guard_dir``. A per-expid marker in ``guard_dir``
is created before the test split is opened, and the script refuses to run if
the marker already exists, so each checkpoint is read on test at most once.

Example::

    python benchmarks/rankmixer/eval_cvr_clicked.py --gpu 0 --stage valid \\
        --checkpoint "$CKPT" --config "$CFG" --expid "$EXPID" \\
        --workdir model_zoo/multitask/MT_RankMixer --output out.json
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import exp_loader  # noqa: E402


def _auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    if y_true.size == 0 or y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def _logloss(y_true, y_pred):
    from sklearn.metrics import log_loss
    if y_true.size == 0:
        return float("nan")
    eps = np.finfo(np.float64).eps
    return float(log_loss(y_true, np.clip(y_pred, eps, 1 - eps), labels=[0, 1]))


def compute_metrics(click, conv, p_click, p_conv):
    """All metrics from label and prediction arrays (1-D, aligned)."""
    click = np.asarray(click, dtype=np.float64)
    conv = np.asarray(conv, dtype=np.float64)
    p_click = np.asarray(p_click, dtype=np.float64)
    p_conv = np.asarray(p_conv, dtype=np.float64)
    clicked = click > 0.5
    out = {
        "n": int(click.size),
        "n_click": int(clicked.sum()),
        "n_conv": int((conv > 0.5).sum()),
        "n_conv_in_clicked": int((conv[clicked] > 0.5).sum()),
        "n_conv_without_click": int(((conv > 0.5) & ~clicked).sum()),
        "click_auc": _auc(click, p_click),
        "click_logloss": _logloss(click, p_click),
        "conv_auc": _auc(conv, p_conv),
        "conv_logloss": _logloss(conv, p_conv),
    }
    out["avg_auc"] = (out["click_auc"] + out["conv_auc"]) / 2.0
    out["cvr_clicked_auc"] = _auc(conv[clicked], p_conv[clicked])
    out["cvr_clicked_logloss"] = _logloss(conv[clicked], p_conv[clicked])
    ratio = p_conv[clicked] / np.clip(p_click[clicked], 1e-12, None)
    out["cvr_clicked_auc_ratio"] = _auc(conv[clicked], ratio)
    return out


def predict_split(model, data_gen):
    import torch
    labels = list(model.feature_map.labels)
    preds = {label: [] for label in labels}
    trues = {label: [] for label in labels}
    model.eval()
    with torch.no_grad():
        for batch in data_gen:
            out = model.forward(batch)
            y = model.get_labels(batch)
            for index, label in enumerate(labels):
                preds[label].append(out["{}_pred".format(label)].detach().float().cpu().numpy().reshape(-1))
                trues[label].append(y[index].detach().float().cpu().numpy().reshape(-1))
    return ({k: np.concatenate(v) for k, v in preds.items()},
            {k: np.concatenate(v) for k, v in trues.items()})


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Checkpoint eval with clicked-only CVR AUC.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stage", choices=("valid", "test"), default="valid")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--allow_test", action="store_true",
                        help="Required with --stage test. Only for the frozen final report.")
    parser.add_argument("--guard_dir", default="",
                        help="Required with --stage test. One marker per expid; refuses a second read.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit("checkpoint not found: {}".format(checkpoint))
    config = str(Path(args.config).resolve())
    output = Path(args.output).resolve()
    workdir = str(Path(args.workdir).resolve())

    marker = None
    if args.stage == "test":
        if not args.allow_test or not args.guard_dir:
            raise SystemExit("--stage test needs --allow_test and --guard_dir. Test is read once, "
                             "after the design is frozen.")
        guard = Path(args.guard_dir).resolve()
        guard.mkdir(parents=True, exist_ok=True)
        marker = guard / "{}.test_read".format(args.expid)
        try:
            fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit("REFUSED: test split already read for {} (marker {}).".format(args.expid, marker))
        with os.fdopen(fd, "w") as handle:
            handle.write("{} host={} pid={} checkpoint={}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"), socket.gethostname(), os.getpid(), checkpoint))
        print("NOTE: reading the test split once for {}. Marker {}".format(args.expid, marker), flush=True)

    exp_loader.bootstrap_workdir(repo_root, workdir)
    import src as model_zoo
    params, feature_map = exp_loader.prepare_params(config, args.expid, args.gpu)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    model.load_weights(str(checkpoint))
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "embedding_layer" in n)
    data_gen = exp_loader.open_stage(feature_map, params, args.stage)
    start = time.time()
    preds, trues = predict_split(model, data_gen)
    labels = list(feature_map.labels)
    if labels[:2] != ["click", "conversion"]:
        raise SystemExit("expected labels [click, conversion], got {}".format(labels))
    metrics = compute_metrics(trues["click"], trues["conversion"], preds["click"], preds["conversion"])
    record = {
        "expid": args.expid,
        "stage": args.stage,
        "checkpoint": str(checkpoint),
        "config": config,
        "model": params["model"],
        "loss_weight": str(params.get("loss_weight", "EQ")),
        "seed": params.get("seed"),
        "params_total": int(total),
        "params_embedding": int(emb),
        "params_non_embedding": int(total - emb),
        "seconds": round(time.time() - start, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    record.update(metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(output)
    print(json.dumps(record, indent=2))
    print("wrote {}".format(output))


if __name__ == "__main__":
    main()
