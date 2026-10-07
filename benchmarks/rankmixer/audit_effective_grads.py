#!/usr/bin/env python3
"""Per-task trunk gradient norms, raw and as weighted by EQ and NORM.

``audit_trunk_grads.py`` reports ||grad_trunk L_k|| for the raw per-task BCE.
That is what EQ (unnormalized sum) differentiates. Under ``loss_weight: NORM``
the objective is sum_k L_k / stopgrad(L_k), so task k reaches the trunk as
grad(L_k) / L_k. This script measures, on a few validation batches and without
updating weights:

* raw:  ||grad_trunk L_k||               (the EQ contribution)
* norm: ||grad_trunk (L_k / sg(L_k))||   (the NORM contribution)
* the trunk gradient norm of the full EQ and NORM objectives

and reports click/conversion ratios for both. Pass several checkpoints (for
example an EQ-trained and a NORM-trained g6_mean) to compare. Trunk is
``embedding_layer``, ``tokenizer``, ``encoder`` as in audit_trunk_grads.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import exp_loader  # noqa: E402
from audit_trunk_grads import grad_norm_of, is_trunk_parameter, _take_batches  # noqa: E402


def batch_report(model, batch):
    model.eval()
    model.zero_grad()
    output = model(batch)
    y_true = model.get_labels(batch)
    labels = list(model.feature_map.labels)
    losses = [model.loss_fn[i](output["{}_pred".format(l)], y_true[i], reduction="mean")
              for i, l in enumerate(labels)]
    row = {"n_pos": {l: int(y_true[i].sum().item()) for i, l in enumerate(labels)}}
    for i, label in enumerate(labels):
        model.zero_grad()
        losses[i].backward(retain_graph=True)
        raw = grad_norm_of(model, is_trunk_parameter)
        row[label] = {"loss": float(losses[i].detach()), "raw": raw,
                      "norm": raw / max(abs(float(losses[i].detach())), 1e-12)}
    for mode in ("EQ", "NORM"):
        model.zero_grad()
        stacked = torch.stack(losses)
        if mode == "EQ":
            total = stacked.sum()
        else:
            total = (stacked / stacked.detach().abs().clamp_min(1e-12)).sum()
        total.backward(retain_graph=(mode == "EQ"))
        row["total_" + mode] = grad_norm_of(model, is_trunk_parameter)
    model.zero_grad()
    return row


def audit(repo_root, workdir, config, expid, checkpoint, gpu, batches):
    exp_loader.bootstrap_workdir(repo_root, workdir)
    import src as model_zoo
    params, feature_map = exp_loader.prepare_params(config, expid, gpu)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    model.load_weights(checkpoint)
    gen = exp_loader.open_stage(feature_map, params, "valid")
    rows = [batch_report(model, b) for b in _take_batches(gen, batches)]
    return params, rows


def summarize(rows, key):
    click = np.array([r["click"][key] for r in rows])
    conv = np.array([r["conversion"][key] for r in rows])
    return click.mean(), conv.mean(), click.mean() / conv.mean(), np.median(click / conv)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--workdir", default=str(repo_root / "model_zoo" / "multitask" / "MT_RankMixer"))
    parser.add_argument("--item", action="append", required=True,
                        help="label|config_dir|expid|checkpoint ; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lines = ["# Trunk gradient audit: raw (EQ) vs NORM-weighted per-task contributions", "",
             "Validation split, {} batches of 8192 per checkpoint, eval mode, no weight updates. "
             "`raw` = ||grad_trunk L_k|| (what EQ sums). `NORM` = ||grad_trunk L_k|| / L_k "
             "(what `loss_weight: NORM` sums). Ratio = click / conversion of the batch-mean norms; "
             "median ratio is over batches. Trunk = embedding_layer, tokenizer, encoder.".format(args.batches), "",
             "| checkpoint | trained with | mean click BCE | mean conv BCE | raw click | raw conv | raw ratio (median) "
             "| NORM click | NORM conv | NORM ratio (median) | ratio under its own loss_weight | ||grad|| EQ total | ||grad|| NORM total |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for item in args.item:
        label, config, expid, ckpt = item.split("|")
        params, rows = audit(repo_root, args.workdir, config, expid, ckpt, args.gpu, args.batches)
        rc, rv, rr, rm = summarize(rows, "raw")
        nc, nv, nr, nm = summarize(rows, "norm")
        lc = np.mean([r["click"]["loss"] for r in rows])
        lv = np.mean([r["conversion"]["loss"] for r in rows])
        te = np.mean([r["total_EQ"] for r in rows])
        tn = np.mean([r["total_NORM"] for r in rows])
        lw = params.get("loss_weight", "EQ")
        if isinstance(lw, (list, tuple)):
            own = rr * float(lw[0]) / float(lw[1])
        elif str(lw).upper() == "NORM":
            own = nr
        else:
            own = rr
        lines.append("| {} | {} | {:.4f} | {:.5f} | {:.3e} | {:.3e} | {:.2f} ({:.2f}) | {:.3e} | {:.3e} | {:.3f} ({:.3f}) | {:.3f} | {:.3e} | {:.3e} |".format(
            label, str(lw).replace("|", "/"), lc, lv, rc, rv, rr, rm, nc, nv, nr, nm, own, te, tn))
        print(label, "batches", len(rows), "pos per batch", [r["n_pos"] for r in rows][:4], flush=True)
    lines += ["", "Ratio > 1 means click pushes the shared trunk harder than conversion. "
              "Small-sample diagnostic (one checkpoint per row), not a population estimate.", ""]
    Path(args.output).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
