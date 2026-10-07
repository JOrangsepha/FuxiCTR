#!/usr/bin/env python3
"""Pre-launch smoke for ``loss_weight: NORM``. Not part of the recorded suite.

Checks, on a few validation batches and without claiming an Ali-CCP result:

* the loaded model actually holds ``loss_weight: NORM``
* ``add_loss`` equals 2.0 when both task losses are nonzero
* the trunk gradient of the NORM objective matches ``grad(L_c)/L_c + grad(L_v)/L_v``
* ``train_step`` returns about 2.0

Paths come from the command line. The AutoDL smoke used a copied config under
``/root/autodl-tmp/fu_smoke/configs/MTR_fu_g6_mean_norm_s2025``.

This script does not guard the empty-conversion blow-up. That failure mode is
documented on ``combine_task_losses`` and in
``results/followup/NORM_collapse_evidence.md``.
"""

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import exp_loader  # noqa: E402


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Smoke-check that NORM is wired through.")
    parser.add_argument("--config", required=True, help="Prepared config directory for one NORM expid.")
    parser.add_argument("--expid", required=True)
    parser.add_argument("--workdir", default=str(repo_root / "model_zoo" / "multitask" / "MT_RankMixer"))
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()

    exp_loader.bootstrap_workdir(repo_root, str(Path(args.workdir).resolve()))
    import src as model_zoo
    params, feature_map = exp_loader.prepare_params(str(Path(args.config).resolve()), args.expid, args.gpu)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    print("params loss_weight:", params["loss_weight"], "| model.loss_weight:", model.loss_weight)
    if str(params.get("loss_weight", "")).upper() != "NORM":
        raise SystemExit("SMOKE_NORM_FAIL: config loss_weight is not NORM")
    gen = exp_loader.open_stage(feature_map, params, "valid")
    batches = []
    for batch in gen:
        batches.append(batch)
        if len(batches) >= args.batches:
            break
    if not batches:
        raise SystemExit("SMOKE_NORM_FAIL: validation split produced no batches")
    model.train()
    model.compile(params["optimizer"], params["loss"], params["learning_rate"])
    # fit() is what normally sets this. train_step crashes without it.
    model._max_gradient_norm = float(params.get("max_gradient_norm", 10.0))
    batch = batches[0]
    output = model(batch)
    y_true = model.get_labels(batch)
    labels = list(feature_map.labels)
    losses = [model.loss_fn[i](output["{}_pred".format(label)], y_true[i], reduction="mean")
              for i, label in enumerate(labels)]
    print("per-task BCE:", [float(value) for value in losses])
    combined = model.add_loss(output, y_true)
    print("model.add_loss (NORM) =", float(combined), "EQ would be", float(sum(losses)))
    trunk = [param for name, param in model.named_parameters() if name.startswith(("tokenizer.", "encoder."))]
    grad_norm = torch.autograd.grad(combined, trunk, retain_graph=True)
    grad_click = torch.autograd.grad(losses[0], trunk, retain_graph=True)
    grad_conv = torch.autograd.grad(losses[1], trunk, retain_graph=True)
    err = max(float((got - (click / losses[0].detach() + conv / losses[1].detach())).abs().max())
              for got, click, conv in zip(grad_norm, grad_click, grad_conv))
    click_norm = torch.sqrt(sum((value ** 2).sum() for value in grad_click))
    conv_norm = torch.sqrt(sum((value ** 2).sum() for value in grad_conv))
    print("max |grad_NORM - (gc/Lc + gv/Lv)| =", err)
    print("trunk(excl. emb) raw grad ratio click/conv = %.3f ; NORM-weighted ratio = %.4f" % (
        float(click_norm / conv_norm), float((click_norm / losses[0]) / (conv_norm / losses[1]))))
    step_losses = [float(model.train_step(item)) for item in batches]
    print("train_step losses (NORM, expect 2.0 when both task losses are nonzero):", step_losses)
    # A batch with an underflowed conversion BCE makes NORM return 1 instead of 2.
    # That is the collapse mode, not a wiring failure, so it is reported and not
    # required for SMOKE_NORM_OK.
    nonzero = all(float(value.detach()) > 1e-12 for value in losses)
    ok = nonzero and all(abs(value - 2.0) < 1e-4 for value in step_losses) and err < 1e-3
    print("SMOKE_NORM_OK" if ok else "SMOKE_NORM_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
