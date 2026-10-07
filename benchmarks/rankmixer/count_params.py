#!/usr/bin/env python3
"""Instantiate one config on CPU and print parameter counts as a CSV row.

Does not read any data split. Embedding parameters are those under
``embedding_layer``. Also prints the loss_weight the model actually holds,
which is how the follow-up smoke test confirms NORM is wired through.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import exp_loader  # noqa: E402


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--header", action="store_true")
    args = parser.parse_args()
    config = str(Path(args.config).resolve())
    exp_loader.bootstrap_workdir(repo_root, str(Path(args.workdir).resolve()))
    import src as model_zoo
    params, feature_map = exp_loader.prepare_params(config, args.expid, -1)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "embedding_layer" in n)
    keys = ["embedding_dim", "learning_rate", "batch_size", "optimizer", "embedding_regularizer",
            "net_regularizer", "net_dropout", "epochs", "early_stop_patience", "loss_weight"]
    if args.header:
        print("PARAMS,label,expid,model,total,embedding,non_embedding,model_loss_weight," + ",".join(keys))
    print("PARAMS,{},{},{},{},{},{},{},{}".format(
        args.label or args.expid, args.expid, params["model"], total, emb, total - emb,
        str(getattr(model, "loss_weight", "?")).replace(",", ";"),
        ",".join(str(params.get(k, "")).replace(",", ";") for k in keys)))


if __name__ == "__main__":
    main()
