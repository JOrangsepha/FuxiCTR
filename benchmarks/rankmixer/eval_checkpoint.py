#!/usr/bin/env python3
"""Evaluate one trained checkpoint on one split. Does not train.

``--stage test`` is the frozen final report. The rigor suite does not call
this script. ``--stage valid`` repeats the validation metrics already used
for selection; it does not replace them.

The printed task lines match ``run_expid.py`` so ``summarize_multiseed.py``
can parse a redirected log.
"""

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import exp_loader  # noqa: E402


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on one split.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stage", choices=("valid", "test"), default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expid", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit("checkpoint not found: {}".format(checkpoint))
    if args.stage == "test":
        print("NOTE: reading the test split. Do this only after the design is frozen.", flush=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    exp_loader.bootstrap_workdir(repo_root, args.workdir)
    import src as model_zoo
    params, feature_map = exp_loader.prepare_params(args.config, args.expid, args.gpu)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    model.load_weights(str(checkpoint))
    data_gen = exp_loader.open_stage(feature_map, params, args.stage)
    if args.stage == "valid":
        header = "****** Validation evaluation ******"
    else:
        header = "******** Test evaluation ********"
    print(header, flush=True)
    model.evaluate(data_gen)


if __name__ == "__main__":
    main()
