#!/usr/bin/env python3
"""Clone one multiseed template into a single-seed config directory.

``load_config`` sets ``model_id`` from the experiment id, so each seed needs
its own expid. That keeps checkpoints and logs from overwriting each other.
"""

import argparse
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Template config directory.")
    parser.add_argument("--dst", required=True, help="Directory to write.")
    parser.add_argument("--template", required=True, help="Expid key to copy.")
    parser.add_argument("--expid", required=True, help="Expid key to write.")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    with open(src / "model_config.yaml", "r") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    if args.template not in config:
        raise SystemExit("template expid {!r} is not in {}".format(args.template, src))
    block = dict(config[args.template])
    block["seed"] = int(args.seed)
    written = {"Base": config.get("Base", {}), args.expid: block}
    with open(dst / "model_config.yaml", "w") as handle:
        yaml.safe_dump(written, handle, sort_keys=False, allow_unicode=True)
    shutil.copy(src / "dataset_config.yaml", dst / "dataset_config.yaml")
    print("wrote {} expid={} seed={}".format(dst, args.expid, args.seed))


if __name__ == "__main__":
    main()
