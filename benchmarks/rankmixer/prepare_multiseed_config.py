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
    parser.add_argument("--model-root", default="",
                        help="Override Base.model_root in the cloned config.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Override Base.num_workers when positive.")
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
    base = dict(config.get("Base", {}))
    if args.model_root:
        base["model_root"] = args.model_root
    if args.num_workers > 0:
        base["num_workers"] = int(args.num_workers)
    written = {"Base": base, args.expid: block}
    with open(dst / "model_config.yaml", "w") as handle:
        yaml.safe_dump(written, handle, sort_keys=False, allow_unicode=True)
    shutil.copy(src / "dataset_config.yaml", dst / "dataset_config.yaml")
    print("wrote {} expid={} seed={}".format(dst, args.expid, args.seed))


if __name__ == "__main__":
    main()
