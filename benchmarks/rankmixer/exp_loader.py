#!/usr/bin/env python3
"""Load a FuxiCTR experiment the same way ``run_expid.py`` does, without training."""

import os
import sys


def prepare_params(config_dir, expid, gpu):
    """Load config and the on-disk feature map. Does not rebuild ``feature_map.json``."""
    from fuxictr.utils import load_config
    from fuxictr.features import FeatureMap

    params = load_config(config_dir, expid)
    params["gpu"] = gpu
    if params["data_format"] == "csv":
        from fuxictr.preprocess import FeatureProcessor, build_dataset
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    return params, feature_map


def bootstrap_workdir(repo_root, workdir):
    """chdir to the model directory and put the repo and that directory on ``sys.path``."""
    os.chdir(workdir)
    root = os.path.abspath(repo_root)
    work = os.path.abspath(workdir)
    if root not in sys.path:
        sys.path.insert(0, root)
    if work not in sys.path:
        sys.path.insert(0, work)


def open_stage(feature_map, params, stage):
    """Open one split. ``stage`` is ``valid`` or ``test`` and is not remapped."""
    from fuxictr.pytorch.dataloaders import RankDataLoader

    if stage not in ("valid", "test"):
        raise ValueError("stage must be 'valid' or 'test'.")
    return RankDataLoader(feature_map, stage=stage, **params)
