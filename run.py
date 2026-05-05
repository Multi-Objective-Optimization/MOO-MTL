"""Unified entry point for all MOO-MTL experiments.

Usage:
    python run.py --config configs/synthetic/pmtl.yaml
    python run.py --config configs/synthetic/moomtl.yaml
    python run.py --config configs/mtl/pmtl.yaml
    python run.py --config configs/mtl/pmtl.yaml --pref-idx 2
    python run.py --config configs/mtl/cpmtl.yaml

The config must have a top-level 'domain' key: 'synthetic' or 'mtl'.
To add a new paper: create a method class, a trainer class, register both, create a config.
"""

import argparse
import logging
import random

import numpy as np
import yaml


def _setup_logging_and_seed(cfg: dict) -> None:
    exp_cfg = cfg.get("experiment", {})
    logging.basicConfig(
        level=getattr(logging, exp_cfg.get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    seed = exp_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Run a MOO-MTL experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--pref-idx", type=int, default=None,
        help="Run a single preference vector index (MTL only, for cluster parallelism)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    domain = cfg.get("domain")
    if domain is None:
        raise ValueError(
            "Config must have a top-level 'domain' key. "
            "Options: 'synthetic' | 'mtl'"
        )

    _setup_logging_and_seed(cfg)

    if domain == "synthetic":
        from src.synthetic.trainers import build_trainer
        trainer = build_trainer(cfg["method"]["name"], cfg)
        trainer.run()

    elif domain == "mtl":
        import torch
        seed = cfg.get("experiment", {}).get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        from src.mtl.trainers import build_trainer
        trainer = build_trainer(cfg["method"]["name"], cfg)
        trainer.train(pref_idx=args.pref_idx)

    else:
        raise ValueError(f"Unknown domain '{domain}'. Options: synthetic | mtl")


if __name__ == "__main__":
    main()
