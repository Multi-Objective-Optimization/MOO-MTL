# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run synthetic experiments
python run_synthetic.py --config configs/synthetic/pmtl.yaml
python run_synthetic.py --config configs/synthetic/moomtl.yaml

# Run MTL experiments (single preference vector by index)
python train_mtl.py --config configs/mtl/pmtl.yaml
python train_mtl.py --config configs/mtl/cpmtl.yaml

# Run all preference vectors (loop externally)
for i in $(seq 0 9); do python train_mtl.py --config configs/mtl/pmtl.yaml --pref-idx $i; done
```

## Architecture

Two independent domains with the same algorithmic structure:
- **Synthetic** (`src/synthetic/`): NumPy/autograd for test problems (concave, ZDT2)
- **MTL** (`src/mtl/`): PyTorch for neural network multi-task learning (MultiMNIST, ResNet18)

Both domains use a **registry + YAML config** pattern. Methods are registered in `__init__.py` files and instantiated via `build_method()`. Adding a new method requires: subclass `MOOMethod` in `src/core/base_method.py`, implement `get_descent_direction()` and `step()`, register in the appropriate `METHOD_REGISTRY`.

**Core solvers** (`src/core/solvers/`) are shared:
- `min_norm_solver_*.py`: Frank-Wolfe for finding task weights minimizing gradient norm (numpy and torch variants)
- `hvp_solver.py` + `kkt_solver.py` + `linalg_solver.py`: Hessian-Vector Product and KKT systems for CPMTL

**CPMTL is two-phase**: Phase 1 trains with `WeightedSum` (initialization), Phase 2 applies the KKT-based solver to explore the Pareto front.

**Entry points**:
- `run_synthetic.py`: runs one synthetic problem, plots Pareto front
- `train_mtl.py`: three training modes (`train_paretomtl`, `train_weighted_sum`, `train_cpmtl`) selected by method name in config

## Configuration

YAML configs in `configs/` control all hyperparameters. MTL configs specify: `dataset`, `model`, `method`, `training` (optimizer, scheduler, epochs), and `experiment` (output dir, preference vectors). The `--pref-idx` CLI flag selects a single preference vector from `experiment.num_pref` evenly-spaced vectors, enabling parallelism across runs.

## Data

MultiMNIST pickle files must be present in `data/multi_mnist/`. Three variants exist: standard MNIST, FashionMNIST, and mixed. These are loaded by `src/mtl/datasets/multi_mnist.py`.
