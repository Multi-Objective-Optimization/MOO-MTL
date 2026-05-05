# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run any experiment — one entry point for everything
python run.py --config configs/synthetic/pmtl.yaml
python run.py --config configs/synthetic/cpmtl.yaml
python run.py --config configs/mtl/pmtl.yaml
python run.py --config configs/mtl/cpmtl.yaml

# Run a single preference vector (MTL only, for cluster parallelism)
python run.py --config configs/mtl/pmtl.yaml --pref-idx 2
for i in $(seq 0 4); do python run.py --config configs/mtl/pmtl.yaml --pref-idx $i; done
```

## Architecture

Two domains sharing the same registry + YAML config pattern:
- **Synthetic** (`src/synthetic/`): NumPy/autograd for test problems (concave, zdt2)
- **MTL** (`src/mtl/`): PyTorch for neural network multi-task learning (MultiMNIST)

### Adding a new paper (checklist)

1. **Method class** — subclass `MOOMethod` in the appropriate domain (`src/synthetic/methods/` or `src/mtl/methods/`), implement `get_descent_direction()` and `step()`, register in `METHOD_REGISTRY`
2. **Trainer class** — subclass `SyntheticTrainer` or `MTLTrainer` in the corresponding `trainers/` folder, implement `run()` or `train()`, register in `TRAINER_REGISTRY`
3. **Config YAML** — create `configs/<domain>/<paper>.yaml` with `domain: synthetic` or `domain: mtl`
4. Run with `python run.py --config configs/<domain>/<paper>.yaml`

### Key registries

| Registry | Location | Role |
|---|---|---|
| `PROBLEM_REGISTRY` | `src/synthetic/problems/__init__.py` | synthetic test problems |
| `SYNTHETIC_METHOD_REGISTRY` | `src/synthetic/methods/__init__.py` | synthetic MOO methods |
| `SYNTHETIC_TRAINER_REGISTRY` | `src/synthetic/trainers/__init__.py` | synthetic training loops |
| `MTL_METHOD_REGISTRY` | `src/mtl/methods/__init__.py` | MTL MOO methods |
| `MTL_TRAINER_REGISTRY` | `src/mtl/trainers/__init__.py` | MTL training loops |
| `BACKBONE_REGISTRY` | `src/mtl/models/__init__.py` | neural network backbones |

### Core solvers (`src/core/solvers/`) — shared across all methods

- `min_norm_solver_*.py`: Frank-Wolfe for minimum-norm task weights (numpy and torch variants)
- `hvp_solver.py` + `kkt_solver.py` + `linalg_solver.py`: Hessian-Vector Product and KKT systems (CPMTL)

### CPMTL is two-phase

Phase 1 trains with `WeightedSum` (initialization), Phase 2 applies the KKT-based solver to explore the Pareto front. `CPMTLTrainer` runs Phase 1 automatically if checkpoints are not found.

### Entry point

- `run.py`: unified entry point — reads `domain` from config and dispatches to the right trainer

## Configuration

All YAML configs must have a top-level `domain: synthetic` or `domain: mtl` key.

MTL configs specify: `dataset`, `model`, `method`, `training` (optimizer, scheduler, epochs), `experiment`.
Synthetic configs specify: `problem`, `method`, `optimization`, `experiment`.

The `--pref-idx` flag selects a single preference vector from `method.n_pref` evenly-spaced vectors, enabling parallelism across cluster jobs.

## Data

MultiMNIST pickle files must be present in `data/multi_mnist/`. Three variants exist: standard MNIST (`multi_mnist.pickle`), FashionMNIST (`multi_fashion.pickle`), and mixed (`multi_fashion_and_mnist.pickle`). These are loaded by `src/mtl/datasets/multi_mnist.py`.
