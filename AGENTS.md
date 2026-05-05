# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python research codebase for multi-objective optimization in multi-task learning. Shared abstractions and numerical solvers live in `src/core/`, including `base_method.py` and `src/core/solvers/`. Synthetic experiment code is under `src/synthetic/`, split into `problems/`, `methods/`, `trainers/`, utilities, and visualization. Multi-task learning code is under `src/mtl/`, with `models/`, `datasets/`, `methods/`, and `trainers/`. The unified experiment entry point is `run.py` — pass any config and it dispatches to the right trainer based on the `domain` key. YAML configs live in `configs/synthetic/` and `configs/mtl/`. Dataset files are expected under `data/`, especially `data/multi_mnist/`.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run any experiment:

```bash
# Synthetic
python run.py --config configs/synthetic/pmtl.yaml
python run.py --config configs/synthetic/cpmtl.yaml

# MTL
python run.py --config configs/mtl/pmtl.yaml
python run.py --config configs/mtl/pmtl.yaml --pref-idx 2
python run.py --config configs/mtl/cpmtl.yaml
```

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type hints where practical, and concise docstrings for public classes and methods. Keep module names lowercase with underscores, for example `weighted_sum.py`. Use `PascalCase` for classes such as `LeNetPMTL`, and `snake_case` for functions, variables, and config keys. Prefer existing registries and base classes: new methods should subclass `MOOMethod` and be registered in `METHOD_REGISTRY`; new trainers should subclass `MTLTrainer` or `SyntheticTrainer` and be registered in `TRAINER_REGISTRY`.

## Adding a New Paper

1. **Method**: subclass `MOOMethod` → register in `METHOD_REGISTRY`
2. **Trainer**: subclass `MTLTrainer` or `SyntheticTrainer` → register in `TRAINER_REGISTRY`
3. **Config**: create `configs/<domain>/<paper>.yaml` with `domain: synthetic` or `domain: mtl`
4. Run: `python run.py --config configs/<domain>/<paper>.yaml`

No changes to `run.py` or any existing file required.

## Testing Guidelines

There is currently no dedicated test suite. Validate changes by running the smallest relevant experiment command and, for solver or method changes, at least one synthetic config. If adding tests, place them under `tests/`, mirror the source layout, and name files `test_<module>.py`. Keep tests deterministic by setting seeds consistently with the experiment config.

## Commit & Pull Request Guidelines

Git history currently uses short messages such as `update`; prefer improving this with imperative, scoped commits, for example `add mgda synthetic trainer`. Pull requests should describe the changed method, config, or dataset behavior; include the exact commands run; mention required data files; and attach plots or metric summaries when experiment output changes.

## Security & Configuration Tips

Do not commit large datasets, checkpoints, generated plots, or local environment files. Keep dataset paths configurable through YAML, and avoid hard-coding machine-specific absolute paths.
