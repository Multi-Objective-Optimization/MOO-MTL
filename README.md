# MOO-MTL

Unified research repository for **Multi-Objective Optimization in Multi-Task Learning**.

Integrates multiple papers with a consistent structure so that adding a new MOO method never requires learning a new layout.

| Paper | Algorithm(s) | Domain |
|---|---|---|
| Lin et al., NeurIPS 2019 | ParetoMTL, LinearScalarization | Synthetic + MTL |
| Sener & Koltun, NeurIPS 2018 | MOO-MTL | Synthetic |
| Ma et al., ICML 2020 | WeightedSum, CPMTL (KKT-based) | MTL |

---

## Project Structure

```
MOO-MTL/
├── run.py                  # Unified entry point for all experiments
├── src/
│   ├── core/               # Shared: base class, solvers (MinNorm, HVP, KKT, linalg)
│   ├── synthetic/          # Domain 1: synthetic MOO problems and methods
│   │   ├── problems/       # ConcaveProblem, Zdt2Variant + PROBLEM_REGISTRY
│   │   ├── methods/        # ParetoMTL, MOOMTL, LinearScalarization (numpy/autograd)
│   │   └── trainers/       # StandardSyntheticTrainer + SYNTHETIC_TRAINER_REGISTRY
│   └── mtl/                # Domain 2: multi-task learning
│       ├── models/         # LeNetPMTL, LeNetCPMTL, MultiTaskWrapper + BACKBONE_REGISTRY
│       ├── datasets/       # MultiMNIST (pickle-based)
│       ├── methods/        # ParetoMTL, WeightedSum, CPMTL (PyTorch)
│       └── trainers/       # ParetoMTLTrainer, CPMTLTrainer + MTL_TRAINER_REGISTRY
├── configs/
│   ├── synthetic/          # pmtl.yaml, cpmtl.yaml
│   └── mtl/                # pmtl.yaml, cpmtl.yaml
└── data/
    └── multi_mnist/        # multi_mnist.pickle, multi_fashion.pickle, ...
```

---

## Data Setup

Đặt pickle files vào `data/multi_mnist/`:
```
data/multi_mnist/multi_mnist.pickle
data/multi_mnist/multi_fashion.pickle
data/multi_mnist/multi_fashion_and_mnist.pickle
```

---

## Running Experiments

Tất cả experiment đều chạy qua một entry point duy nhất. Config tự khai báo domain của nó.

```bash
# Synthetic experiments
python run.py --config configs/synthetic/pmtl.yaml
python run.py --config configs/synthetic/cpmtl.yaml

# MTL experiments — train all preference vectors
python run.py --config configs/mtl/pmtl.yaml
python run.py --config configs/mtl/cpmtl.yaml

# MTL — single preference index (for cluster parallelism)
python run.py --config configs/mtl/pmtl.yaml --pref-idx 2
for i in $(seq 0 4); do python run.py --config configs/mtl/pmtl.yaml --pref-idx $i; done
```

---

## Adding a New MOO Method

1. **Method class** — subclass `MOOMethod` in `src/<domain>/methods/your_method.py`, implement `get_descent_direction()` and `step()`, register in `METHOD_REGISTRY`
2. **Trainer class** — subclass `SyntheticTrainer` or `MTLTrainer` in `src/<domain>/trainers/your_trainer.py`, implement `run()` or `train()`, register in `TRAINER_REGISTRY`
3. **Config YAML** — create `configs/<domain>/your_paper.yaml` with `domain: synthetic` or `domain: mtl`
4. Run: `python run.py --config configs/<domain>/your_paper.yaml`

No changes to `run.py` or any existing file required.

---

## References

- Lin, X. et al. "Pareto Multi-Task Learning." NeurIPS 2019.
- Ma, P. et al. "Efficient Continuous Pareto Exploration in Multi-Task Learning." ICML 2020.
- Sener, O. & Koltun, V. "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018.
