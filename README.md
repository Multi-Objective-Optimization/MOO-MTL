# MOO-MTL

Unified research repository for **Multi-Objective Optimization in Multi-Task Learning**.

Integrates two papers with a consistent structure so that adding new MOO methods never requires learning a new layout.

| Paper | Algorithm(s) | Domain |
|---|---|---|
| Lin et al., NeurIPS 2019 | ParetoMTL, MOO-MTL, LinearScalarization | Synthetic + MTL |
| Ma et al., ICML 2020 | WeightedSum, CPMTL (KKT-based) | Synthetic + MTL |

---

## Project Structure

```
MOO-MTL/
├── train.py                # MTL entry point
├── run.py                  # Synthetic entry point
├── src/
│   ├── core/               # Shared: base class, solvers (MinNorm, HVP, KKT, linalg)
│   ├── synthetic/          # Domain 1: synthetic MOO problems and methods
│   │   ├── problems/       # ConcaveProblem, Zdt2Variant
│   │   └── methods/        # ParetoMTL, MOOMTL, LinearScalarization (numpy/autograd)
│   └── mtl/                # Domain 2: multi-task learning
│       ├── models/         # LeNetPMTL, LeNetCPMTL, MultiTaskWrapper
│       ├── datasets/       # MultiMNIST (pickle-based)
│       └── methods/        # ParetoMTL, WeightedSum, CPMTL (PyTorch)
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

### Synthetic

```bash
python run_synthetic.py --config configs/synthetic/pmtl.yaml
python run_synthetic.py --config configs/synthetic/cpmtl.yaml
```

### Multi-Task Learning

```bash
# ParetoMTL — train all preference vectors
python train_mtl.py --config configs/mtl/pmtl.yaml

# Single preference index (for cluster parallelism)
python train_mtl.py --config configs/mtl/pmtl.yaml --pref-idx 2

# CPMTL (2-phase: WeightedSum init → KKT exploration)
python train_mtl.py --config configs/mtl/cpmtl.yaml
```

---

## Adding a New MOO Method

### For synthetic problems:

1. Create `src/moo_mtl/synthetic/methods/your_method.py`
2. Subclass `MOOMethod` from `moo_mtl.core.base_method`
3. Implement `get_descent_direction()` and `step()`
4. Add one line to `src/moo_mtl/synthetic/methods/__init__.py`:
   ```python
   SYNTHETIC_METHOD_REGISTRY["YourMethod"] = YourMethodSynthetic
   ```
5. Set `method.name: YourMethod` in your config YAML

### For MTL:

Same process under `src/moo_mtl/mtl/methods/` and `MTL_METHOD_REGISTRY`.

The two domains are completely independent — adding a method to one does not affect the other.

---

## References

- Lin, X. et al. "Pareto Multi-Task Learning." NeurIPS 2019.
- Ma, P. et al. "Efficient Continuous Pareto Exploration in Multi-Task Learning." ICML 2020.
- Sener, O. & Koltun, V. "Multi-Task Learning as Multi-Objective Optimization." NeurIPS 2018.
