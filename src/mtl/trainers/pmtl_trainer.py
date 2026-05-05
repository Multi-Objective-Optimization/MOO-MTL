import torch
from pathlib import Path

from src.mtl.datasets.multi_mnist import build_dataloaders
from src.mtl.models import build_model
from src.mtl.methods import build_method
from src.mtl.metrics import compute_accuracy
from src.synthetic.utils import circle_points
from .base_trainer import MTLTrainer


def _build_optimizer(model, opt_cfg: dict):
    name = opt_cfg["name"]
    lr = opt_cfg["lr"]
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=opt_cfg.get("momentum", 0.9))
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer '{name}'. Options: SGD | Adam")


def _build_scheduler(optimizer, sched_cfg: dict, n_steps_per_epoch: int = None):
    name = sched_cfg["name"]
    if name == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sched_cfg["milestones"], gamma=sched_cfg["gamma"]
        )
    elif name == "CosineAnnealingLR":
        t_max = sched_cfg.get("T_max", 30)
        total_steps = t_max * (n_steps_per_epoch or 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    raise ValueError(f"Unknown scheduler '{name}'. Options: MultiStepLR | CosineAnnealingLR")


class ParetoMTLTrainer(MTLTrainer):
    """Trainer for ParetoMTL (and any standard gradient-weighting MTL method)."""

    def train(self, pref_idx: int = None) -> None:
        cfg = self.cfg
        dataset_cfg = cfg["dataset"]
        method_cfg = cfg["method"]
        train_cfg = cfg["training"]
        exp_cfg = cfg["experiment"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_tasks = dataset_cfg["n_tasks"]
        n_pref = method_cfg["n_pref"]

        ref_vec = torch.tensor(circle_points([1], [n_pref])[0]).float().to(device)

        train_loader, test_loader = build_dataloaders(
            {**dataset_cfg, "batch_size": train_cfg.get("batch_size", 256)}
        )
        self.log.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        indices = [pref_idx] if pref_idx is not None else range(n_pref)
        for i in indices:
            self._train_single(cfg, i, ref_vec, train_loader, test_loader, device, n_tasks, n_pref)

    def _train_single(self, cfg, pref_idx, ref_vec, train_loader, test_loader, device, n_tasks, n_pref):
        model_cfg = cfg["model"]
        method_cfg = cfg["method"]
        train_cfg = cfg["training"]
        exp_cfg = cfg["experiment"]

        model = build_model(model_cfg).to(device)
        method = build_method(method_cfg)
        optimizer = _build_optimizer(model, train_cfg["optimizer"])
        scheduler = _build_scheduler(optimizer, train_cfg["scheduler"])

        self.log.info(f"Preference vector ({pref_idx + 1}/{n_pref}): {ref_vec[pref_idx].cpu().numpy()}")

        for _ in range(method_cfg.get("init_epochs", 2)):
            model.train()
            for batch in train_loader:
                out = method.step(model, optimizer, batch, ref_vec, pref_idx, n_tasks, init_phase=True)
                if out.metadata.get("flag", False):
                    self.log.info("Feasible solution found, stopping init phase early.")
                    break

        n_epochs = train_cfg["n_epochs"]
        log_every = train_cfg.get("log_every", 2)
        out_dir = Path(exp_cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        for t in range(n_epochs):
            scheduler.step()
            model.train()
            for batch in train_loader:
                method.step(model, optimizer, batch, ref_vec, pref_idx, n_tasks, init_phase=False)

            if t == 0 or (t + 1) % log_every == 0:
                accs = compute_accuracy(model.model, train_loader, n_tasks, device)
                self.log.info(
                    f"Epoch {t + 1}/{n_epochs} | pref={pref_idx} | acc={[f'{a:.4f}' for a in accs]}"
                )

        if train_cfg.get("save_model", True):
            ckpt_path = out_dir / f"model_pref{pref_idx}.pt"
            torch.save(model.model.state_dict(), ckpt_path)
            self.log.info(f"Saved model to {ckpt_path}")
