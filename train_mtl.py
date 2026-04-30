"""MTL training entry point.

Supports ParetoMTL (PMTL), WeightedSum (CPMTL Phase 1), and CPMTL (Phase 2).

Usage:
    # Train all preference vectors:
    python train_mtl.py --config configs/mtl/pmtl.yaml

    # Train one preference index (for cluster parallelism):
    python train_mtl.py --config configs/mtl/pmtl.yaml --pref-idx 2

    # CPMTL (2-phase):
    python train_mtl.py --config configs/mtl/cpmtl.yaml
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.mtl.datasets.multi_mnist import build_dataloaders
from src.mtl.models import build_model
from src.mtl.methods import build_method
from src.mtl.metrics import compute_accuracy, evaluate_cpmtl
from src.synthetic.utils import circle_points, evenly_dist_weights


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ParetoMTL training
# ---------------------------------------------------------------------------

def train_paretomtl(cfg: dict, pref_idx: int, log: logging.Logger) -> None:
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
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
    log.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = build_model(model_cfg).to(device)
    method = build_method(method_cfg)
    optimizer = _build_optimizer(model, train_cfg["optimizer"])
    scheduler = _build_scheduler(optimizer, train_cfg["scheduler"])

    log.info(f"Preference vector ({pref_idx + 1}/{n_pref}): {ref_vec[pref_idx].cpu().numpy()}")

    # Init phase
    for _ in range(method_cfg.get("init_epochs", 2)):
        model.train()
        for batch in train_loader:
            out = method.step(model, optimizer, batch, ref_vec, pref_idx, n_tasks, init_phase=True)
            if out.metadata.get("flag", False):
                log.info("Feasible solution found, stopping init phase early.")
                break

    # Main training
    niter = train_cfg["n_epochs"]
    log_every = train_cfg.get("log_every", 2)
    out_dir = Path(exp_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in range(niter):
        scheduler.step()
        model.train()
        for batch in train_loader:
            out = method.step(model, optimizer, batch, ref_vec, pref_idx, n_tasks, init_phase=False)

        if t == 0 or (t + 1) % log_every == 0:
            accs = compute_accuracy(model.model, train_loader, n_tasks, device)
            log.info(f"Epoch {t + 1}/{niter} | pref={pref_idx} | acc={[f'{a:.4f}' for a in accs]}")

    if train_cfg.get("save_model", True):
        ckpt_path = out_dir / f"model_pref{pref_idx}.pt"
        torch.save(model.model.state_dict(), ckpt_path)
        log.info(f"Saved model to {ckpt_path}")


# ---------------------------------------------------------------------------
# WeightedSum training (CPMTL Phase 1)
# ---------------------------------------------------------------------------

def _resize_to_28(batch):
    import torch.nn.functional as F
    imgs, labels = torch.utils.data.dataloader.default_collate(batch)
    imgs = F.interpolate(imgs, size=(28, 28), mode="bilinear", align_corners=False)
    return imgs, labels


def train_weighted_sum(cfg: dict, log: logging.Logger) -> None:
    import torch.nn.functional as F

    dataset_cfg = cfg["dataset"]
    method_cfg = cfg["method"]
    exp_cfg = cfg["experiment"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_pref = method_cfg["n_pref"]
    bs = method_cfg.get("init_batch_size", 256)

    base_train, base_test = build_dataloaders({**dataset_cfg, "batch_size": bs})
    trainloader = torch.utils.data.DataLoader(
        base_train.dataset, batch_size=bs, shuffle=True, num_workers=2, collate_fn=_resize_to_28
    )
    testloader = torch.utils.data.DataLoader(
        base_test.dataset, batch_size=bs, shuffle=False, num_workers=2, collate_fn=_resize_to_28
    )

    closures = [
        lambda n, l, t: F.cross_entropy(l[0], t[:, 0]),
        lambda n, l, t: F.cross_entropy(l[1], t[:, 1]),
    ]

    from src.mtl.models.lenet import LeNetCPMTL
    prefs = evenly_dist_weights(n_pref + 2, 2)
    out_dir = Path(exp_cfg["output_dir"]) / "weighted_sum"
    out_dir.mkdir(parents=True, exist_ok=True)

    method = build_method({**method_cfg, "name": "WeightedSum"})
    lr = method_cfg.get("init_lr", 1e-2)
    n_epochs = method_cfg.get("init_epochs", 30)

    random_ckpt_path = out_dir / "random.pth"

    for pref_idx, pref in enumerate(prefs):
        pref_tensor = torch.tensor(pref).float().to(device)
        network = LeNetCPMTL().to(device)
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs * len(trainloader))

        if not random_ckpt_path.is_file():
            torch.save({"state_dict": network.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()}, random_ckpt_path)
        ckpt = torch.load(random_ckpt_path, map_location="cpu")
        network.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        method.evaluate(network, testloader, device, closures, header=str(pref_idx))

        for epoch in range(1, n_epochs + 1):
            for batch in trainloader:
                method.step(network, optimizer, lr_scheduler, batch, pref_tensor, closures)
            losses, tops = method.evaluate(network, testloader, device, closures, header=f"{pref_idx}: {epoch}/{n_epochs}")

        ckpt = {"state_dict": network.state_dict(), "preference": pref, "record": {"losses": losses, "tops": tops}}
        torch.save(ckpt, out_dir / f"{pref_idx}.pth")
        log.info(f"Saved WeightedSum checkpoint: {out_dir / f'{pref_idx}.pth'}")


# ---------------------------------------------------------------------------
# CPMTL training (Phase 2)
# ---------------------------------------------------------------------------

def train_cpmtl(cfg: dict, log: logging.Logger) -> None:
    import torch.nn.functional as F

    dataset_cfg = cfg["dataset"]
    method_cfg = cfg["method"]
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = method_cfg.get("init_batch_size", 256)

    base_train, base_test = build_dataloaders({**dataset_cfg, "batch_size": bs})
    trainloader = torch.utils.data.DataLoader(
        base_train.dataset, batch_size=bs, shuffle=True, num_workers=2, collate_fn=_resize_to_28
    )
    testloader = torch.utils.data.DataLoader(
        base_test.dataset, batch_size=bs, shuffle=False, num_workers=2, collate_fn=_resize_to_28
    )

    closures = [
        lambda n, l, t: F.cross_entropy(l[0], t[:, 0]),
        lambda n, l, t: F.cross_entropy(l[1], t[:, 1]),
    ]

    from src.mtl.models.lenet import LeNetCPMTL
    ws_dir = Path(exp_cfg["output_dir"]) / "weighted_sum"
    out_dir = Path(exp_cfg["output_dir"]) / "cpmtl"
    out_dir.mkdir(parents=True, exist_ok=True)

    method = build_method(method_cfg)
    beta = torch.tensor([1.0, 0.0]).to(device)
    n_steps = method_cfg.get("n_steps", 10)
    lr = train_cfg["optimizer"]["lr"]

    for start_path in sorted(ws_dir.glob("[0-9]*.pth"), key=lambda x: int(x.stem)):
        ckpt_name = start_path.stem
        ckpt_out = out_dir / ckpt_name
        ckpt_out.mkdir(parents=True, exist_ok=True)

        network = LeNetCPMTL().to(device)
        start_ckpt = torch.load(start_path, map_location="cpu")
        network.load_state_dict(start_ckpt["state_dict"])

        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.0)
        hvp_solver, kkt_solver = method.build_kkt_solver(network, device, trainloader, closures)

        method.evaluate(network, testloader, device, closures, header=ckpt_name)

        for step in range(1, n_steps + 1):
            out = method.step(network, optimizer, kkt_solver, beta)
            losses, tops = method.evaluate(network, testloader, device, closures, header=f"{ckpt_name}: {step}/{n_steps}")
            torch.save({"state_dict": network.state_dict(), "beta": beta, "record": {"losses": losses, "tops": tops}}, ckpt_out / f"{step}.pth")

        hvp_solver.close()
        log.info(f"CPMTL done for {ckpt_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--pref-idx", type=int, default=None, help="Train single preference index (ParetoMTL only)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    logging.basicConfig(
        level=getattr(logging, exp_cfg.get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    seed = exp_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    method_name = cfg["method"]["name"]

    if method_name == "ParetoMTL":
        if args.pref_idx is not None:
            train_paretomtl(cfg, args.pref_idx, log)
        else:
            for i in range(cfg["method"]["n_pref"]):
                train_paretomtl(cfg, i, log)

    elif method_name == "WeightedSum":
        train_weighted_sum(cfg, log)

    elif method_name == "CPMTL":
        log.info("Phase 1: WeightedSum initialization")
        ws_dir = Path(exp_cfg["output_dir"]) / "weighted_sum"
        if not ws_dir.exists() or not list(ws_dir.glob("[0-9]*.pth")):
            train_weighted_sum(cfg, log)
        log.info("Phase 2: CPMTL exploration")
        train_cpmtl(cfg, log)

    else:
        raise ValueError(f"Unknown method '{method_name}'. Options: ParetoMTL | WeightedSum | CPMTL")


if __name__ == "__main__":
    main()
