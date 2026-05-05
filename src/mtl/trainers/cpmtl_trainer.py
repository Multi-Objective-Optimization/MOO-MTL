import torch
import torch.nn.functional as F
from pathlib import Path

from src.mtl.datasets.multi_mnist import build_dataloaders
from src.mtl.methods import build_method
from src.synthetic.utils import evenly_dist_weights
from .base_trainer import MTLTrainer


def _resize_to_28(batch):
    imgs, labels = torch.utils.data.dataloader.default_collate(batch)
    imgs = F.interpolate(imgs, size=(28, 28), mode="bilinear", align_corners=False)
    return imgs, labels


def _build_cpmtl_loaders(dataset_cfg: dict, method_cfg: dict):
    bs = method_cfg.get("init_batch_size", 256)
    base_train, base_test = build_dataloaders({**dataset_cfg, "batch_size": bs})
    trainloader = torch.utils.data.DataLoader(
        base_train.dataset, batch_size=bs, shuffle=True, num_workers=2, collate_fn=_resize_to_28
    )
    testloader = torch.utils.data.DataLoader(
        base_test.dataset, batch_size=bs, shuffle=False, num_workers=2, collate_fn=_resize_to_28
    )
    return trainloader, testloader


def _default_closures():
    return [
        lambda n, l, t: F.cross_entropy(l[0], t[:, 0]),
        lambda n, l, t: F.cross_entropy(l[1], t[:, 1]),
    ]


class WeightedSumTrainer(MTLTrainer):
    """Phase 1 of CPMTL: pre-train one model per preference vector via weighted sum."""

    def train(self, pref_idx: int = None) -> None:
        cfg = self.cfg
        dataset_cfg = cfg["dataset"]
        method_cfg = cfg["method"]
        exp_cfg = cfg["experiment"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainloader, testloader = _build_cpmtl_loaders(dataset_cfg, method_cfg)
        closures = _default_closures()

        from src.mtl.models.lenet import LeNetCPMTL
        n_pref = method_cfg["n_pref"]
        prefs = evenly_dist_weights(n_pref + 2, 2)
        out_dir = Path(exp_cfg["output_dir"]) / "weighted_sum"
        out_dir.mkdir(parents=True, exist_ok=True)

        method = build_method({**method_cfg, "name": "WeightedSum"})
        lr = method_cfg.get("init_lr", 1e-2)
        n_epochs = method_cfg.get("init_epochs", 30)
        random_ckpt_path = out_dir / "random.pth"

        for idx, pref in enumerate(prefs):
            pref_tensor = torch.tensor(pref).float().to(device)
            network = LeNetCPMTL().to(device)
            optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, n_epochs * len(trainloader)
            )

            if not random_ckpt_path.is_file():
                torch.save({
                    "state_dict": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }, random_ckpt_path)
            ckpt = torch.load(random_ckpt_path, map_location="cpu")
            network.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

            method.evaluate(network, testloader, device, closures, header=str(idx))

            for epoch in range(1, n_epochs + 1):
                for batch in trainloader:
                    method.step(network, optimizer, lr_scheduler, batch, pref_tensor, closures)
                losses, tops = method.evaluate(
                    network, testloader, device, closures, header=f"{idx}: {epoch}/{n_epochs}"
                )

            torch.save(
                {"state_dict": network.state_dict(), "preference": pref,
                 "record": {"losses": losses, "tops": tops}},
                out_dir / f"{idx}.pth",
            )
            self.log.info(f"Saved WeightedSum checkpoint: {out_dir / f'{idx}.pth'}")


class CPMTLTrainer(MTLTrainer):
    """CPMTL: 2-phase trainer. Phase 1 (WeightedSum) runs automatically if not found."""

    def train(self, pref_idx: int = None) -> None:
        cfg = self.cfg
        exp_cfg = cfg["experiment"]

        ws_dir = Path(exp_cfg["output_dir"]) / "weighted_sum"
        if not ws_dir.exists() or not list(ws_dir.glob("[0-9]*.pth")):
            self.log.info("Phase 1: WeightedSum initialization")
            WeightedSumTrainer(cfg).train()

        self.log.info("Phase 2: CPMTL exploration")
        self._run_phase2()

    def _run_phase2(self) -> None:
        cfg = self.cfg
        dataset_cfg = cfg["dataset"]
        method_cfg = cfg["method"]
        exp_cfg = cfg["experiment"]
        train_cfg = cfg["training"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainloader, testloader = _build_cpmtl_loaders(dataset_cfg, method_cfg)
        closures = _default_closures()

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
                method.step(network, optimizer, kkt_solver, beta)
                losses, tops = method.evaluate(
                    network, testloader, device, closures,
                    header=f"{ckpt_name}: {step}/{n_steps}"
                )
                torch.save(
                    {"state_dict": network.state_dict(), "beta": beta,
                     "record": {"losses": losses, "tops": tops}},
                    ckpt_out / f"{step}.pth",
                )

            hvp_solver.close()
            self.log.info(f"CPMTL done for {ckpt_name}")
