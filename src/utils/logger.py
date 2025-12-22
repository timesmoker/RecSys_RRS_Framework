# src/utils/logger.py
from omegaconf import OmegaConf
from datetime import datetime
import os


class Logger:
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.use_wandb = cfg.get("wandb", False)
        self.verbose = cfg.get("verbose", True)

        if self.use_wandb:
            import wandb
            self.wandb = wandb
        else:
            self.wandb = None

    # -------------------------
    # config
    # -------------------------
    def save_args(self):
        path = f"{self.log_dir}/config.yaml"
        with open(path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        if self.wandb:
            self.wandb.config.update(
                OmegaConf.to_container(self.cfg, resolve=True),
                allow_val_change=True
            )

    # -------------------------
    # train
    # -------------------------
    def log_train_metrics(self, metrics: dict, step: int):
        payload = {f"train/{k}": v for k, v in metrics.items()}

        if self.wandb:
            self.wandb.log(payload, step=step)

    def log_valid_metrics(self, metrics: dict, step: int):
        payload = {f"valid/{k}": v for k, v in metrics.items()}

        if self.wandb:
            self.wandb.log(payload, step=step)

    # -------------------------
    # predict
    # -------------------------
    def log_predict_info(self, info: dict, step: int | None = None):
        payload = {f"predict/{k}": v for k, v in info.items()}

        if self.wandb:
            if step is None:
                self.wandb.log(payload)
            else:
                self.wandb.log(payload, step=step)

        path = f"{self.log_dir}/predict_info.txt"
        with open(path, "a") as f:
            f.write(f"\n[{datetime.now()}]\n")
            for k, v in info.items():
                f.write(f"{k}: {v}\n")

        summary = ", ".join(f"{k}={v}" for k, v in info.items())
        self._print(f"Predict logged: {summary}")


    # -------------------------
    # artifact
    # -------------------------
    def log_artifact(self, path: str, name: str):
        if self.wandb:
            self.wandb.save(path)

        record = f"{self.log_dir}/artifacts.txt"
        with open(record, "a") as f:
            f.write(f"{name}: {path}\n")

        self._print(f"Artifact saved [{name}]: {path}")


    def _print(self, msg: str):
        if self.verbose:
            print(f"[LOG] {msg}")

