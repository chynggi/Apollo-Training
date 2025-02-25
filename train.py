import json
import os
import argparse
import pytorch_lightning as pl
import torch
import warnings
import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from look2hear.utils import print_only

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig, model_path: str) -> None:
    # seed everything
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas, expdir=os.path.join(cfg.exp.dir, cfg.exp.name))

    # instantiate model
    print_only(f"Instantiating AudioNet <{cfg.model._target_}>")
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model)

    print_only(f"Instantiating Discriminator <{cfg.discriminator._target_}>")
    discriminator: torch.nn.Module = hydra.utils.instantiate(cfg.discriminator)

    # instantiate optimizer
    print_only(f"Instantiating optimizer <{cfg.optimizer_g._target_}>")
    optimizer_g: torch.optim = hydra.utils.instantiate(cfg.optimizer_g, params=model.parameters())
    optimizer_d: torch.optim = hydra.utils.instantiate(cfg.optimizer_d, params=discriminator.parameters())

    # instantiate scheduler
    print_only(f"Instantiating scheduler <{cfg.scheduler_g._target_}>")
    scheduler_g: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler_g, optimizer=optimizer_g)
    scheduler_d: torch.optim.lr_scheduler = hydra.utils.instantiate(cfg.scheduler_d, optimizer=optimizer_d)

    # instantiate loss
    print_only(f"Instantiating loss <{cfg.loss_g._target_}>")
    loss_g: torch.nn.Module = hydra.utils.instantiate(cfg.loss_g)
    loss_d: torch.nn.Module = hydra.utils.instantiate(cfg.loss_d)

    # instantiate metrics
    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics: torch.nn.Module = hydra.utils.instantiate(cfg.metrics)

    # instantiate callbacks
    callbacks = []
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping <{cfg.early_stopping._target_}>")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint <{cfg.checkpoint._target_}>")
        checkpoint: pl.callbacks.ModelCheckpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)

    # instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger: WandbLogger = hydra.utils.instantiate(cfg.logger)

    # instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        config=cfg.system,
        model=model,
        discriminator=discriminator,
        loss_func={"g": loss_g,"d": loss_d},
        metrics=metrics,
        optimizer=[optimizer_g, optimizer_d],
        scheduler=[scheduler_g, scheduler_d],
    )

    # instantiate trainer
    print_only(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='gloo' if os.name == "nt" else 'nccl'),
    )

    # train and save the best model
    trainer.fit(system, datamodule=datamodule, ckpt_path=model_path if model_path else None)
    print_only("Training finished!")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path, weights_only=False)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    if trainer.is_global_zero:
        torch.save(to_save, os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/apollo.yaml",)
    parser.add_argument("-m", "--model", type=str, default=None, help="Path to the checkpoint model for resuming training")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))

    if "MASTER_ADDR" not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if "MASTER_PORT" not in os.environ:
        os.environ['MASTER_PORT'] = '8001'
    if "USE_LIBUV" not in os.environ:
        os.environ["USE_LIBUV"] = "0"

    train(cfg, model_path=args.model)
