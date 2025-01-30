import argparse
import os
import warnings
from pathlib import Path
from typing import Literal, Optional, Union

import hydra
import omegaconf
import torch
import wandb
import wandb.plot
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import src.augmentations as augmentations
import src.model_utils as model_utils
import src.tree_loss as tree_loss
from src.lightning_modul import LitClassifier, LitSegmenter


def test(
    checkpoint: Union[str, Path],
    cfg: DictConfig,
    task: Literal["test", "predict"] = "test",
    save_path: Optional[str] = None,
    eval_on: str = "test",
    log_wandb: bool = True,
    log_wandb_entity: str = "dkfz",
    log_wandb_project: str = "GleasonXAI",
):

    if task == "predict" and save_path is None:
        raise RuntimeError("Provided no save_path when task == predict!")
    BATCH_SIZE = cfg.dataloader.batch_size
    NUM_WORKERS = cfg.dataloader.num_workers
    EFFECTIVE_BATCH_SIZE = cfg.dataloader.effective_batch_size
    assert EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0
    LR = min(cfg.optimization.lr * EFFECTIVE_BATCH_SIZE / 2, 1e-3)
    WEIGHT_DECAY = cfg.optimization.weight_decay
    PATIENCE = cfg.optimization.patience
    MAX_EPOCHS = cfg.trainer.max_epochs
    assert "EXPERIMENT_LOCATION" in os.environ
    LOG_DIR = Path(os.environ["EXPERIMENT_LOCATION"])
    EXPERIMENT_NAME = cfg.logger.experiment_name
    SAVE_METRIC = cfg.save_metric
    ACCELERATOR = cfg.trainer.accelerator
    LABEL_LEVEL = cfg.dataset.label_level
    DIRECTION = cfg.metric_direction
    DIRECTION_SHORT = "min" if DIRECTION == "minimize" else "max"

    TASK = cfg.task
    AUGMENTATION_TEST = getattr(cfg.augmentations, "test", cfg.augmentations.eval)

    USE_SW = cfg.augmentations.use_sw if task != "test" else True
    LOG_WANDB = cfg.logger.get("log_wandb", False)
    BATCH_LIMIT = cfg.trainer.get("limit_batches", 1.0)
    BATCH_LIMIT = {f"limit_{split}_batches": BATCH_LIMIT for split in ["train", "val", "test"]}

    # Transforms
    transforms_test = augmentations.AUGMENTATIONS[AUGMENTATION_TEST]

    dataset_test = hydra.utils.instantiate(cfg.dataset, split="test", transforms=transforms_test)
    NUM_CLASSES = dataset_test.num_classes

    remapped_datasets = [
        hydra.utils.instantiate(cfg.dataset, split="test", transforms=transforms_test, label_level=ll) for ll in range(LABEL_LEVEL - 1, -1, -1)
    ]

    test_datasets = [dataset_test, *remapped_datasets]

    test_dataloaders = [
        DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False) for dataset_test in test_datasets
    ]

    net = hydra.utils.instantiate(cfg.model, classes=NUM_CLASSES)

    loss_functions = hydra.utils.instantiate(cfg.loss_functions)

    if not isinstance(loss_functions, list):
        loss_functions = [loss_functions]

    for loss_function in loss_functions:
        if isinstance(loss_function, tree_loss.TreeLoss):
            loss_function.init_runtime(dataset_test.exp_numbered_lvl_remapping, start_level=dataset_test.label_level)

    if TASK == "segmentation":
        lit_mod = LitSegmenter(
            net,
            num_classes=NUM_CLASSES,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            opti_metric=SAVE_METRIC,
            patience=PATIENCE,
            direction=DIRECTION_SHORT,
            metrics_to_track=["loss", "accuracy", "b_accuracy", "DICE", "soft_DICE", "b_DICE", "b_soft_DICE", "L1"],
            loss_functions=loss_functions,
            sliding_window_in_test=USE_SW,
        )
    elif TASK == "classification":
        lit_mod = LitClassifier(
            net,
            num_classes=NUM_CLASSES,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            opti_metric=SAVE_METRIC,
            patience=PATIENCE,
            direction=DIRECTION_SHORT,
            metrics_to_track=["loss", "multilabel_accuracy", "multilabel_b_accuracy"],
            loss_functions=loss_functions,
        )
    else:
        raise RuntimeError(f"Task: {TASK} not defined!")

    remappers = {
        i: model_utils.LabelRemapper(dataset_test.exp_numbered_lvl_remapping, LABEL_LEVEL, ll) for i, ll in enumerate(range(LABEL_LEVEL - 1, -1, -1), start=1)
    }
    remappers[0] = None
    lit_mod.label_remapper = remappers
    lit_mod.save_predictions = False

    if task == "test":
        if eval_on != "test":
            raise RuntimeError()

        logger = []

        if LOG_WANDB:

            entity = log_wandb_entity
            project = log_wandb_project

            # Find the run to log into.
            runs = wandb.Api().runs(
                f"{entity}/{project}",
            )
            runs = [run for run in runs if run.name == EXPERIMENT_NAME]
            assert len(runs) == 1
            existing_run = runs[0]
            run_id = existing_run.id

            # Initialize W&B run with the specific run ID
            wandb.init(project=project, id=run_id, resume="must", reinit=True)
            logger += [WandbLogger(save_dir=str(LOG_DIR / "wandb"))]

        trainer = Trainer(
            accelerator=ACCELERATOR,
            max_epochs=MAX_EPOCHS,
            precision="16-mixed",
            inference_mode=True,
            logger=logger,
            callbacks=None,
            log_every_n_steps=1,
            enable_checkpointing=False,
        )

        trainer.test(lit_mod, dataloaders=test_dataloaders, ckpt_path=checkpoint)

        if log_wandb:
            wandb.finish()

    elif task == "predict":

        logger = []

        trainer = Trainer(
            accelerator=ACCELERATOR,
            max_epochs=MAX_EPOCHS,
            precision="16-mixed",
            inference_mode=True,
            logger=logger,
            callbacks=None,
            log_every_n_steps=1,
            enable_checkpointing=False,
        )
        preds = trainer.predict(lit_mod, dataloaders=test_dataloaders[0], ckpt_path=checkpoint)

        combined_preds = [pred["logits"].detach().cpu().squeeze() for pred in preds]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(combined_preds, str(save_path.absolute().expanduser()))

    test_metrics = trainer.callback_metrics

    return test_metrics


if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument("--experiment_path", default="/home/Documents/GleasonXAI")
    parse.add_argument("--checkpoint", default="GleasonFinal2/label_level1")
    parse.add_argument("--config")
    parse.add_argument("--no_logging", default=False, action="store_true")
    parse.add_argument("--glob_checkpoints", default="SoftDiceBalanced*")
    parse.add_argument("--dry_run", default=False, action="store_true")
    parse.add_argument("--task", default="predict", choices=["test", "predict"])
    parse.add_argument("--save_path")
    parse.add_argument("--eval_on", default="test", choices=["train", "val", "test"])
    parse.add_argument("--wandb_entity", default="dkfz")
    parse.add_argument("--wandb_project", default="GleasonXAI")

    args = parse.parse_args()

    assert "DATASET_LOCATION" in os.environ, f"Environment variable DATASET_LOCATION not set!"
    assert Path(os.environ["DATASET_LOCATION"]).exists(), f"DATASET_LOCATION {os.environ['DATASET_LOCATION']} does not exist!"

    checkpoint = Path(args.experiment_path) / Path(args.checkpoint)
    assert checkpoint.exists(), f"Checkpoint path {checkpoint} does not exist!"

    if args.glob_checkpoints is not None:
        checkpoints = list(checkpoint.glob(args.glob_checkpoints))
        assert len(checkpoints) > 0

    else:
        checkpoints = [checkpoint]

    print(f"Starting evaluation: \n Number of runs: {len(checkpoints)} \n ---------------------------------------- ")

    for checkpoint in checkpoints:

        if checkpoint.is_dir():

            if (checkpoint / "version_0").exists() and len(list(checkpoint.glob("version_*"))) == 1:
                checkpoint = checkpoint / "version_0"

            if (checkpoint / "best_model.ckpt").exists():
                checkpoint = checkpoint / "best_model.ckpt"
            elif (checkpoint / "checkpoints" / "best_model.ckpt").exists():
                checkpoint = checkpoint / "checkpoints" / "best_model.ckpt"
            else:
                warnings.warn(f"No checkpoint found for checkpoint path {checkpoint}.")
                continue

        if args.config is None:
            config = checkpoint.parents[1] / "logs" / "config.yaml"

        if args.save_path is None:

            save_path = checkpoint.parents[1] / "preds" / f"pred_{args.eval_on}.pt"

        assert config.exists(), f"Config in {config} does not exist!"

        config = omegaconf.OmegaConf.load(config)

        if args.dry_run:
            print(checkpoint)
        else:

            try:
                metrics = test(
                    checkpoint,
                    config,
                    task=args.task,
                    save_path=save_path,
                    eval_on=args.eval_on,
                    log_wandb=not args.no_logging,
                    log_wandb_entity=args.wandb_entity,
                    log_wandb_project=args.wandb_project,
                )
                print(metrics)

            except Exception as e:
                warnings.warn(f"Error {e} while evaluating {checkpoint}")
