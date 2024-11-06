import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

import src.augmentations as augmentations
import src.tree_loss as tree_loss
import wandb
from src.lightning_modul import LitClassifier, LitSegmenter
from src.model_utils import LabelRemapper, PatchedOptunaCallback

# We need to include modules that can be accessed by hydra through the config files. Gives linter warnings, but we cant change that.


def train(cfg: DictConfig, trial=None):

    BATCH_SIZE = cfg.dataloader.batch_size
    NUM_WORKERS = cfg.dataloader.num_workers
    EFFECTIVE_BATCH_SIZE = cfg.dataloader.effective_batch_size
    assert EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    # Was previously always defined for batch_size = 2. We now scale it proportional to the effective batchsize.
    LR = min(cfg.optimization.lr * EFFECTIVE_BATCH_SIZE/2, 1e-3)  # Clip it to stable region. Found out through super hard to debug bug.
    WEIGHT_DECAY = cfg.optimization.weight_decay
    PATIENCE = cfg.optimization.patience
    MAX_EPOCHS = cfg.trainer.max_epochs
    LOG_DIR = Path(os.environ["EXPERIMENT_LOCATION"])
    EXPERIMENT_NAME = cfg.logger.experiment_name
    SAVE_METRIC = cfg.save_metric
    ACCELERATOR = cfg.trainer.accelerator
    LABEL_LEVEL = cfg.dataset.label_level
    DIRECTION = cfg.metric_direction
    DIRECTION_SHORT = "min" if DIRECTION == "minimize" else "max"

    HPARAM_SEARCH_RUNNING = cfg.get("optuna", None) is not None

    TASK = cfg.task
    AUGMENTATION_TRAIN = cfg.augmentations.train
    AUGMENTATION_VAL = cfg.augmentations.eval

    AUGMENTATION_TEST = getattr(cfg.augmentations, 'test', cfg.augmentations.eval)

    USE_SW = cfg.augmentations.use_sw
    SAVE_TEST_PREDS = cfg.logger.get("save_test_preds", False)

    SAVE_MODEL_CHECKPOINTS = cfg.logger.get("save_model_checkpoints", True)
    if HPARAM_SEARCH_RUNNING:
        SAVE_MODEL_CHECKPOINTS = False

    LOG_WANDB = cfg.logger.get("log_wandb", True)
    BATCH_LIMIT = cfg.trainer.get("limit_batches", 1.0)
    BATCH_LIMIT = {f"limit_{split}_batches": BATCH_LIMIT for split in ["train", "val", "test"]}

    # Transforms
    transforms_train = augmentations.AUGMENTATIONS[AUGMENTATION_TRAIN]
    transforms_val_test = augmentations.AUGMENTATIONS[AUGMENTATION_VAL]
    transforms_test = augmentations.AUGMENTATIONS[AUGMENTATION_TEST]

    dataset = hydra.utils.instantiate(cfg.dataset, split="train", transforms=transforms_train)
    dataset_val = hydra.utils.instantiate(cfg.dataset, split="val", transforms=transforms_val_test)

    NUM_CLASSES = dataset.num_classes

    dataloader_train = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True)

    net = hydra.utils.instantiate(cfg.model, classes=NUM_CLASSES)

    loss_functions = hydra.utils.instantiate(cfg.loss_functions)

    if not isinstance(loss_functions, list):
        loss_functions = [loss_functions]

    for loss_function in loss_functions:
        if isinstance(loss_function, tree_loss.TreeLoss):
            loss_function.init_runtime(dataset.exp_numbered_lvl_remapping, start_level=dataset.label_level)

    if TASK == "segmentation":
        lit_mod = LitSegmenter(net, num_classes=NUM_CLASSES, lr=LR, weight_decay=WEIGHT_DECAY, opti_metric=SAVE_METRIC,
                               patience=PATIENCE, direction=DIRECTION_SHORT, metrics_to_track=["loss", "accuracy", "b_accuracy", "DICE", "soft_DICE", "b_DICE", "b_soft_DICE", "L1"], loss_functions=loss_functions, sliding_window_in_test=USE_SW)
    elif TASK == "classification":
        lit_mod = LitClassifier(net, num_classes=NUM_CLASSES, lr=LR, weight_decay=WEIGHT_DECAY, opti_metric=SAVE_METRIC,
                                patience=PATIENCE, direction=DIRECTION_SHORT, metrics_to_track=["loss", "multilabel_accuracy", "multilabel_b_accuracy"], loss_functions=loss_functions)
    else:
        raise RuntimeError(f"Task: {TASK} not defined!")

    logger = [TensorBoardLogger(LOG_DIR, name=EXPERIMENT_NAME, sub_dir="logs")]

    if LOG_WANDB:
        wandb.init(project="GleasonXAI", config=OmegaConf.to_container(cfg, resolve=False), name=EXPERIMENT_NAME, group=EXPERIMENT_NAME[:-2], reinit=True)
        logger += [WandbLogger(save_dir=str(LOG_DIR/"wandb"))]


    model_checkpoint = ModelCheckpoint(monitor=SAVE_METRIC, auto_insert_metric_name=True, save_last=True, save_top_k=1, mode=DIRECTION_SHORT)

    callbacks = []
    if SAVE_MODEL_CHECKPOINTS:
        callbacks += [model_checkpoint]

    if HPARAM_SEARCH_RUNNING:
        callbacks += [PatchedOptunaCallback(trial=trial, monitor=SAVE_METRIC)]

    trainer = Trainer(accelerator=ACCELERATOR, max_epochs=MAX_EPOCHS, precision="16-mixed",  inference_mode=False,
                      logger=logger, callbacks=callbacks, log_every_n_steps=1, accumulate_grad_batches=ACCUMULATION_STEPS, enable_checkpointing=not HPARAM_SEARCH_RUNNING, **BATCH_LIMIT)

    # Dump config from hydra.
    Path(trainer.log_dir).mkdir(parents=True, exist_ok=True)
    config_dump_path = Path(trainer.log_dir)/"config.yaml"


    with open(config_dump_path, "w") as f:
        OmegaConf.save(cfg, f)

    trainer.fit(lit_mod, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    train_metrics = trainer.callback_metrics


    if HPARAM_SEARCH_RUNNING:
        if LOG_WANDB:
            wandb.finish()
        return train_metrics.get(SAVE_METRIC)

    os.link(model_checkpoint.best_model_path, Path(model_checkpoint.best_model_path).parent/"best_model.ckpt")
    lit_mod.save_predictions = SAVE_TEST_PREDS

    dataset_test = hydra.utils.instantiate(cfg.dataset, split="test", transforms=transforms_test)

    remapped_datasets = [hydra.utils.instantiate(cfg.dataset, split="test", transforms=transforms_test, label_level=ll) for ll in range(LABEL_LEVEL-1, -1, -1)]

    remappers = {i: LabelRemapper(dataset_test.exp_numbered_lvl_remapping, LABEL_LEVEL, ll) for i, ll in enumerate(range(LABEL_LEVEL-1, -1, -1), start=1)}
    remappers[0] = None
    lit_mod.label_remapper = remappers

    test_datasets = [dataset_test, *remapped_datasets]

    test_dataloaders = [DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
                        for dataset_test in test_datasets]



    trainer.test(lit_mod, dataloaders=test_dataloaders, ckpt_path=model_checkpoint.best_model_path)
    test_metrics = trainer.callback_metrics

    if SAVE_TEST_PREDS:

        preds = lit_mod.predictions

        pred_save_dir = Path(trainer.log_dir).parent/"preds"
        pred_save_dir.mkdir(parents=True, exist_ok=True)

        if len(preds.keys()) == 1:
            torch.save(preds[0], str((pred_save_dir/"pred_test.pt").absolute().expanduser()))
        else:
            for data_idx, pred in preds.items():
                torch.save(pred, str((pred_save_dir/f"pred_test_{data_idx}.pt").absolute().expanduser()))

    if LOG_WANDB:
        wandb.finish()

    return test_metrics
