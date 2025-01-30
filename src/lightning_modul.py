# %%
from collections import defaultdict
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.inferers import SlidingWindowInferer
from pytorch_lightning import LightningModule
from torchmetrics import Dice
from torchmetrics.classification import (MulticlassConfusionMatrix,
                                         MulticlassF1Score)
from torchmetrics.classification.accuracy import (MulticlassAccuracy,
                                                  MultilabelAccuracy)
from torchmetrics.classification.auroc import MulticlassAUROC
from torchmetrics.classification.average_precision import \
    MulticlassAveragePrecision
from torchmetrics.classification.calibration_error import MulticlassCalibrationError

from src import jdt_losses

from . import model_utils

METRIC_MODE = {"val_loss": "min", "val_f1": "max", "val_b_acc": "max", "val_acc": "max"}


def initialize_torchmetrics(nn_module, num_classes, max_num_datasets=1, metrics="all"):

    if metrics == "all" or "accuracy" in metrics:
        nn_module.accuracy = nn.ModuleDict(
            {
                step: nn.ModuleList([MulticlassAccuracy(num_classes=num_classes, average="micro") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "b_accuracy" in metrics:

        nn_module.b_accuracy = nn.ModuleDict(
            {
                step: nn.ModuleList([MulticlassAccuracy(num_classes=num_classes, average="macro") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if "multilabel_accuracy" in metrics:
        nn_module.multilabel_accuracy = nn.ModuleDict(
            {
                step: nn.ModuleList([MultilabelAccuracy(num_labels=num_classes, average="macro") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "DICE" in metrics:

        nn_module.DICE = nn.ModuleDict(
            {
                step: nn.ModuleList([Dice(num_classes=num_classes, average="micro") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "b_DICE" in metrics:

        nn_module.b_DICE = nn.ModuleDict(
            {
                step: nn.ModuleList([Dice(num_classes=num_classes, average="macro") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "soft_DICE" in metrics:

        nn_module.soft_DICE = nn.ModuleDict(
            {
                step: nn.ModuleList([jdt_losses.SoftCorrectDICEMetric(average="mIoUD") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "b_soft_DICE" in metrics:

        nn_module.b_soft_DICE = nn.ModuleDict(
            {
                step: nn.ModuleList([jdt_losses.SoftCorrectDICEMetric(average="mIoUC") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "soft_DICE" in metrics:

        nn_module.soft_DICEDataset = nn.ModuleDict(
            {
                step: nn.ModuleList([jdt_losses.SoftDICECorrectAccuSemiMetric() for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "L1" in metrics:

        nn_module.L1 = nn.ModuleDict(
            {step: nn.ModuleList([model_utils.L1CalibrationMetric() for _ in range(max_num_datasets)]) for step in ["train_split", "val_split", "test_split"]}
        )

    if metrics == "all" or "conf_matrix" in metrics:

        nn_module.conf_matrix = nn.ModuleDict(
            {
                step: nn.ModuleList([MulticlassConfusionMatrix(num_classes=num_classes, normalize=None) for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "f1_score" in metrics:

        nn_module.f1_score = nn.ModuleDict(
            # Compute F1 Score only for tumor class
            {step: nn.ModuleList([MulticlassF1Score(num_classes) for _ in range(max_num_datasets)]) for step in ["train_split", "val_split", "test_split"]}
        )

    if metrics == "all" or "auroc" in metrics:

        # Only in val and test split
        nn_module.auroc = nn.ModuleDict(
            # Compute AUROC only for tumor class
            {
                step: nn.ModuleList([MulticlassAUROC(num_classes=2 if num_classes == 2 else num_classes, average="macro") for _ in range(max_num_datasets)])
                for step in ["val_split", "test_split"]
            }
        )

    if metrics == "all" or "avg_prec" in metrics:

        nn_module.avg_prec = nn.ModuleDict(
            # Compute AvgPREC Score only for tumor class
            {
                step: nn.ModuleList([MulticlassAveragePrecision(num_classes=num_classes, average="macro") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )

    if metrics == "all" or "ece" in metrics:

        nn_module.ece = nn.ModuleDict(
            {
                step: nn.ModuleList([MulticlassCalibrationError(num_classes=num_classes, n_bins=20, norm="l1") for _ in range(max_num_datasets)])
                for step in ["train_split", "val_split", "test_split"]
            }
        )


def log_metrics(l_model, split, d_idx, logits, y, losses):
    """Logs the metrics given a split"""

    assert split in ["train", "val", "test"]

    split_nn_dict = split + "_split"

    if y.dim() in [2, 4]:
        y_one_hot = torch.argmax(y, dim=1)
    else:
        y_one_hot = y

    sm = torch.nn.functional.softmax(logits, dim=1)
    preds = torch.argmax(sm, dim=1)

    if "loss" in l_model.metrics_to_track:

        for loss_name, loss in losses.items():

            loss_str = f"{split}_loss"
            loss_str = loss_str + "_" + loss_name if loss_name != "loss" else loss_str

            l_model.log(
                loss_str, loss, on_step=True if loss_name in ["loss", "KL"] else False, on_epoch=True, prog_bar=True if loss_name in ["loss", "KL"] else False
            )

    use_unique_max = l_model.use_unique_max

    if use_unique_max:

        label_max = torch.max(y, dim=1)[0]
        duplicated_max = torch.sum(y == label_max.unsqueeze(1), dim=1) > 1
        unique_max = ~duplicated_max

        if unique_max.sum() > 0:

            preds = preds[unique_max]
            y_one_hot = y_one_hot[unique_max]

            if "accuracy" in l_model.metrics_to_track:

                l_model.accuracy[split_nn_dict][d_idx](preds, y_one_hot)
                l_model.log(f"{split}_acc_unique_max", l_model.accuracy[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)

            if "b_accuracy" in l_model.metrics_to_track:
                l_model.b_accuracy[split_nn_dict][d_idx](preds, y_one_hot)
                l_model.log(
                    f"{split}_b_acc_unique_max",
                    l_model.b_accuracy[split_nn_dict][d_idx],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

            if "f1_score" in l_model.metrics_to_track:
                l_model.f1_score[split_nn_dict][d_idx](preds, y_one_hot)
                l_model.log(f"{split}_f1_unique_max", l_model.f1_score[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)

            if "DICE" in l_model.metrics_to_track:
                l_model.DICE[split_nn_dict][d_idx](preds, y_one_hot)
                l_model.log(f"{split}_DICE_unique_max", l_model.DICE[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)

            if "b_DICE" in l_model.metrics_to_track:
                l_model.b_DICE[split_nn_dict][d_idx](preds, y_one_hot)
                l_model.log(f"{split}_b_DICE_unique_max", l_model.b_DICE[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)
    else:
        if "multilabel_accuracy" in l_model.metrics_to_track:
            y_multilabel = y >= 0.5
            preds_multilabel = torch.nn.functional.sigmoid(logits) >= 0.5

            l_model.multilabel_accuracy[split_nn_dict][d_idx](preds_multilabel, y_multilabel)
            l_model.log(
                f"{split}_multilabel_acc",
                l_model.multilabel_accuracy[split_nn_dict][d_idx],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        RuntimeError()

    if "soft_DICE" in l_model.metrics_to_track:
        l_model.soft_DICE[split_nn_dict][d_idx](sm, y)
        l_model.log(f"{split}_soft_DICE", l_model.soft_DICE[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)

    if "b_soft_DICE" in l_model.metrics_to_track:
        l_model.b_soft_DICE[split_nn_dict][d_idx](sm, y)
        l_model.log(f"{split}_b_soft_DICE", l_model.b_soft_DICE[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)

    if "soft_DICE" in l_model.metrics_to_track:
        l_model.soft_DICEDataset[split_nn_dict][d_idx](sm, y)
        l_model.log(f"{split}_soft_DICEDataset", l_model.soft_DICEDataset[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)

    if "L1" in l_model.metrics_to_track:
        l_model.L1[split_nn_dict][d_idx](sm, y)
        l_model.log(f"{split}_L1", l_model.L1[split_nn_dict][d_idx], on_step=False, on_epoch=True, prog_bar=False)


LIT_SEGMENTER_LOSS_FUNCTIONS = {
    "CE": F.cross_entropy,
    "DICE": lambda out, y: model_utils.dice_loss_soft(out, y),
    "KL": lambda out, y: F.kl_div(out, y, reduction="mean", log_target=False),
}


class LitSegmenter(LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 2,
        lr: float = 1e-4,
        weight_decay=0.0,
        opti_metric="val_loss",
        patience=3,
        direction="minimize",
        loss_functions=[(1.0, "CE")],
        metrics_to_track: list[Literal["loss"]] = ["loss"],
        use_unique_max=True,
        sliding_window_in_test=False,
        label_remapper=None,
    ):
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.use_unique_max = use_unique_max

        self.loss_functions = []
        for loss_function in loss_functions:
            if isinstance(loss_function, (tuple, list)) and len(loss_function) == 2:
                weight, loss_function = loss_function
            else:
                weight = 1.0

            if isinstance(loss_function, str):
                loss_function = LIT_SEGMENTER_LOSS_FUNCTIONS[loss_function]

            self.loss_functions.append(nn.ParameterList([loss_function, weight]))

        self.loss_functions = nn.ModuleList(self.loss_functions)

        # Workaround for logging the hparams.yaml file, without acutally logging the whole weightmatrices to .yaml
        self.save_hyperparameters(ignore=["model"], logger=True)
        self.save_hyperparameters(logger=False)

        self.save_predictions = True
        self.predictions = defaultdict(list)
        initialize_torchmetrics(self, num_classes=num_classes, metrics=metrics_to_track, max_num_datasets=3)

        self.metrics_to_track = metrics_to_track

        if sliding_window_in_test:
            self.sw_inferer = SlidingWindowInferer(roi_size=(512, 512), sw_batch_size=1, overlap=0.5, mode="gaussian")
        else:
            self.sw_inferer = None

        self.label_remapper = label_remapper

    def forward(self, x, *args, **kwargs):
        out = self.model(x)

        return out

    def evaluate(self, batch, loss=True, inferer=None, label_remapper=None):
        x, y, ignore_mask = batch

        if inferer is not None:
            out = inferer(x, self)
        else:
            out = self.forward(x)

        if isinstance(out, list):
            out = out[0]

        if label_remapper is not None:
            out = label_remapper(out)
        # Cast to 64 to preserve softmax accuracy!
        # Jaeger, P. F., Lüth, C. T., Klein, L. & Bungert, T. J. A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification. Preprint at https://doi.org/10.48550/arXiv.2211.15259 (2023).
        out = out.to(dtype=torch.double)

        # Mask out background
        # y = y[~ignore_mask]
        # out = out[~ignore_mask]

        # Set all the values of the probability distribution to zero. This of course is not a prob distribution. However CE is zero for all inputs
        # through the formula: -sum_c_in_classes y_c * log(sm_c), where sm_c is the softmax value for class c. This results for 0 for all degenerate.

        # ignore_mask_expanded = ~ignore_mask.unsqueeze(1).expand(-1, y.size(1), -1, -1)
        # y = y[ignore_mask_expanded]
        # out = out[ignore_mask_expanded]

        num_classes = y.size(1)

        def flatten_pixel_dims(t):
            return t.permute(0, 2, 3, 1).reshape(-1, num_classes)

        out_org = out.detach().clone()

        out = flatten_pixel_dims(out)
        y = flatten_pixel_dims(y)

        ignore_mask = ignore_mask.reshape(-1).bool()

        out = out[~ignore_mask, :]
        y = y[~ignore_mask, :]

        losses = {}
        if loss:
            loss = 0.0
            for loss_function, weight in self.loss_functions:

                loss_name = self._get_loss_function_name(loss_function=loss_function)

                sub_loss = loss_function(out, y)
                losses[loss_name] = sub_loss
                loss += weight * sub_loss
        else:
            loss = None

        losses["loss"] = loss

        losses["KL"] = torch.nn.functional.kl_div(nn.functional.log_softmax(out, dim=1), y)

        return losses, out, y, out_org

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        losses, logits, y, _ = self.evaluate(batch)
        log_metrics(self, "train", dataloader_idx, logits, y, losses)
        predictions = {"loss": losses["loss"], "logits": logits, "label": y}

        return predictions

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        losses, logits, y, _ = self.evaluate(batch)
        log_metrics(self, "val", dataloader_idx, logits, y, losses)

        predictions = {"loss": losses["loss"], "logits": logits, "label": y}

        return predictions

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        if self.label_remapper is not None:
            if isinstance(self.label_remapper, (list, tuple, dict)):
                label_remapper = self.label_remapper[dataloader_idx]
            else:
                label_remapper = self.label_remapper

        losses, logits, y, org_logits = self.evaluate(batch, inferer=self.sw_inferer, label_remapper=label_remapper)
        log_metrics(self, "test", dataloader_idx, logits, y, losses)

        predictions = {"loss": losses["loss"], "logits": logits, "label": y}
        if self.save_predictions and dataloader_idx == 0:
            self.predictions[dataloader_idx].append(org_logits.cpu())

        return predictions

    def on_test_epoch_start(self):
        self.predictions.clear()

    def on_test_epoch_end(self):

        if self.save_predictions:
            for data_idx, full_logits in self.predictions.items():

                # logits = [out["logits"] for out in logged_outputs]
                # label = [out["label"] for out in logged_outputs]

                # logits = torch.cat(full_logits, dim=0)
                # label = torch.cat(label, dim=0)
                pass
                # self.predictions[data_idx] = logits

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Can't log in predict step

        if isinstance(batch, (list, tuple)):
            x, _, _ = batch
        else:
            x = batch

        if self.sw_inferer is not None:
            out = self.sw_inferer(x, self)
        else:
            out = self.forward(x)

        return {"logits": out}

        # return {"softmax": out_sm, "label": y}

    def configure_optimizers(self):

        optimizers = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

        patience = self.hparams["patience"]

        if patience > 0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode=self.hparams["direction"], patience=self.hparams["patience"])

            lr_schedulers = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": self.hparams["opti_metric"],
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMponitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }

            return ([optimizers], [lr_schedulers])

        return optimizers

    def _get_loss_function_name(self, loss_function):
        try:
            loss_name = str(loss_function.__name__)
        except:
            loss_name = str(type(loss_function).__name__)

        return loss_name


def efficientnet(size, classes, weights="IMAGENET1K_V1"):

    net = getattr(torchvision.models, f"efficientnet_b{int(size)}")(weights=weights)

    net.classifier[1] = nn.Linear(net.classifier[1].in_features, classes)

    return net


class LitClassifier(LitSegmenter):

    def evaluate(self, batch, loss=True):

        x, y, _ = batch
        out = self.forward(x)
        if isinstance(out, list):
            out = out[0]

        # Cast to 64 to preserve softmax accuracy!
        # Jaeger, P. F., Lüth, C. T., Klein, L. & Bungert, T. J. A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification. Preprint at https://doi.org/10.48550/arXiv.2211.15259 (2023).
        out = out.to(dtype=torch.double)
        losses = {}
        if loss:
            loss = 0.0
            for loss_function, weight in self.loss_functions:

                loss_name = self._get_loss_function_name(loss_function=loss_function)

                sub_loss = loss_function(out, y)
                losses[loss_name] = sub_loss
                loss += weight * sub_loss
        else:
            loss = None

        losses["loss"] = loss

        losses["KL"] = torch.nn.functional.kl_div(nn.functional.log_softmax(out, dim=1), y)

        return losses, out, y
