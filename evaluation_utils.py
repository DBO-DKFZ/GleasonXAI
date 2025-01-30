import math
from itertools import zip_longest
from pathlib import Path
from textwrap import wrap
from typing import Literal

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torchmetrics import Accuracy, ConfusionMatrix, Dice
from tqdm import tqdm

import src.augmentations as augmentations
from src.augmentations import (basic_transforms_val_test_scaling512,
                               normalize_only_transform)
from src.gleason_data import GleasonX
from src.gleason_utils import create_composite_plot
from src.jdt_losses import SoftDICECorrectAccuSemiMetric
from src.model_utils import L1CalibrationMetric
from src.tree_loss import generate_label_hierarchy

# VISUALISATIONS


def create_ensemble_plot(dataset, idx, ensemble_predictions, individual_predictions, show_ensemble_preds=0):

    img, masks, background_mask = dataset.__getitem__(idx, False)
    org_img = augmentations.basic_transforms_val_test_colorpreserving(image=np.array(dataset.get_raw_image(idx)))["image"]
    np_seg = np.array(ensemble_predictions[idx].argmax(dim=0)).astype(np.uint8)

    masks = {"segmentation": np_seg} | {f"Annotator {i}": mask.astype(np.uint8) for i, mask in enumerate(masks)}

    sub_ensemble_preds = []
    rnd_ensemble_subset = np.random.permutation(show_ensemble_preds)

    for i in rnd_ensemble_subset:
        np_seg = np.array(individual_predictions[i, idx].argmax(dim=0)).astype(np.uint8)
        sub_ensemble_preds.append(np_seg)

    if len(sub_ensemble_preds) > 0:
        masks = {f"ensemble_pred_{i}": sub_ensemble_pred for i, sub_ensemble_pred in enumerate(sub_ensemble_preds)} | masks

    f = create_composite_plot(dataset, org_img, masks, background_mask.astype(np.uint8), label_level=1, only_show_existing_annotation=True)


def composite_prediction_plot(predictions, dataset, indices, mask_background=False, full_legend=True):

    num_plots = len(indices)
    rows = int(np.ceil(num_plots / 3))  # Assuming 3 columns per row, adjust as needed

    for idx in indices:

        img, masks, background_mask = dataset.__getitem__(idx, False)
        org_img = augmentations.basic_transforms_val_test_colorpreserving(image=np.array(dataset.get_raw_image(idx)))["image"]

        background = background_mask if mask_background else None

        out = predictions[idx]
        out = torch.nn.functional.softmax(out, 0)
        np_seg = np.array(out.argmax(dim=0)).astype(np.uint8)
        f = create_composite_plot(
            dataset,
            org_img,
            {"segmentation": np_seg} | {f"Annotator {i}": mask for i, mask in enumerate(masks)},
            background,
            only_show_existing_annotation=not full_legend,
        )

        return f


def create_single_class_acti_maps(
    predictions,
    dataset,
    idx,
    plot_mode: Literal["heatmap", "contourf", "contour", "thresholded"] = "contourf",
    thresholds: list[float] = None,
    strip_background=False,
    plot=True,
):

    img, masks, background_mask = dataset.__getitem__(idx, False)

    out = predictions[idx]
    out = torch.nn.functional.softmax(out, 0)

    np_seg = np.array(out.argmax(dim=0)).astype(np.uint8)

    org_img = np.array(dataset.get_raw_image(idx).resize(np_seg.shape))

    colormap = ListedColormap(dataset.colormap.colors)
    num_class_to_vis = dataset.num_classes

    if strip_background:

        for mask in masks:
            mask += 1
            mask[background_mask] = 0

        np_seg += 1
        np_seg[background_mask] = 0

        out[:, torch.tensor(background_mask).bool()] = 0.0

        colormap = ListedColormap(np.concatenate([np.array([[0.0, 0.0, 0.0, 1.0]]), dataset.colormap.colors]))
        num_class_to_vis = dataset.num_classes + 1

    f, axes = plt.subplots(2, 3 + math.ceil(dataset.num_classes / 2), sharex=False, sharey=False, constrained_layout=False, figsize=(12, 4))
    f.tight_layout()

    axes[0, 0].imshow(org_img)
    axes[0, 0].set_title("Image", size=7)
    axes[0, 0].set_axis_off()

    # axes[1, 0].imshow(img)
    axes[1, 0].imshow(np_seg.astype(int), cmap=colormap, vmin=0, vmax=num_class_to_vis - 1, interpolation_stage="rgba")
    axes[1, 0].set_title("Segmentation", size=7)
    axes[1, 0].set_axis_off()

    for sub_ax, mask in zip_longest(list(axes[:, 1:3].flatten()), masks):
        sub_ax.set_axis_off()

        if mask is not None:
            sub_ax.imshow(mask.astype(int), cmap=colormap, vmin=0, vmax=num_class_to_vis - 1, interpolation_stage="rgba")
            sub_ax.set_title("Annotation", size=7)

    for i in range(dataset.num_classes):
        active_axis = axes[:, 3:].flatten()[i]

        class_out = out[i, :].detach().numpy()

        if strip_background:
            class_out[background_mask] = 0

        temp_colormap = ListedColormap(np.concatenate([np.array([[0.0, 0.0, 0.0, 1.0]]), dataset.colormap.colors[i].reshape(1, -1)]))

        match plot_mode:
            case "heatmap":
                active_axis.matshow(class_out, cmap=cm["Grays"].reversed(), vmin=out[out != 0.0].min(), vmax=out.max())
            case "multilabel":
                active_axis.matshow(class_out >= 0.32, cmap=temp_colormap)
            case "contour":
                active_axis.contour(np.flipud(class_out), cmap=cm["Grays"].reversed(), vmin=0.0, vmax=out.max())
            case "contourf":
                active_axis.contourf(np.flipud(class_out), cmap=cm["Grays"].reversed(), vmin=0.0, vmax=out.max())
            case "thresholded":
                active_axis.matshow(class_out > thresholds[i], cmap=cm["Grays"].reversed())

        if i == 0:
            title = "Benign"
        else:
            exp = dataset.explanations[i - 1]
            classes_named = [
                "benign tissue",
                "3 - individual glands",
                "3 - compressed glands",
                "4 - poorly formed glands",
                "4 - cribriform glands",
                "4 - glomeruloid glands",
                "5 - group of tumor cells",
                "5 - single cells",
                "5 - cords",
                "5 - comedenocrosis",
            ]
            exp_short = classes_named[i]
            title = "\n".join(wrap(str(dataset.exp_grade_mapping[exp]) + ": " + exp_short, width=20))

        active_axis.set_title(title, size=7)
        active_axis.set_axis_off()

    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    if plot:
        plt.show()
    else:
        return f, axes


def create_simple_seg_anno_plot(
    predictions,
    dataset,
    idx,
    plot_mode: Literal["heatmap", "contourf", "contour", "thresholded"] = "contourf",
    thresholds: list[float] = None,
    strip_background=False,
    plot=True,
):

    img, masks, background_mask = dataset.__getitem__(idx, False)

    out = predictions[idx]

    out = torch.nn.functional.softmax(out, 0)

    np_seg = np.array(out.argmax(dim=0)).astype(np.uint8)

    org_img = np.array(dataset.get_raw_image(idx).resize(np_seg.shape))

    colormap = ListedColormap(dataset.colormap.colors)
    num_class_to_vis = dataset.num_classes

    if strip_background:

        for mask in masks:
            mask += 1
            mask[background_mask] = 0

        np_seg += 1
        np_seg[background_mask] = 0

        out[:, torch.tensor(background_mask).bool()] = 0.0

        colormap = ListedColormap(np.concatenate([np.array([[0.0, 0.0, 0.0, 1.0]]), dataset.colormap.colors]))
        num_class_to_vis = dataset.num_classes + 1

    f, axes = plt.subplots(1, 5, sharex=False, sharey=False, constrained_layout=False, figsize=(12, 4))
    f.tight_layout()
    axes[0].imshow(org_img)
    axes[0].set_title("Image", size=7)
    axes[0].set_axis_off()

    axes[1].imshow(np_seg.astype(int), cmap=colormap, vmin=0, vmax=num_class_to_vis - 1, interpolation_stage="rgba")
    axes[1].set_title("Segmentation", size=7)
    axes[1].set_axis_off()

    for sub_ax, mask in zip_longest(axes[2:], masks):

        if sub_ax is None:
            continue
        sub_ax.set_axis_off()

        if mask is not None:
            sub_ax.imshow(mask.astype(int), cmap=colormap, vmin=0, vmax=num_class_to_vis - 1, interpolation_stage="rgba")
            sub_ax.set_title("Annotation", size=7)
    plt.tight_layout()


def create_multi_seg_anno_plot(predictions, dataset, idcs, strip_background=False, legend=False, class_names=None):

    if class_names is None:
        class_names = dataset.classes_named
    num_plots = len(idcs)

    f, top_axes = plt.subplots(num_plots, 5, sharex=False, sharey=False, constrained_layout=False, figsize=(10, 2 * num_plots))
    f.tight_layout()

    encountered_classes = set()

    if strip_background:
        colormap = ListedColormap(np.concatenate([np.array([[0.0, 0.0, 0.0, 1.0]]), dataset.colormap.colors]))
        num_class_to_vis = dataset.num_classes + 1
    else:
        colormap = ListedColormap(dataset.colormap.colors)
        num_class_to_vis = dataset.num_classes

    for i, idx in enumerate(idcs):

        axes = top_axes[i]

        _, masks, background_mask = dataset.__getitem__(idx, False)

        out = predictions[idx]

        out = torch.nn.functional.softmax(out, 0)

        np_seg = np.array(out.argmax(dim=0)).astype(np.uint8)

        org_img = np.array(dataset.get_raw_image(idx).resize(np_seg.shape))

        if strip_background:

            for mask in masks:
                mask += 1
                mask[background_mask] = 0

            np_seg += 1
            np_seg[background_mask] = 0

            out[:, torch.tensor(background_mask).bool()] = 0.0

        encountered_classes |= set(np.unique(np_seg))

        for mask in masks:
            encountered_classes |= set(np.unique(mask))

        axes[0].imshow(org_img)
        axes[0].set_axis_off()

        axes[1].imshow(np_seg.astype(int), cmap=colormap, vmin=0, vmax=num_class_to_vis - 1, interpolation_stage="rgba")
        axes[1].set_axis_off()

        for sub_ax, mask in zip_longest(axes[2:], masks):

            if sub_ax is None:
                continue
            sub_ax.set_axis_off()

            if mask is not None:
                sub_ax.imshow(mask.astype(int), cmap=colormap, vmin=0, vmax=num_class_to_vis - 1, interpolation_stage="rgba")

    if legend:
        if strip_background:
            legend_handels = [mpatches.Patch(color=np.array([0.0, 0.0, 0.0, 1.0]), label=f"Background")]
            legend_handels += [
                mpatches.Patch(
                    color=colormap(dataset.classes_number_mapping[cls] + 1), label=cls_renamed if len(cls_renamed) < 60 else cls_renamed[:60] + "..."
                )
                for cls, cls_renamed in zip(dataset.classes_named, class_names)
                if dataset.classes_number_mapping[cls] + 1 in encountered_classes
            ]
        else:
            legend_handels = [
                mpatches.Patch(
                    color=colormap(dataset.classes_number_mapping[cls]), label=cls_renamed[:40] if len(cls_renamed) < 60 else cls_renamed[:60] + "..."
                )
                for cls, cls_renamed in zip(dataset.classes_named, class_names)
                if dataset.classes_number_mapping[cls] in encountered_classes
            ]

        f.legend(handles=legend_handels, loc="center left", fontsize=12, bbox_to_anchor=[1.0, 0.5])
    print([dataset.classes_named[e - 1] for e in encountered_classes if e != 0])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.05)


# Hardcoded settings for our experiments that setup the model and dataset accordingly.


def get_model_settings(base_path, selected_model, remap_ll0):
    label_level = 1 if "ll1" in selected_model else 0
    eval_on = "test"

    num_classes = 10 if (label_level == 1 and not remap_ll0) else 4

    if remap_ll0:
        assert "ll1" in selected_model
        label_level = 0

    data_opts = "final" if "final" in selected_model else "final" if "final2" in selected_model else "org"

    if selected_model == "ens/ll1/SDB":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/SoftDICEBalancedNoZoomCont-{i}/version_0/") for i in [1, 2, 3, 4]]
        model_paths += [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/SoftDICEBalancedNoZoom-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "ens/ll1/Tree":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/FinalTreeLossNoZoomCont-{i}/version_0/") for i in [1, 2, 3, 4]]
        model_paths += [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/NoZoomFinalTreeLoss-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "ll1/CE":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/NoZoomFinalCE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "ll1/Tree":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/NoZoomFinalTreeLoss-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "ll1/SDB":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level1/HoleMask/SoftDICEBalancedNoZoom-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "ll0/CE":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level0/HoleMask/NoZoomFinalCE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "ll0/SDB":
        model_paths = [Path(f"GleasonBackgroundMasking/label_level0/HoleMask/SoftDICEBalancedNoZoom-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll0/CE":
        model_paths = [Path(f"GleasonFinal/label_level0/CE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll1/CE":
        model_paths = [Path(f"GleasonFinal/label_level1/CE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll0/SDB":
        model_paths = [Path(f"GleasonFinal/label_level0/SoftDiceBalanced-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll1/SDB":
        model_paths = [Path(f"GleasonFinal/label_level1/SoftDiceBalanced-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll0/OHCE":
        model_paths = [Path(f"GleasonFinal/label_level0/OH_CE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll1/OHCE":
        model_paths = [Path(f"GleasonFinal/label_level1/OH_CE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll1/SDBML":
        model_paths = [Path(f"GleasonFinal/label_level1/SoftDiceBalancedMultiLevel-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll0/DICE":
        model_paths = [Path(f"GleasonFinal/label_level0/DICE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final/ll1/DICE":
        model_paths = [Path(f"GleasonFinal/label_level1/DICE-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final2/ll1/SDB":
        model_paths = [Path(f"GleasonFinal2/label_level1/SoftDiceBalanced-{i}/version_0/") for i in [1, 2, 3]]

    elif selected_model == "final2/ll1/CE":
        model_paths = [Path(f"GleasonFinal2/label_level1/CE-{i}/version_0/") for i in [1, 2, 3]]

    else:
        raise RuntimeError()

    preds_paths = []
    for path in model_paths:
        assert (base_path / path).exists(), f"Could not find {str(base_path/path)}"
        assert (base_path / path / "preds" / f"pred_{eval_on}.pt").exists(), base_path / path / "preds" / "pred_test.pt"
        preds_paths.append(base_path / path / "preds" / f"pred_{eval_on}.pt")

    data_options = {
        "org": {
            "scaling": "1024",
            "transforms": basic_transforms_val_test_scaling512,
            "label_level": label_level,
            "create_seg_masks": True,
            "tissue_mask_kwargs": {"open": False, "close": False, "flood": False},
        },
        "final": {
            "scaling": "MicronsCalibrated",
            "transforms": normalize_only_transform,
            "label_level": label_level,
            "create_seg_masks": True,
            "tissue_mask_kwargs": {"open": False, "close": False, "flood": False},
            "drawing_order": "grade_frame_order",
            "explanation_file": "final_filtered_explanations_df.csv",
            "data_split": (0.7, 0.15, 0.15),
        },
    }

    # data = GleasonX(base_path, split="test", scaling="1024", transforms=basic_transforms_val_test_scaling512, label_level=label_level, create_seg_masks=True, tissue_mask_kwargs={"open": False, "close":False, "flood":False})
    data = GleasonX(base_path, split="test", **data_options[data_opts])

    labels = []
    bgs = []
    for i in tqdm(range(len(data))):
        _, label, background = data[i]
        labels.append(label)
        bgs.append(background)

    def remapping_function(out):

        out_remappings = generate_label_hierarchy(out, data.exp_numbered_lvl_remapping, start_level=1)

        return out_remappings[0]

    return model_paths, preds_paths, label_level, num_classes, data, labels, bgs, remapping_function


# Helper function that computes all metrics for a list of predictions and labels. The returned metrics are: mDICED, L1, L1Compare, DICEmacro_unique_max, DICE_unique_max, Acc, Bacc, which translates in our paper to
# SoftDiceLoss, L1, nothing, Macro DICE, Dice, Accuracy and Balanced Accuracy
def get_all_metrics_from_list(preds, labels, bgs):

    num_classes = preds[0].shape[0]

    d_mac = Dice(num_classes=num_classes, average="macro")
    d_mic = Dice(num_classes=num_classes, average="micro")
    b_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
    acc = Accuracy(task="multiclass", num_classes=num_classes, average="micro")

    emd = []
    L1 = L1CalibrationMetric()
    mDICED = SoftDICECorrectAccuSemiMetric()

    for i in tqdm(range(len(preds))):
        pred = preds[i]
        bg = bgs[i]
        label = labels[i]
        fg = ~bg

        emd.append(((pred[:, fg] - label[:, fg]).abs().sum(dim=1) / 2) / fg.sum())

        label_max = torch.max(label, dim=0)[0]
        duplicated_max = torch.sum(label == label_max.unsqueeze(0), dim=0) > 1

        unique_max = ~duplicated_max

        unique_max_fg = torch.logical_and(fg, unique_max)

        unique_max_fg_pred_maj = pred[:, unique_max_fg].unsqueeze(0).argmax(dim=1)
        unique_max_fg_label_maj = label[:, unique_max_fg].unsqueeze(0).argmax(dim=1)

        d_mac.update(unique_max_fg_pred_maj, unique_max_fg_label_maj)
        d_mic.update(unique_max_fg_pred_maj, unique_max_fg_label_maj)
        b_acc.update(unique_max_fg_pred_maj, unique_max_fg_label_maj)
        acc.update(unique_max_fg_pred_maj, unique_max_fg_label_maj)

        L1.update(pred[:, fg].unsqueeze(0), label[:, fg].unsqueeze(0))
        mDICED.update(pred[:, fg].unsqueeze(0), label[:, fg].unsqueeze(0))
        # corrected_masked_mIoUD.update(preds_ensemble[i].unsqueeze(0), labels[i].unsqueeze(0), keep_mask=~rel_bg)

    emd = torch.stack(emd).mean(dim=0)
    emd = emd.sum()

    d_mac = d_mac.compute()
    d_mic = d_mic.compute()
    mDICED = mDICED.compute()
    L1 = L1.compute()
    acc = acc.compute()
    b_acc = b_acc.compute()

    return {
        "mDICED": mDICED.item(),
        "L1": emd.item(),
        "L1Compare": L1.item(),
        "DICEmacro_unique_max": d_mac.item(),
        "DICE_unique_max": d_mic.item(),
        "Acc": acc.item(),
        "Bacc": b_acc.item(),
    }


# Helper function that computes all metrics for a list of predictions and labels. The returned metrics are: mDICED, L1, L1Compare, DICEmacro_unique_max, DICE_unique_max, Acc, Bacc, which translates in our paper to


def compute_model_metrics(base_path, preds_path, labels, bgs, remap=False):

    mets = []

    for p_path in preds_path:
        print(f"Loading {p_path.parents[1]}")
        preds = torch.load(base_path / p_path)
        print("Softmax")
        preds = [torch.nn.functional.softmax(img_pred.float().squeeze(0), dim=0) for img_pred in preds]

        if remap:
            print("Remapping to LL0")
            preds = [remap(pred.unsqueeze(0)).squeeze(0) for pred in preds]

        print("Computing metrics")
        mets.append(get_all_metrics_from_list(preds, labels, bgs))

    print("Results")
    met_df = pd.DataFrame(mets).aggregate(["mean", "std"])
    print(met_df)

    return met_df


def get_save_path(model_name, remapped):

    save_name = model_name.replace("/", "_")
    remap_str = "_remaped" if remapped else ""
    save_path = Path(f"./results/metrics_final/{save_name}{remap_str}.csv")
    return save_path


def compute_confusion_matrices_and_frequencies(num_classes, base_path, p_paths, data, labels, bgs):

    STRIP_BACKGROUND = True

    pix_freq = torch.zeros(num_classes, dtype=torch.int)
    max_freq = torch.zeros(num_classes, dtype=torch.int)
    pred_freq = torch.zeros(num_classes, dtype=torch.int)

    one_annotator_pred_freq = torch.zeros(num_classes, dtype=torch.int)

    conf_matrices = [ConfusionMatrix(task="multiclass", num_classes=num_classes) for _ in range(len(p_paths))]

    for i, p_path in enumerate(p_paths):

        count_unique_max = 0.0
        count_foreground = 0.0
        count_pixels = 0.0

        conf_matrix = conf_matrices[i]

        preds = torch.load(base_path / p_path)
        preds = [torch.nn.functional.softmax(img_pred.float().squeeze(0), dim=0) for img_pred in preds]

        for i in tqdm(range(len(data))):

            out = preds[i]

            background_mask = bgs[i]
            mask = labels[i]
            if STRIP_BACKGROUND:
                foreground_mask = ~background_mask.bool()
            else:
                foreground_mask = torch.ones_like(background_mask).bool()

            label_max = torch.max(mask, dim=0)[0]
            duplicated_max = torch.sum(mask == label_max.unsqueeze(0), dim=0) > 1

            unique_max = ~duplicated_max

            unique_max_fg = torch.logical_and(foreground_mask, unique_max)

            count_unique_max += torch.logical_and(foreground_mask, unique_max).sum()
            count_foreground += foreground_mask.sum()
            count_pixels += foreground_mask.numel()

            mask_unique = mask[:, unique_max_fg]
            out_unique = out[:, unique_max_fg]

            mask = mask[:, foreground_mask].flatten(start_dim=1)
            out = out[:, foreground_mask].flatten(start_dim=1)

            pix_freq += torch.sum((mask > 0), dim=(1))
            max_freq += torch.bincount(torch.argmax(mask_unique, dim=0).reshape(-1), minlength=num_classes)
            pred_freq += torch.bincount(torch.argmax(out, dim=0).reshape(-1), minlength=num_classes)
            one_annotator_pred_freq += torch.sum(out >= 0.33, dim=(1))  # torch.bincount(torch.argmax(out,  dim=0).reshape(-1), minlength=num_classes)

            conf_matrix(out_unique.argmax(dim=0), mask_unique.argmax(dim=0))

    return pix_freq, max_freq, pred_freq, one_annotator_pred_freq, conf_matrices


def count_unique_max_and_agreement(data):
    count_unique_max = 0.0
    count_foreground = 0.0
    count_pixels = 0.0
    count_agg_annotators = torch.zeros(4)

    label_counts = torch.zeros(data.num_classes, 4)
    STRIP_BACKGROUND = True

    for i in tqdm(range(len(data))):

        _, mask, background_mask = data[i]

        if STRIP_BACKGROUND:
            foreground_mask = ~background_mask.bool()
        else:
            foreground_mask = torch.ones_like(background_mask).bool()

        mask = (mask * 3).int()
        mask_fg = mask[:, foreground_mask].flatten(start_dim=1)

        count_agg_annotators += torch.bincount(mask_fg.max(dim=0)[0], minlength=4)
        for c in range(data.num_classes):
            occ_count = torch.bincount(mask_fg[c].flatten(), minlength=4)
            label_counts[c, :] += occ_count

        label_max = torch.max(mask, dim=0)[0]
        duplicated_max = torch.sum(mask == label_max.unsqueeze(0), dim=0) > 1

        unique_max = ~duplicated_max

        count_unique_max += torch.logical_and(foreground_mask, unique_max).sum()
        count_foreground += foreground_mask.sum()
        count_pixels += foreground_mask.numel()
    return count_unique_max, count_foreground, label_counts
