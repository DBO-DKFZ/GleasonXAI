import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as tt
from matplotlib.colors import ListedColormap
from monai.inferers import SlidingWindowInferer
from PIL import Image

from src.augmentations import normalize_only_transform
from src.gleason_utils import tissue_filter_image
from src.lightning_modul import LitSegmenter

CLASS_NUMBER_MAPPING = {
    "Benign": 0,
    "variable sized well-formed individual and discrete glands": 1,
    "compressed or angular discrete glands": 2,
    "poorly formed and fused glands": 3,
    "Cribriform glands": 4,
    "Glomeruloid glands": 5,
    "solid groups of tumor cells": 6,
    "single cells": 7,
    "cords": 8,
    "presence of comedonecrosis": 9,
}

DATA_COLOR_MAP = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.64364475, 0.85614764, 0.61979239, 1.0],
        [0.24521338, 0.66025375, 0.35695502, 1.0],
        [0.71618608, 0.83320261, 0.91615532, 1.0],
        [0.41708574, 0.68063053, 0.83823145, 1.0],
        [0.17914648, 0.49287197, 0.73542484, 1.0],
        [0.98823529, 0.72198385, 0.61836217, 1.0],
        [0.98683583, 0.51657055, 0.39108035, 1.0],
        [0.95404844, 0.29714725, 0.21453287, 1.0],
        [0.80604383, 0.10380623, 0.11787774, 1.0],
    ]
)

NAMED_CLASSES = [
    "Benign",
    "variable sized well-formed individual and discrete glands",
    "compressed or angular discrete glands",
    "poorly formed and fused glands",
    "Cribriform glands",
    "Glomeruloid glands",
    "solid groups of tumor cells",
    "single cells",
    "cords",
    "presence of comedonecrosis",
]


def generate_model_output(model, img, device="cpu", label_remapping=None, inferer=None, transform=normalize_only_transform):
    model.eval()

    img = transform(image=img)["image"]

    if not isinstance(img, torch.Tensor):
        img = tt.functional.to_tensor(img)

    if len(img.size()) == 3:
        no_batch_input = True
        img = img.unsqueeze(0)
    else:
        no_batch_input = False

    img = img.to(device)

    with torch.no_grad():
        if inferer is not None:
            out = inferer(img, model)
        else:
            out = model(img)

    # Move back and strip batch_dim
    out = out.cpu()

    if label_remapping is not None:
        out = label_remapping(out)

    if no_batch_input:
        out = out[0, ...]

    return out


def generate_image(image_path, models, device, save_path):
    tissue_mask_kwargs = {"open": False, "close": False, "flood": False}
    inferer = SlidingWindowInferer(roi_size=(512, 512), sw_batch_size=1, overlap=0.5, mode="gaussian")

    mask_colormap = ListedColormap(np.concatenate([np.array([[0.0, 0.0, 0.0, 1.0]]), DATA_COLOR_MAP]))
    no_mask_colormap = ListedColormap(DATA_COLOR_MAP)
    mask_num_class_to_vis = 11

    try:
        example_img = Image.open(image_path)
    except IOError:
        print(f"image {image_path} can't be opened and is skipped.")
        return

    example_img = np.array(example_img)

    background_mask = ~tissue_filter_image(example_img, size=np.array(example_img.shape[:2])[::-1], **tissue_mask_kwargs)

    print("- generate predictions for:", image_path.name)
    preds_ensemble = generate_model_output(models[0], example_img, device, None, inferer, normalize_only_transform)
    for model in models[1:]:
        out = generate_model_output(model, example_img, device, None, inferer, normalize_only_transform)
        preds_ensemble += out
    preds_ensemble = torch.nn.functional.softmax(preds_ensemble, dim=0)
    np_seg = np.array(preds_ensemble.argmax(dim=0)).astype(np.uint8)

    mask_np_seg = np_seg + 1
    mask_np_seg[background_mask.astype(bool)] = 0


    plt.figure(figsize=(100, 200))
    fig, (ax0, ax1) = plt.subplots(1, 2)

    ax0.imshow(example_img)
    ax0.set_title("Image")
    ax0.axis("off")
    ax1.imshow(mask_np_seg, cmap=mask_colormap, interpolation_stage="rgba", vmax=mask_num_class_to_vis)
    ax1.set_title("Prediction")
    ax1.axis("off")

    encountered_classes = set(np.unique(np_seg))
    legend_handels = [mpatches.Patch(color=np.array([0.0, 0.0, 0.0, 1.0]), label=f"Background")]
    legend_handels += [
        mpatches.Patch(color=no_mask_colormap(CLASS_NUMBER_MAPPING[cls]), label=cls if len(cls) < 60 else cls[:60] + "...")
        for cls in NAMED_CLASSES
        if CLASS_NUMBER_MAPPING[cls] in encountered_classes
    ]
    fig.legend(handles=legend_handels, loc="outside right", fontsize=6, bbox_to_anchor=(1.4, 0.5))

    plt.savefig(save_path / f"pred_{image_path.stem}.png", dpi=300, bbox_inches="tight")
    print(f"- created prediction at: {save_path / f'pred_{image_path.stem}.png'}")


def generate_predictions(input_path, checkpoints, save_path):
    device = "cpu"

    models = []
    for model_path in checkpoints:
        models.append(LitSegmenter.load_from_checkpoint(str(model_path), map_location=device))

    input_path = Path(input_path)
    save_path = Path(save_path)

    if input_path.is_file():
        generate_image(input_path, models, device, save_path)
    else:
        for file in input_path.iterdir():
            if file.is_file() and str(file).lower().endswith((".png", ".jpg", ".jpeg")):
                generate_image(input_path, models, device, save_path)


if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument("--images")
    parse.add_argument("--checkpoint_absolute", action="store_true")
    parse.add_argument("--checkpoint_1", default="GleasonFinal2/label_level1/SoftDiceBalanced-1/version_0/checkpoints/best_model.ckpt")
    parse.add_argument("--checkpoint_2", default="GleasonFinal2/label_level1/SoftDiceBalanced-2/version_0/checkpoints/best_model.ckpt")
    parse.add_argument("--checkpoint_3", default="GleasonFinal2/label_level1/SoftDiceBalanced-3/version_0/checkpoints/best_model.ckpt")
    parse.add_argument("--save_path")

    args = parse.parse_args()

    assert args.images is not None, "Please specify an image location"
    assert Path(args.images).exists(), f"File / directory {args.images} does not exist"
    assert args.save_path is not None, "Please specify an output path"
    assert Path(args.save_path).exists(), f"File / directory {args.save_path} does not exist"

    checkpoints = []
    if not args.checkpoint_absolute:
        assert "DATASET_LOCATION" in os.environ, f"Environment variable DATASET_LOCATION needed for relative checkpoint but not set!"
        base_path = Path(os.environ["DATASET_LOCATION"]) / "GleasonXAI"
        assert base_path.exists(), f"DATASET_LOCATION {base_path.resolve} does not exist!"

        checkpoint_1 = base_path / Path(args.checkpoint_1)
        assert checkpoint_1.exists(), f"Checkpoint path {checkpoint_1} does not exist!"
        checkpoint_2 = base_path / Path(args.checkpoint_2)
        assert checkpoint_2.exists(), f"Checkpoint path {checkpoint_2} does not exist!"
        checkpoint_3 = base_path / Path(args.checkpoint_3)
        assert checkpoint_3.exists(), f"Checkpoint path {checkpoint_3} does not exist!"
        checkpoints = [checkpoint_1, checkpoint_2, checkpoint_3]

    else:
        checkpoint_1 = Path(args.checkpoint_1)
        assert checkpoint_1.exists(), f"Checkpoint path {checkpoint_1} does not exist!"
        checkpoint_2 = Path(args.checkpoint_2)
        assert checkpoint_2.exists(), f"Checkpoint path {checkpoint_2} does not exist!"
        checkpoint_3 = Path(args.checkpoint_3)
        assert checkpoint_3.exists(), f"Checkpoint path {checkpoint_3} does not exist!"
        checkpoints = [checkpoint_1, checkpoint_2, checkpoint_3]

    generate_predictions(args.images, checkpoints, args.save_path)
