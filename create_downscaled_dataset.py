import math
import os
from pathlib import Path
from typing import Union

import albumentations as alb
import numpy as np
from PIL import Image
from tqdm import tqdm

import src.gleason_data as gleason_data
from src.gleason_utils import create_segmentation_masks


def save_downscaled_TMAs(path_to_tmas: Union[str, Path],
                         new_path_to_tmas: Union[str, Path],
                         shorter_edge_length: int,
                         file_format: str):

    downscale_transform = alb.SmallestMaxSize(max_size=shorter_edge_length, interpolation=2)

    path_to_tmas = Path(path_to_tmas)
    new_path_to_tmas = Path(new_path_to_tmas)

    assert path_to_tmas.exists()
    assert not (new_path_to_tmas.exists() and len(list(new_path_to_tmas.glob("*"))) > 0)
    new_path_to_tmas.mkdir(parents=True, exist_ok=True)

    tma_paths = gleason_data.load_tmas(path_to_tmas)

    if not file_format[0] == ".":
        file_format = "."+file_format

    for tma_identifier, tma_path in tqdm(tma_paths.items()):

        tma_path = Path(path_to_tmas/tma_path)
        new_path = Path(new_path_to_tmas/(tma_identifier+file_format))

        img = Image.open(tma_path)
        img_np = np.array(img)

        img_alb = downscale_transform(image=img_np)["image"]

        img_pil = Image.fromarray(img_alb)

        img_pil.save(str(new_path))


def save_downscaled_TMAs_microns_based(path_to_tmas: Union[str, Path],
                                       new_path_to_tmas: Union[str, Path],
                                       desired_microns_per_pixel: float,
                                       file_format: str):

    dataset_micron_mapping = {"Gleason19": 0.25, "Harvard": 0.23, "TMA": 0.5455}

    path_to_tmas = Path(path_to_tmas)
    new_path_to_tmas = Path(new_path_to_tmas)

    assert path_to_tmas.exists()
    assert not (new_path_to_tmas.exists() and len(list(new_path_to_tmas.glob("*"))) > 0)
    new_path_to_tmas.mkdir(parents=True, exist_ok=True)

    tma_paths = gleason_data.load_tmas(path_to_tmas)

    if not file_format[0] == ".":
        file_format = "."+file_format

    for tma_identifier, tma_path in tqdm(tma_paths.items()):

        tma_path = Path(path_to_tmas/tma_path)
        new_path = Path(new_path_to_tmas/(tma_identifier+file_format))

        dataset = "Gleason19" if tma_identifier.startswith("slide") else "TMA" if tma_identifier.startswith(
            "PR") else "Harvard" if tma_identifier.startswith("ZT") else None

        microns_image = dataset_micron_mapping[dataset]

        downscale_factor = microns_image/desired_microns_per_pixel

        img = Image.open(tma_path)
        size = img.size

        new_size = math.ceil(min(size[0]*downscale_factor, size[1]*downscale_factor))
        downscale_transform = alb.SmallestMaxSize(max_size=new_size, interpolation=2)

        img_np = np.array(img)
        img_alb = downscale_transform(image=img_np)["image"]

        img_pil = Image.fromarray(img_alb)

        img_pil.save(str(new_path))


def save_segmentation_masks(data, save_path, save_path_background, indices, shorter_edge_length=None, mask_kwargs=None):

    if indices is None:
        indices = range(len(data))

    assert not (save_path.exists() and len(list(save_path.glob("*"))) > 0)
    save_path.mkdir(parents=True, exist_ok=True)

    # assert not (save_path_background.exists() and len(list(save_path_background.glob("*"))) > 0)
    save_path_background.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(indices):

        _, seg_images, background_mask = create_segmentation_masks(data, idx, shorter_edge_length,  tissue_mask_kwargs=mask_kwargs)

        slide_name = data.used_slides[idx]
        palette_array = data.color_palette

        seg_mask_save_path = Path(save_path)/slide_name
        seg_mask_save_path.mkdir(exist_ok=True)

        for anno, seg_mask in seg_images.items():
            anno_save_path = seg_mask_save_path/str(anno+".png")

            pil_img = Image.fromarray(seg_mask.astype(np.uint8), mode='P')
            pil_img.putpalette(palette_array)
            pil_img.save(anno_save_path)

        background_img_save_path = save_path_background/f"{slide_name}.png"
        if not background_img_save_path.exists():
            pil_img = Image.fromarray(background_mask.astype(np.uint8)*255, mode='L').convert("1")
            pil_img.save(background_img_save_path)


# Create a downscaled dataset
def main():
    path = Path(os.environ["DATASET_LOCATION"])/"GleasonXAI"

    TARGET_SPACING = 1.39258  # microns/pixel
    LABEL_LEVELS_TO_CREATE = [0, 1, 2]
    tissue_mask_kwargs = {"open": False, "close": False, "flood": False}
    SAVE_TMAs = True
    SAVE_MASKS = False  # No need to save masks can be created on the fly.

    save_res_name = "MicronsCalibrated"

    if SAVE_TMAs:
        load_path_TMAS = path/"TMA"/"original"
        save_path_tmas = path/"TMA"/save_res_name

        save_downscaled_TMAs_microns_based(load_path_TMAS, save_path_tmas, TARGET_SPACING, ".jpg")

    if SAVE_MASKS:

        for label_level in LABEL_LEVELS_TO_CREATE:

            save_path_seg = path/"segmentation_masks"/("label_level_"+str(label_level))/save_res_name
            save_path_background = path/"background_masks"/save_res_name
            if tissue_mask_kwargs is None or tissue_mask_kwargs == {}:
                save_path_background /= "default"

            else:
                save_path_background /= str(tissue_mask_kwargs)

            data = gleason_data.GleasonX(path=path, split="all", scaling="MicronsCalibrated", transforms=None,
                                         label_level=label_level, create_seg_masks=True, explanation_file="final_filtered_explanations_df.csv", data_split=[0.7, 0.15, 0.15], tissue_mask_kwargs={"open": False, "close": False, "flood": False})

            save_segmentation_masks(data=data, save_path=save_path_seg, save_path_background=save_path_background,
                                    indices=None, shorter_edge_length=None, mask_kwargs=tissue_mask_kwargs)

if __name__ == "__main__":
    main()
