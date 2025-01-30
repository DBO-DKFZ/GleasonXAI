import math
from pathlib import Path
from typing import Union

import albumentations as alb
import numpy as np
from PIL import Image
from tqdm import tqdm

import src.gleason_data as gleason_data


def save_downscaled_TMAs(path_to_tmas: Union[str, Path], new_path_to_tmas: Union[str, Path], shorter_edge_length: int, file_format: str):

    downscale_transform = alb.SmallestMaxSize(max_size=shorter_edge_length, interpolation=2)

    path_to_tmas = Path(path_to_tmas)
    new_path_to_tmas = Path(new_path_to_tmas)

    assert path_to_tmas.exists()
    assert not (new_path_to_tmas.exists() and len(list(new_path_to_tmas.glob("*"))) > 0)
    new_path_to_tmas.mkdir(parents=True, exist_ok=True)

    tma_paths = gleason_data.load_tmas(path_to_tmas)

    if not file_format[0] == ".":
        file_format = "." + file_format

    for tma_identifier, tma_path in tqdm(tma_paths.items()):

        tma_path = Path(path_to_tmas / tma_path)
        new_path = Path(new_path_to_tmas / (tma_identifier + file_format))

        img = Image.open(tma_path)
        img_np = np.array(img)

        img_alb = downscale_transform(image=img_np)["image"]

        img_pil = Image.fromarray(img_alb)

        img_pil.save(str(new_path))


def save_downscaled_TMAs_microns_based(path_to_tmas: Union[str, Path], new_path_to_tmas: Union[str, Path], desired_microns_per_pixel: float, file_format: str):

    dataset_micron_mapping = {"Gleason19": 0.25, "Harvard": 0.23, "TMA": 0.5455}

    path_to_tmas = Path(path_to_tmas)
    new_path_to_tmas = Path(new_path_to_tmas)

    assert path_to_tmas.exists()
    assert not (new_path_to_tmas.exists() and len(list(new_path_to_tmas.glob("*"))) > 0)
    new_path_to_tmas.mkdir(parents=True, exist_ok=True)

    tma_paths = gleason_data.load_tmas(path_to_tmas)

    if not file_format[0] == ".":
        file_format = "." + file_format

    for tma_identifier, tma_path in tqdm(tma_paths.items()):

        tma_path = Path(path_to_tmas / tma_path)
        new_path = Path(new_path_to_tmas / (tma_identifier + file_format))

        dataset = (
            "Gleason19"
            if tma_identifier.startswith("slide")
            else "TMA" if tma_identifier.startswith("PR") else "Harvard" if tma_identifier.startswith("ZT") else None
        )

        microns_image = dataset_micron_mapping[dataset]

        downscale_factor = microns_image / desired_microns_per_pixel

        img = Image.open(tma_path)
        size = img.size

        new_size = math.ceil(min(size[0] * downscale_factor, size[1] * downscale_factor))
        downscale_transform = alb.SmallestMaxSize(max_size=new_size, interpolation=2)

        img_np = np.array(img)
        img_alb = downscale_transform(image=img_np)["image"]

        img_pil = Image.fromarray(img_alb)

        img_pil.save(str(new_path))
