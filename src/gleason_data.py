# %%
import json
import os
import shutil
import warnings
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Literal, Optional, Union

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tt
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
from PIL import Image
from torch.utils.data import random_split
from tqdm import tqdm

from .gleason_utils import create_segmentation_masks
from .tree_loss import get_explanation_level_mapping, parse_label_hierarchy

# Necessary because somthing changed in Pandas again...
warnings.simplefilter(action="ignore", category=FutureWarning)


def reformat_dataset_to_flat_structure(path_to_tmas: Union[str, Path], new_path_to_tmas: Union[str, Path], file_format: str):

    path_to_tmas = Path(path_to_tmas)
    new_path_to_tmas = Path(new_path_to_tmas)

    assert path_to_tmas.exists()
    new_path_to_tmas.mkdir(parents=True, exist_ok=True)

    tma_paths = load_tmas(path_to_tmas)

    if not file_format[0] == ".":
        file_format = "." + file_format

    for tma_identifier, tma_path in tqdm(tma_paths.items()):

        tma_path = Path(path_to_tmas / tma_path)
        new_path = Path(new_path_to_tmas / (tma_identifier + file_format))

        if tma_path.suffix == ".png":
            img = Image.open(tma_path)
            img.save(new_path)
        elif tma_path.suffix == ".jpg":
            shutil.copy(tma_path, new_path)
        else:
            raise RuntimeError(tma_path.suffix)


def prepare_torch_inputs(img, seg_masks, num_classes):

    img = tt.functional.to_tensor(img)
    _, H, W = img.shape

    label = torch.zeros((H, W, num_classes), dtype=torch.int64)

    for seg_mask in seg_masks:

        seg_mask = torch.tensor(seg_mask, dtype=torch.int64)
        one_hot = torch.zeros((H, W, num_classes), dtype=torch.int64)
        one_hot.scatter_(2, seg_mask.unsqueeze(2), 1)
        label += one_hot

    label = label / len(seg_masks)

    label = label.permute([2, 0, 1])

    return img, label


def get_class_colormaps(num_classes_per_grade, min=0.1, max=0.9):

    def generate_colormap(
        cm_name,
        num,
        min=0,
        max=1,
    ):
        colors = cm[cm_name](np.linspace(min, max, num + 2))[1:-1]
        colormap = ListedColormap(colors)
        return colormap

    greens = generate_colormap("Greens", num_classes_per_grade["3"], min=min, max=max)
    blues = generate_colormap("Blues", num_classes_per_grade["4"], min=min, max=max)
    reds = generate_colormap("Reds", num_classes_per_grade["5"], min=min, max=max)

    colormap = ListedColormap(np.concatenate([np.array([[1.0, 1.0, 1.0, 1.0]]), greens.colors, blues.colors, reds.colors]))

    return colormap


def load_explanations(path, explanation_file="explanations_df.csv"):

    df = pd.read_csv(path / explanation_file)
    df = df.dropna(how="all")
    df["explanations"] = df["explanations"].astype("category")

    return df


def postprocess_df(df, tma_paths, exp_lvl_remapping, label_level):

    def match_to_tma(tma_identifier):
        return tma_identifier in tma_paths.keys()

    # df[["TMA_identifier", "TMA_path"]] = df["TMA"].apply(
    #     match_to_tma).apply(pd.Series)
    assert df["TMA_identifier"].apply(match_to_tma).all(), "Some files could not be matched"

    # Convert the coordinates of the polygons
    def str_list_convert_pd(x):
        try:
            return np.array(json.loads(x))
        except:
            return np.nan

    converted_coords = df["coords"].apply(str_list_convert_pd)
    df["coords"] = converted_coords

    assert df["coords"].isnull().sum() == 0, "Some coordinates are null."

    # Remap explanations with label hierarchy

    # COLUMN_NAMES = ("explanation_grade", "explanations_grouped", "explanations_base")
    COLUMN_NAMES = [f"explanation_lvl_{i}" for i in range(len(exp_lvl_remapping) + 1)]

    # Go from lowest to highest
    COLUMN_NAMES = COLUMN_NAMES[::-1]
    exp_lvl_remapping = exp_lvl_remapping[::-1]

    df[COLUMN_NAMES[0]] = df["explanations"]

    for (column_name_low, column_name_high), remapping in zip(pairwise(COLUMN_NAMES), exp_lvl_remapping):

        reversed_remapping = {}
        for high_exp, low_exps in remapping:
            for low_exp in low_exps:
                reversed_remapping[low_exp] = high_exp

        df[column_name_high] = df[column_name_low].replace(reversed_remapping).astype("category")

    # Leads to problems due to different datatypes between df and the label_hierarchy json file (which is always loaded as string)
    # df["explanation_grade"] = df["explanation_grade"].astype(int)

    # Use "explanations" to store the exps for the desired label_level
    df["explanations"] = df[COLUMN_NAMES[::-1][label_level]]

    return df


def load_tmas(base_dataset_path, flat_structure=None):
    """Loads the TMAs provided a $base_dataset_path. Uses the $flat_structure layout if provided.

    Args:
        base_dataset_path (Path): Path to the TMA folder.
        flat_structure (bool, optional): Use the flat_structure layout. If provided as None, it will be figured out. Defaults to None.

    Raises:
        RuntimeError: If the folder layout does not fit.

    Returns:
        Dict[str, Path]: TMA identifiers mapped to relativ paths.
    """

    if flat_structure is None:

        folders_exist = [(base_dataset_path / folder_name).exists() for folder_name in ["dataverse", "Gleason2019", "tissuemicroarray"]]

        if all(folders_exist):
            flat_structure = False

        elif not any(folders_exist):
            flat_structure = True
        else:
            raise RuntimeError("wtf")

    if flat_structure:

        TMA_files = base_dataset_path.glob("*.jpg")
        return {f.stem: f.relative_to(base_dataset_path) for f in TMA_files}

    else:

        gleason_2019_paths = [
            base_dataset_path / "Gleason2019/Train Imgs",
            base_dataset_path / "Gleason2019/Test_imgs",
        ]

        tissuemicroarray_paths = [
            base_dataset_path / "tissuemicroarray",
        ]

        dataverse_paths = [
            base_dataset_path / "dataverse/dataverse_files",
        ]

        dataset_paths = {"Gleason2019": gleason_2019_paths, "dataverse": dataverse_paths, "tissuemicroarray": tissuemicroarray_paths}

        TMA_files = {}  # {"Gleason2019":{}, "dataverse":{}, "tissuemicroarray":{}}
        encountered_file_names = defaultdict(list)

        valid_extensions = {"Gleason2019": (".jpg"), "dataverse": ".jpg", "tissuemicroarray": ".png"}

        for dataset in ["Gleason2019", "dataverse", "tissuemicroarray"]:
            for dataset_path in dataset_paths[dataset]:

                if dataset_path.exists() and dataset_path.is_dir():
                    for ext in valid_extensions[dataset]:
                        found_files = dataset_path.rglob(f"*{ext}")

                        for found_file in found_files:

                            if dataset == "Gleason2019":
                                file_name = str(found_file.relative_to(dataset_path)).replace("/", "_").split(".")[0]

                            elif dataset == "tissuemicroarray":

                                file_name = str(found_file.relative_to(dataset_path)).replace("/", "_").split(".")[0]
                                rename_mapping = {
                                    "HE_PR482a-073": "PR482a",
                                    "HE_PR633a-019": "PR633a",
                                    "HE_PR1001-SK112": "PR1001",
                                    "HE_PR1921b-SD089": "PR1921b",
                                }

                                for key, item in rename_mapping.items():
                                    file_name = file_name.replace(key, item)

                            elif dataset == "dataverse":

                                if "Gleason_masks" in str(found_file):
                                    continue

                                file_name = str(found_file.stem)

                            duplicate = file_name in encountered_file_names

                            encountered_file_names[file_name].append(found_file)

                            if duplicate:
                                print(f"Found duplicate of {file_name}. Locations: {encountered_file_names[file_name]}")
                            else:
                                TMA_files[file_name] = found_file.relative_to(base_dataset_path)

    return TMA_files


def load_segmentations(path_seg, path_background):

    image_paths = {}

    for folder in Path(path_seg).iterdir():

        if folder.is_dir():
            image_paths[folder.name] = [img.relative_to(path_seg) for img in folder.glob("*.png")]

    background_images = [img.relative_to(path_background) for img in path_background.glob("*.png")]

    background_paths = {img.stem: img for img in background_images}

    return image_paths, background_paths


def show_TMA(img_path):

    img = np.array(Image.open(img_path))
    return Image.fromarray(img)


def draw_on_TMA(img, explanations, coords, color_exp_mapping):

    shorter_edge_length = 512

    # img = Image.open(base_dataset_path/img_path)

    shorter_edge = min(img.size)
    scale_factor = shorter_edge / shorter_edge_length
    new_img_size = np.int32(np.array(img.size) / scale_factor)

    img = img.resize(new_img_size)
    img = np.array(img)

    for coord, explanation in zip(coords, explanations):

        exp_color = color_exp_mapping[explanation]

        coord = np.int32(coord.T * np.array(img.shape[:2]).reshape(-1, 1))
        img = cv2.polylines(img, [coord.T], True, exp_color, 2)

    return Image.fromarray(img)


class GleasonX(torch.utils.data.Dataset):

    BACKGROUND_VALUE = 0

    def __init__(
        self,
        path: Union[str, Path],
        split: Literal["train", "val", "test", "all"],
        scaling: Union[str, int] = "original",
        transforms: Optional[albumentations.BasicTransform] = None,
        label_level: Literal[0, 1, 2] = 1,
        create_seg_masks=True,
        tissue_mask_kwargs={},
        explanation_file="explanations_df.csv",
        label_hierarchy_file="label_remapping.json",
        drawing_order="grade_frame_order",
        data_split=(0.7, 0.15, 0.15),
    ):

        path = Path(path)
        self.path = path
        assert self.path.exists()

        scaling = str(scaling)
        self.tma_base_path = self.path / "TMA" / scaling
        self.segmentation_masks_base_path = self.path / "segmentation_masks" / ("label_level_" + str(label_level)) / scaling
        self.background_mask_base_path = self.path / "background_masks" / scaling

        if tissue_mask_kwargs is None or tissue_mask_kwargs == {}:
            self.background_mask_base_path /= "default"
        else:
            self.background_mask_base_path /= str(tissue_mask_kwargs)

        if scaling == "MicronsCalibrated" and not drawing_order in ["custom_order", "grade_frame_order"]:
            raise RuntimeError("Use grade_frame_order instead")

        assert (self.tma_base_path).exists(), f"{self.tma_base_path} should contain the TMAs but does not exist!"

        if not create_seg_masks:
            assert self.segmentation_masks_base_path.exists(), f"{self.segmentation_masks_base_path} should contain segmentation masks but does not exist"
            assert self.background_mask_base_path.exists(), f"{self.background_mask_base_path} should contain background masks but does not exist"
            self.CREATE_SEG_MASKS = False
        else:
            self.CREATE_SEG_MASKS = True

        try:
            f = open(self.path / label_hierarchy_file, "r")
        except IOError:
            print(f"label hierarchy file {self.path/label_hierarchy_file} does not exist.")
        with open(self.path / label_hierarchy_file, "r") as f:
            label_mapping = json.load(f)["hierarchy"]

        self.exp_per_level, self.exp_per_level_numbered, self.exp_lvl_remapping, self.exp_numbered_lvl_remapping = parse_label_hierarchy(label_mapping)

        # Find all TMAs from all 3 datasets. Remove file suffices and grade_indicators.
        tma_paths = load_tmas(self.tma_base_path)
        self.tma_paths = tma_paths

        # Load all segmentations
        if not self.CREATE_SEG_MASKS:

            seg_paths, background_paths = load_segmentations(self.segmentation_masks_base_path, self.background_mask_base_path)

            # Not all TMAs have a segmentation.
            assert set(tma_paths.keys()).issuperset(seg_paths.keys())
            # assert len(set(tma_paths.keys()).difference(seg_paths.keys())) == 570

            self.seg_paths = seg_paths
            self.background_paths = background_paths

        df = load_explanations(path, explanation_file=explanation_file)
        df = postprocess_df(df, tma_paths, self.exp_lvl_remapping, label_level)

        self.transforms = transforms
        self.drawing_order = drawing_order

        assert split in ["train", "val", "test", "all"]
        self.split = split

        unique_slides = sorted(df["TMA_identifier"].unique())
        # if not self.CREATE_SEG_MASKS:
        #    assert len(unique_slides) == len(seg_paths.keys())

        train, val, test = random_split(unique_slides, data_split, torch.Generator().manual_seed(95967))

        self.train_slides = train
        self.val_slides = val
        self.test_slides = test

        if self.split == "train":
            self.used_slides = list(train)
        elif self.split == "val":
            self.used_slides = list(val)
        elif self.split == "test":
            self.used_slides = list(test)
        elif self.split == "all":
            self.used_slides = list(unique_slides)

        grade_label_level_mapping = get_explanation_level_mapping(higher_level=0, lower_level=label_level, label_hierarchy=self.exp_numbered_lvl_remapping)

        # self.exp_grade_mapping = {exp: int(grade[0]) for exp, grade in df.groupby("explanations", observed=False)["explanation_grade"].unique().items()}
        self.explanations = self.exp_per_level[label_level]
        self.exp_number_mapping = self.exp_per_level_numbered[label_level]

        self.num_classes = len(self.explanations) + 1
        self.classes_named = ["Benign"] + self.explanations
        self.classes_number_mapping = {"Benign": 0} | self.exp_number_mapping
        self.scaling = scaling
        self.label_level = label_level

        self.df = df
        self.df_tma_groups = df.groupby(by="TMA_identifier")

        self.tissue_mask_kwargs = tissue_mask_kwargs

        # dict(df.groupby("explanation_grade", observed=False)["explanations"].nunique())
        num_classes_per_grade = {k: len(v) for k, v in grade_label_level_mapping.items()}
        num_classes_per_grade = {"3": num_classes_per_grade[1], "4": num_classes_per_grade[2], "5": num_classes_per_grade[3]}

        if label_level == 0:
            self.exp_grade_mapping = {"3": 3, "4": 4, "5": 5}
        else:
            self.exp_grade_mapping = {}

            for idx, grade in enumerate([3, 4, 5], start=1):
                rel_labels = grade_label_level_mapping[idx]

                for r_l in rel_labels:
                    self.exp_grade_mapping[self.explanations[r_l - 1]] = grade

        self.colormap = get_class_colormaps(num_classes_per_grade=num_classes_per_grade)
        self.color_palette = (self.colormap.colors[:, :3] * 255).astype(np.uint8)

    def __getitem__(self, idx, prepare_torch=True, drawing_order=None):

        if drawing_order is None:
            drawing_order = self.drawing_order

        # Load image
        img_name = self.used_slides[idx]

        img_path = self.tma_paths[img_name]
        img = Image.open(self.tma_base_path / img_path)
        img = np.array(img)

        if not self.CREATE_SEG_MASKS:
            seg_paths = self.seg_paths[img_name]
            seg_images = [Image.open(self.segmentation_masks_base_path / seg_path) for seg_path in seg_paths]
            seg_images = [np.array(seg_img) for seg_img in seg_images]

            background_mask = np.array(Image.open(self.background_mask_base_path / self.background_paths[img_name]))

        else:
            # NOTE not sure if this is absolutly correct with the way CV and PIL handle resizing... but will not be used for training anyway.
            desired_shorter_edge_length = None
            if self.scaling.isdigit():
                desired_shorter_edge_length = int(self.scaling)
            elif self.scaling == "original":
                desired_shorter_edge_length = None
            elif self.scaling == "MicronsCalibrated":
                desired_shorter_edge_length = min(img.shape[:2])
            else:
                RuntimeError()

            _, seg_images, background_mask = create_segmentation_masks(
                self, idx, desired_shorter_edge_length, tissue_mask_kwargs=self.tissue_mask_kwargs, drawing_order=drawing_order
            )

            # SegImages are dict(AnnotatorName:str, seg_img:np.array)
            seg_images = list(seg_images.values())

        if self.transforms is not None:
            max_trials = 100
            c = 0

            while c < max_trials:
                album_dict = self.transforms(image=img, masks=seg_images + [background_mask.astype(np.uint8)])
                new_img = album_dict["image"]
                *new_seg_images, new_background_mask = album_dict["masks"]
                new_background_mask = new_background_mask.astype(bool)
                if new_background_mask.sum() - new_background_mask.size != 0:
                    break
                else:
                    c += 1
                    if c == max_trials:
                        raise RuntimeError("Extracted background only {max_trials} times")
                    # Probably counts one to few. Of by 1 error, but who cares.

            background_mask = new_background_mask
            seg_images = new_seg_images
            img = new_img

        if prepare_torch:
            img, mask = prepare_torch_inputs(img, seg_images, self.num_classes)
            return img, mask, torch.tensor(background_mask).bool()

        return img, seg_images, background_mask

    def get(self, idx, prepare_torch):
        return self.__getitem__(idx, prepare_torch)

    def get_raw_image(self, id):

        img_name = self.used_slides[id]
        img_path = self.tma_paths[img_name]
        img = Image.open(self.tma_base_path / img_path)

        return img

    def get_slide_df(self, id):

        rel_df = self.df_tma_groups.get_group(self.used_slides[id])

        return rel_df

    def __len__(self):

        return len(self.used_slides)


class GleasonXClassification(GleasonX):

    def __getitem__(self, idx):

        img, masks, background_mask = super().__getitem__(idx, prepare_torch=False)

        num_annotators = len(masks)

        # Transform into classification

        annotator_annotated_class = np.stack([np.bincount(mask.flatten(), minlength=self.num_classes) for mask in masks], axis=1) > 0

        # Soft Label
        label = annotator_annotated_class.mean(axis=1)

        # Hard Label
        # label = annotator_annotated_class.mean(axis=0) >= 0.5

        # Return None for background_mask
        return tt.functional.to_tensor(img), torch.tensor(label), tt.functional.to_tensor(background_mask)


if __name__ == "__main__":
    data = GleasonX(Path(os.environ["DATASET_LOCATION"] / "GleasonXAI"), "all", transforms=None, label_level=1)

# %%
