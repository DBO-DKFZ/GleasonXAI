#%%
import math
import shutil
from pathlib import Path
from textwrap import wrap
from typing import Union

import albumentations as alb
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm

classes_ll1_shortform = ["benign tissue", "individual glands", "compressed glands", "poorly formed glands",
                         "cribriform glands", "glomeruloid glands", "group of tumor cells", "single cells", "cords", "comedenocrosis"]

classes_ll0_shortform = ["Gleason pattern 3", "Gleason pattern 4", "Gleason pattern 5"]


def tissue_filter_image(img, kernel_size=17, iterations=2, close=True, open=True, flood=True, size=None):

    # apply median filter to remove artifacts created by transitions to background pixels
    median_filtered_img = cv2.medianBlur(img, 11)

    # convert to HSV color space
    hsv_image = cv2.cvtColor(median_filtered_img, cv2.COLOR_RGB2HSV)

    # get saturation channel
    saturation = hsv_image[:, :, 1]

    # Otsu's thresholding
    _, tissue_mask = cv2.threshold(saturation, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply dilation to image to close spots inside mask regions
    kernel = np.ones(shape=(kernel_size, kernel_size))

    if close:
        # Closing: Dialation -> Erosion.   Connects components. Used here to connect small insulas directly infront of tissue to tissue region.
        tissue_mask = cv2.morphologyEx(tissue_mask, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=iterations)

    if flood:
        # tissue_mask = tissue_mask.astype(bool)

        # 0 is background, 1 is tissue, 2 is floodfilled background
        h, w = tissue_mask.shape
        mask = np.zeros((h+2, w+2), np.uint8)
        mask[1:-1, 1:-1] = tissue_mask
        _, ff_old, ff, _ = cv2.floodFill(mask, None, (0, 0), 2)

        # The third argument is the floodingmask padded by all sides with 1. So we need to extract 1 more.
        tissue_mask = np.logical_not(ff[2:-2, 2:-2]).astype(np.uint8)

        # Old implementation. Keep assert for compare.
        tm_old = ff_old[1:-1, 1:-1]
        tm_old[tm_old == 0] = 1
        tm_old[tm_old == 2] = 0
        assert (tm_old == tissue_mask).all()

    if open:
        # Opening: Erosion -> Dialation. Removes small components while preserving the rest.
        tissue_mask = cv2.morphologyEx(tissue_mask, op=cv2.MORPH_OPEN, kernel=kernel, iterations=iterations)

    if size is not None:

        tissue_mask = cv2.resize(tissue_mask, size, interpolation=cv2.INTER_NEAREST_EXACT)

    return tissue_mask.astype(bool)


def create_composite_plot(data, img, masks, background=None, label_level=None, ax=None, only_show_existing_annotation=True):


    if label_level is None:
        label_level = data.label_level

    colormap = data.colormap
    num_class_to_vis = data.num_classes

    if background is not None:

        for mask in masks.values():

            mask += 1
            mask[cv2.resize(background.astype(np.uint8), mask.shape, interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)] = 0

        colormap = ListedColormap(np.concatenate([np.array([[0., 0., 0., 1.]]), data.colormap.colors]))
        num_class_to_vis = data.num_classes + 1

    encountered_classes = set()

    if ax is None:
        f, ax = plt.subplots(ncols=math.ceil((len(masks)+1)/2), nrows=2)
        ax = ax.T.flatten()

        for a in ax:
            a.set_axis_off()

    else:
        f = None
    if img is not None:
        ax[0].imshow(img, interpolation_stage="rgba")
    for i, (annotator, mask) in enumerate(masks.items()):
        # if img is not None:
        #    ax[i+1].imshow(img)

        encountered_classes |= set(np.unique(mask))

        ax[i+1].imshow(mask.astype(int), alpha=0.8,  cmap=colormap, vmin=0, vmax=num_class_to_vis, interpolation_stage="rgba")
        ax[i+1].set_axis_off()
        ax[i+1].set_title(annotator, size=8)

    ax[0].set_title("WSI", size=8)
    ax[0].set_axis_off()

    if f is not None:
        if background is not None:
            legend_handels = [mpatches.Patch(color=np.array([0., 0., 0., 1.]), label=f"Background")]
            legend_handels += [mpatches.Patch(color=colormap(data.classes_number_mapping[cls]+1), label=cls if len(cls) < 60 else cls[:60]+"...")
                               for cls in data.classes_named if data.classes_number_mapping[cls]+1 in encountered_classes]
        else:
            legend_handels = [mpatches.Patch(color=colormap(data.classes_number_mapping[cls]), label=cls[:40] if len(cls) < 60 else cls[:60]+"...")
                              for cls in data.classes_named if data.classes_number_mapping[cls] in encountered_classes]


        f.legend(handles=legend_handels, loc="outside right", fontsize=6, bbox_to_anchor=(1.4, 0.5))
        
        return f

def create_single_annotator_segmentation_plot(data, image, seg_mask):
    
    f, ax = plt.subplots()

    if image is not None:
        grayscale_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
        ax.imshow(grayscale_image)

    # Create a copy of the original image
    #overlay_image = org_image.copy()
    # Apply the colormap to the segmented regions
    #overlay_image[seg_mask > 0] = seg_colormap(seg_mask[seg_mask > 0])[:, :3]
    #plt.imshow(overlay_image, alpha=0.9)
    
    ax.imshow(seg_mask.astype(int), alpha=0.8, cmap=data.colormap, vmin=0, vmax=data.num_classes-1)

    legend_handels = [mpatches.Patch(color=np.array([0.,0.,0.,1.]), label=f"Unannotated")]
    legend_handels += [mpatches.Patch(color=data.colormap(data.exp_number_mapping[exp]), label=f"{data.exp_grade_mapping[exp]}: " + exp)
        for exp in data.explanations]

    ax.legend(handles=legend_handels, loc="center", bbox_to_anchor=(1.5,0.5), fontsize=6)
    plt.show()


def create_explanation_visualization(data, annotator_exp_slices, show_individual=False):

    if show_individual:
        for (annotator, annotator_exp_slice) in annotator_exp_slices.items():
            plt.imshow(annotator_exp_slice, cmap=cm["Grays"].reversed())
            # plt.title(annotator + "\n" + explanation, size= 8)
            plt.show()

    plt.imshow(np.stack(list(annotator_exp_slices.values()), axis=0).mean(axis=0), cmap=cm["Grays"].reversed(), vmax=1.0)
    plt.show()


def create_segmentation_masks(data, img_index: int, shorter_edge_length=None, img=None, background_mask=None, tissue_mask_kwargs={}, drawing_order="classic"):
    
   
    slide_df = data.get_slide_df(img_index)
    # Open is a lazy operation, so the incurred cost is low, as long as I dont load it.

    if img is None:
        img = data.get_raw_image(img_index)

    annotators_slide = slide_df["annotator"].unique()
    explanations_slide = slide_df["explanations"].unique()

    images_exp_annotator = {}
    images_seg_annotator = {}

    shorter_edge = min(img.size)
    scale_factor = shorter_edge/shorter_edge_length if shorter_edge_length else 1.0
    new_img_size = np.int32(np.round(np.array(img.size) / scale_factor))

    # Removed image loading for better performance. Leaving this legacy code in to document it.
    # Central problem was how PIL, Numpy, CV2 and torch treat image sizes. Also have a look at image_sizes.ipynb.
    # img = img.resize(new_img_size)
    # img_size = np.array(img.size) # I think PIL stores the image in (W,H)
    # img = np.array(img)  # Numpy has the shape (H,W,3)
    # img_size = np.array(img.shape[:2])

    img_size = new_img_size[::-1]

    # Compute the background mask .. this of course reintroduces image loading, but can optionally pass it in if we have loaded it somewhere else.
    if background_mask is None:
        background_mask = ~tissue_filter_image(np.array(img) if not isinstance(img, np.ndarray) else img, size=img_size[::-1], **tissue_mask_kwargs)

    if drawing_order == "frame_order":
        raise RuntimeError('Drawing order frame_order not recommended. Use grade_frame_order instead')
        annotator_images = {}

        for annotator, annotator_frame in slide_df.groupby(by="annotator"):
            annotator_image = np.zeros(list(img_size), dtype=np.int8)  # (H,W)

            for exp, coords in zip(annotator_frame["explanations"], annotator_frame["coords"]):
                new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])  
                label_slice = np.zeros(list(img_size), dtype=np.int8)

                cv2.fillPoly(label_slice, [new_coords.T], color=1)  # Fill Poly expect (W,H) coordinates and a (H,W,C) numpy image.

                annotator_image[label_slice > 0] = np.array(data.exp_number_mapping[exp])

            annotator_images[annotator] = annotator_image.copy()

        return None, annotator_images, background_mask

    if drawing_order == "grade_frame_order":
        annotator_images = {}

        for annotator, annotator_frame in slide_df.groupby(by="annotator"):
            annotator_image = np.zeros(list(img_size), dtype=np.int8)  # (H,W)

            for grade_image, grade_annotator_frame in annotator_frame.groupby("grade", sort=True, observed=True):

                for exp, coords in zip(grade_annotator_frame["explanations"], grade_annotator_frame["coords"]):
                    new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :]) 
                    label_slice = np.zeros(list(img_size), dtype=np.int8)

                    cv2.fillPoly(label_slice, [new_coords.T], color=1)  # Fill Poly expect (W,H) coordinates and a (H,W,C) numpy image.

                    annotator_image[label_slice > 0] = np.array(data.exp_number_mapping[exp])

            annotator_images[annotator] = annotator_image.copy()

        return None, annotator_images, background_mask

    elif drawing_order == "custom_order":
        raise RuntimeError('Drawing order custom_order not recommended. Use grade_frame_order instead.')
        annotator_images = {}

        for annotator, annotator_frame in slide_df.groupby(by="annotator"):
            annotator_image = np.zeros(list(img_size), dtype=np.int8)  # (H,W)

            exp_groups = annotator_frame.groupby("explanations", observed=True)
            custom_order = {0: ["3", "4", "5"],
                            1: ["variable sized well-formed individual and discrete glands",
                                "compressed or angular discrete glands",
                                "poorly formed and fused glands",
                                "Cribriform glands",
                                "Glomeruloid glands",
                                "solid groups of tumor cells",
                                "cords",
                                "single cells",
                                "presence of comedonecrosis",]
                            }
            custom_order = custom_order[data.label_level]

            for exp in custom_order:

                if exp in exp_groups.groups:
                    annotator_exp_frame = exp_groups.get_group(exp)
                else:
                    continue

                for coords in annotator_exp_frame["coords"]:

                    new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])
                    label_slice = np.zeros(list(img_size), dtype=np.int8)  # (H,W)

                    cv2.fillPoly(label_slice, [new_coords.T], color=1)
                    annotator_image[label_slice > 0] = np.array(data.exp_number_mapping[exp])

            annotator_images[annotator] = annotator_image.copy()

        return None, annotator_images, background_mask

    elif drawing_order == "classic":
        raise RuntimeError('Drawing order classic not recommended. Use grade_frame_order instead.')
    
        for exp, exp_frame in slide_df.groupby(by="explanations", observed=True):

            images_exp_annotator[exp] = {}

            num_annotators_per_exp = exp_frame["annotator"].nunique()

            for annotator, annotator_exp_frame in exp_frame.groupby(by="annotator"):

                annotator_exp_slice = np.zeros(list(img_size), dtype=np.int8)  # (H,W)

                for coords in annotator_exp_frame["coords"]:

                    new_coords = np.int32(coords.T * img_size.reshape(-1, 1)[::-1, :])
                    label_slice = np.zeros(list(img_size), dtype=np.int8)  # (H,W)

                    cv2.fillPoly(label_slice, [new_coords.T], color=1)

                    annotator_exp_slice += label_slice

                annotator_exp_slice = annotator_exp_slice > 0
                images_exp_annotator[exp][annotator] = annotator_exp_slice.copy()

        for annotator in annotators_slide:

            relevant_images = [(exp, img) for (exp, exp_images) in images_exp_annotator.items() for (anno, img) in exp_images.items() if anno == annotator]

            relevant_images = sorted(relevant_images, key=lambda x: (data.exp_grade_mapping[x[0]], x))

            seg_mask = np.zeros((*img_size,), dtype=np.int8)
            for exp, rel_img in relevant_images:

                seg_mask[rel_img] = np.array(data.exp_number_mapping[exp])

            images_seg_annotator[annotator] = seg_mask.copy()

        return images_exp_annotator, images_seg_annotator, background_mask
