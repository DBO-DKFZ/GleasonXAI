
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.jdt_losses import SoftCorrectDICEMetric


def add_circular_mask_to_array(center, array, radius, value=1.0):
    """
    Create a circular mask in a zero NumPy array with the given radius and center.
    
    Parameters:
    radius (int): The radius of the circle.
    center (tuple): The (x, y) coordinates of the circle's center.
    array_size (tuple): The size of the array (height, width).
    
    Returns:
    np.ndarray: A NumPy array with a circular mask.
    """
    # Create indices grid
    Y, X = np.ogrid[:radius*2, :radius*2]

    # Calculate the distance from the center
    dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)

    # Create the circular mask
    mask = dist_from_center <= radius
    mask = mask.astype(float) * value

    array[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius] = mask
    return array

# %%


def create_gaussian_distribution(std, center, array_size):
    """
    Create a Gaussian distribution in a NumPy array with the given center and standard deviation.
    
    Parameters:
    center (tuple): The (x, y) coordinates of the center of the Gaussian distribution.
    std (float): The standard deviation of the Gaussian distribution.
    array_size (tuple): The size of the array (height, width).
    
    Returns:
    np.ndarray: A NumPy array with a Gaussian distribution.
    """
    # Create a zero array
    array = np.zeros(array_size, dtype=np.float32)

    # Create indices grid
    Y, X = np.ogrid[:array_size[0], :array_size[1]]

    # Calculate the Gaussian distribution
    gaussian = np.exp(-((X - center[0])**2 + (Y - center[1])**2) / (2 * std**2))

    return gaussian

# %%


def add_truncated_gaussian_distribution_to_array(center, array, std, radius):
    """
    Create a truncated Gaussian distribution in a NumPy array with the given center, standard deviation, and truncation radius.
    
    Parameters:
    center (tuple): The (x, y) coordinates of the center of the Gaussian distribution.
    std (float): The standard deviation of the Gaussian distribution.
    array_size (tuple): The size of the array (height, width).
    radius (float): The radius beyond which the Gaussian distribution is truncated to 0.
    
    Returns:
    np.ndarray: A NumPy array with a truncated Gaussian distribution.
    """

    # Create indices grid
    Y, X = np.ogrid[:radius*2, :radius*2]

    # Calculate the distance from the center
    dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)

    # Calculate the Gaussian distribution
    gaussian = np.exp(-((X - radius)**2 + (Y - radius)**2) / (2 * std**2))

    # Apply the truncation radius
    gaussian[dist_from_center > radius] = 0

    gaussian /= gaussian.sum()

    array[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius] = gaussian

    return array


# %%
img = np.zeros((500, 1000))

img = add_truncated_gaussian_distribution_to_array((250, 250), img, 150, 220)
img = add_circular_mask_to_array((250, 750), img, 220, value=img.max())
plt.imshow(img, cmap="Greys")

# %%


def get_hard_dist(array_size, radius, pos):

    hard_dist = np.zeros((*array_size, 2))
    hard_dist[:, :, 0] = add_circular_mask_to_array(pos, hard_dist[:, :, 0], radius, value=1.0)
    hard_dist[:, :, 1] = 1 - hard_dist[:, :, 0]
    hard_dist = torch.tensor(hard_dist).permute(2, 0, 1).unsqueeze(0)
    return hard_dist


def get_soft_dist(array_size, radius, std, pos):
    soft_dist = soft_dist = np.zeros((*array_size, 2))
    soft_dist[:, :, 0] = add_truncated_gaussian_distribution_to_array(pos, soft_dist[:, :, 0], std, radius)
    soft_dist[:, :, 0] *= 1/soft_dist[:, :, 0].max()
    soft_dist[:, :, 1] = 1 - soft_dist[:, :, 0]
    soft_dist = torch.tensor(soft_dist).permute(2, 0, 1).unsqueeze(0)
    return soft_dist
# scale so that maximum is 1


# %%
hard_dist = np.zeros((500, 1000, 2))
hard_dist[:, :, 0] = add_circular_mask_to_array((250, 350), hard_dist[:, :, 0], 220, value=1.0)
hard_dist[:, :, 1] = 1 - hard_dist[:, :, 0]
hard_dist = torch.tensor(hard_dist).permute(2, 0, 1).unsqueeze(0)

soft_dist = soft_dist = np.zeros((500, 1000, 2))
soft_dist[:, :, 0] = add_truncated_gaussian_distribution_to_array((250, 400), soft_dist[:, :, 0], 150, 220)
soft_dist[:, :, 0] *= 1/soft_dist[:, :, 0].max()
soft_dist[:, :, 1] = 1 - soft_dist[:, :, 0]
soft_dist = torch.tensor(soft_dist).permute(2, 0, 1).unsqueeze(0)
# scale so that maximum is 1

# %%
plt.imshow(soft_dist[0, 0, :, :]+hard_dist[0, 0, :, :], cmap="Greys")

# %%
left = 250
right = 750
radius = 220

img_size = (500, 1000)
hard_dist_left = get_hard_dist(img_size, radius, (250, left))
hard_dist_right = get_hard_dist(img_size, radius, (250, right))

soft_dist_left = get_soft_dist(img_size, radius, 150, (250, left))
soft_dist_right = get_soft_dist(img_size, radius, 150, (250, right))


left = 350
right = 450
hard_dist_over_left = get_hard_dist(img_size, radius, (250, left))
hard_dist_over_right = get_hard_dist(img_size, radius, (250, right))

soft_dist_over_left = get_soft_dist(img_size, radius, 150, (250, left))
soft_dist_over_right = get_soft_dist(img_size, radius, 150, (250, right))

# %%


def get_soft_iou(a, b, class_wise=True):
    metric = SoftCorrectDICEMetric(average="mIoUD")
    metric.update(a[:, 0, :, :].unsqueeze(1), b[:, 0, :, :].unsqueeze(1))
    return metric.compute()


fig, axes = plt.subplots(4, 1)
axes = axes.flatten()
for ax in axes:
    ax.set_axis_off()


def generate_plot(ax, a, b, title):
    color_img = np.zeros((a.shape[2], a.shape[3], 3), dtype=np.uint8) * 255

    dice = get_soft_iou(a, b)

    # A, B sind BATCH,CLASSES,X,Y Tensoren mit den Class Probs
    # Ich nehme mir von dem ersten Bild die Erste Klasse.
    a = a[0, 0].numpy()
    b = b[0, 0].numpy()

    # Nur die Stellen wo die größer Null sind.
    a_mask = a > 0.0
    b_mask = b > 0.0

    # color_img[~np.logical_or(a_mask, b_mask), :] = 0

    # Setze für A den Red Channel und für B den Blue Channel
    color_img[a_mask, 0] = (a[a_mask] * 255).astype(np.uint8)

    color_img[b_mask, 2] = (b[b_mask] * 255).astype(np.uint8)

    color_img = color_img.astype(int)
    # Jetzt sollte man ein weißes Bild haben das an den Stellen von A und B einen roten und blauen Kreis hat
    ax.imshow(color_img, )
    ax.set_title("SoftDICE: " + f"{float(dice):0.3f}", fontsize=5)


axes[0].imshow((hard_dist_over_left[0, 0, :, :]+hard_dist_over_right[0, 0, :, :]).clip(max=1.0), cmap="Greys")
print("Overlapping Hard:Hard", get_soft_iou(hard_dist_over_left, hard_dist_over_right))
axes[1].imshow((soft_dist_over_left[0, 0, :, :]+hard_dist_over_right[0, 0, :, :]).clip(max=1.0), cmap="Greys")
print("Overlapping Soft:Hard", get_soft_iou(soft_dist_over_left, hard_dist_over_right))
axes[2].imshow((soft_dist_over_left[0, 0, :, :]+hard_dist_over_left[0, 0, :, :]).clip(max=1.0), cmap="Greys")
print("Overlapping Hard:Soft", get_soft_iou(hard_dist_over_left, soft_dist_over_right))
axes[3].imshow((soft_dist_over_left[0, 0, :, :]+soft_dist_over_right[0, 0, :, :]).clip(max=1.0), cmap="Greys")
print("Overlapping Soft:Soft", get_soft_iou(soft_dist_over_left, soft_dist_over_right))

# %%
fig, axes = plt.subplots(3, 3, figsize=(10, 7))
axes = axes.flatten()
for ax in axes:
    ax.set_axis_off()

axes = axes.T
generate_plot(axes[0],  hard_dist_left, hard_dist_right, "No Overlap Hard:Hard")
generate_plot(axes[1],  soft_dist_left, soft_dist_right, "No Overlap Soft:Soft")
generate_plot(axes[2], hard_dist_left, soft_dist_right, "No Overlap Hard:Soft")


generate_plot(axes[3], hard_dist_left, hard_dist_left, "Full Overlap Hard:Hard")
generate_plot(axes[4], soft_dist_left, soft_dist_left, "Full Overlap Soft:Soft")
generate_plot(axes[5], hard_dist_left, soft_dist_left, "Full Overlap Hard:Soft")


generate_plot(axes[6], hard_dist_over_left, hard_dist_over_right, "Partial Overlap Hard:Hard")
generate_plot(axes[7], soft_dist_over_left, soft_dist_over_right, "Partial Overlap Soft:Soft")
generate_plot(axes[8], hard_dist_over_left, soft_dist_over_right, "Partial Overlap Hard:Soft")

axes = axes.reshape(3, 3)
row_titles = ["No Overlap", "Full Overlap", "Partial Overlap"]
col_titles = ["Hard:Hard", "Soft:Soft", "Hard:Soft"]


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


fig.tight_layout()
add_headers(fig, row_headers=row_titles, col_headers=col_titles, col_pad=15, row_pad=-20, rotate_row_headers=True)
Path("./figures").mkdir(exist_ok=True)
plt.savefig("./figures/SoftDiceVis.png", dpi=200)

# %%
a = hard_dist_left
b = soft_dist_right

color_img = np.zeros((a.shape[2], a.shape[3], 3), dtype=np.uint8)

dice = get_soft_iou(a, b)

# A, B sind BATCH,CLASSES,X,Y Tensoren mit den Class Probs
# Ich nehme mir von dem ersten Bild die Erste Klasse.
a = a[0, 0].numpy()
b = b[0, 0].numpy()

# Nur die Stellen wo die größer Null sind.
a_mask = a > 0.0
b_mask = b > 0.0

# Setze für A den Red Channel und für B den Blue Channel
color_img[a_mask, 0] = (a[a_mask] * 255).astype(np.uint8)
color_img[b_mask, 2] = (b[b_mask] * 255).astype(np.uint8)

color_img[~np.logical_or(a_mask, b_mask), :] = 255


color_img = color_img.astype(int)
# Jetzt sollte man ein weißes Bild haben das an den Stellen von A und B einen roten und blauen Kreis hat
plt.imshow(color_img, )
