# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib.gridspec import GridSpec

from pathlib import Path

import wandb

# %%

# output dir
fig_dir = Path('./figures')
results_dir = Path('./results')

# Replace these with your values
entity = "dkfz"
project = "GleasonXAI"

# Initialize WandB API
api = wandb.Api()

# Get all runs in the project
runs = api.runs(f"{entity}/{project}",)
runs = [run for run in runs if "GleasonFinal2/" in run.name]
# Initialize an empty list to store run data
run_data = []

for run in runs:
    # Get summary, config, and metrics for each run
    summary = run.summary._json_dict
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    # metrics = run.history()
    summary = {k: v for k, v in summary.items() if "test" in k}
    # Add run data to the list

    label_level = int(config["dataset"]["label_level"])
    _, _, _, name = run.name.split("/")
    name, version = name.split("-")

    # Remap metrics

    new_summary = {}

    for k, v in summary.items():
        if not "dataloader_idx_" in k:
            k = k + f"_label_level_{label_level}"
            new_summary[k] = v

        else:
            base_metric, idx = k.split("/dataloader_idx_")
            idx = int(idx)
            new_ll = label_level-idx

            k = base_metric + f"_label_level_{new_ll}"
            new_summary[k] = v

    run_data.append({
        "name": name,
        "version": version,
        "label_level": label_level,
        "run_name": run.name,
        "run_id": run.id,
        **new_summary,
        # "config": config,
        # "metrics": metrics
    })

# Convert the list of run data to a pandas DataFrame
df = pd.DataFrame(run_data)
df = df.drop(columns=["run_name", "run_id"]).set_index(["name", "version", "label_level"])
df.to_csv(results_dir / "wandb_run_data.csv", index=False)
# %%


def split_column_name(col_name):
    # Split the column name based on the '_label_level_' separator
    if '_label_level_' in col_name:
        metric, label_level = col_name.split('_label_level_')
        return (metric, label_level)
    else:
        # If there's no '_label_level_' in the name, keep it in the first level
        return (col_name, '')


# Apply the function to create tuples for the MultiIndex
multi_index_tuples = [split_column_name(col) for col in df.columns]

# Create the MultiIndex
multi_index = pd.MultiIndex.from_tuples(multi_index_tuples, names=['Metric', 'Result Label Level'])

# Apply the MultiIndex to the columns of the DataFrame
df.columns = multi_index

df.to_csv(results_dir / "downloaded_wandb_results.csv")

# %%

# Clean frame
label_level_name_mapping = {0: "Gleason Patterns", 1: "Grouped Explanations", 2: "Explanations"}

df_melted = df.reset_index().melt(
    id_vars=["name", "version", "label_level"],  # Variables to keep in the long format
    var_name=["Metric", "Label Level"],  # New names for the metric and label level columns
    value_name="Value"  # Name for the metric values
)

label_level_name_mapping = {0: "Gleason patterns", 1: "explanations", 2: "sub-explanations"}

df_melted = df_melted.rename(columns={"Label Level": "Evaluated on", "label_level": "Trained on"})
df_melted["Trained on"] = df_melted["Trained on"].replace(label_level_name_mapping)
df_melted["Evaluated on"] = df_melted["Evaluated on"].astype(int).replace(label_level_name_mapping)

metric_name_remappings = {"test_L1": "L1", "test_b_acc_unique_max": "Balanced Accuracy",
                          "test_acc_unique_max": "Accuracy", "test_soft_DICEDataset": "Macro SoftDice",
                          "test_DICE_unique_max": "Dice", "test_b_DICE_unique_max": "Macro Dice",
                          "test_b_soft_DICE": "SoftDiceBalancedOld", "test_soft_DICE": "SoftDiceOld"}

run_name_remappings = {"SoftDiceBalanced": "SoftDiceLoss", "CE": "cross-entropy loss (soft label)",
                       "SoftDiceBalancedMultiLevel": "TreeLoss(SoftDiceLoss)",
                       "OH_CE": "cross-entropy loss (majority vote)", "DICE": "Dice loss (majority vote)",
                       "JDTLoss": "SoftDice Wang et al."}


df_melted["Metric"] = df_melted["Metric"].replace(metric_name_remappings)
df_melted["name"] = df_melted["name"].replace(run_name_remappings)


available_metrics = [
    "Macro SoftDice",
    "L1",
    "Dice",
    "Macro Dice",
    "Accuracy",
    "Balanced Accuracy",
]

available_names = ["SoftDiceLoss",
                   "cross-entropy loss (soft label)",
                   "TreeLoss(SoftDiceLoss)",
                   "Dice loss (majority vote)",
                   "cross-entropy loss (majority vote)",
                   "SoftDice Wang et al.",
                   ]


df_melted = df_melted[df_melted["name"].isin(available_names)]
df_melted = df_melted[df_melted["Metric"].isin(available_metrics)]
df_melted = df_melted[df_melted["Value"].notna()]

df_melted.to_csv(results_dir / "publication_ready_results.csv")

# %%

blue_colors = cm["Blues"](np.linspace(0.1, 0.9, 3+2))[1:-1]
red_colors = cm["Reds"](np.linspace(0.1, 0.9, 3+2))[1:-1]

color_palette = {"SoftDiceLoss": blue_colors[2],
                 "cross-entropy loss (soft label)": blue_colors[1],
                 "TreeLoss(SoftDiceLoss)": blue_colors[0],
                 "Dice loss (majority vote)": red_colors[2],
                 "cross-entropy loss (majority vote)": red_colors[1]}

df_melted = pd.read_csv(results_dir / "publication_ready_results.csv")


# ----------
# Settings
# ----------
selected_metrics = [
    "Macro SoftDice",
    "L1",
    "Dice",
    "Macro Dice",
    # "Accuracy",
    # "Balanced Accuracy",
    # "SoftDiceOld",
    # "SoftDiceBalancedOld",
]

selected_names = ["SoftDiceLoss",
                  "cross-entropy loss (soft label)",
                  # "TreeLoss(SoftDiceLoss)",
                  "Dice loss (majority vote)",
                  "cross-entropy loss (majority vote)",
                  # "SoftDice Wang et al.",
                  ]

eval_on_levels = ['Gleason patterns',
                  'explanations',
                  # 'sub-explanations',
                  ]

trained_on_levels = ['Gleason patterns',
                     'explanations',
                     # 'sub-explanations',
                     ]

width_ratios = np.arange(len(eval_on_levels), 0, -1)
share_y = True

hspace = 0.05
wspace = 0.0 if share_y else 0.1

# IMPORTANT FOR THE UPPER PLOT THIS WAS 0.2. BUT THAT LEADS TO OVERLAP SO WITH MANY BARS LOWER THIS TO 0.19
bar_width = 0.2

col_width = 1.5
metric_height = 2

eval_on_offset = 0.93
metrics_offset = 0.085

df_melted = df_melted[df_melted["Metric"].isin(selected_metrics)]
df_melted = df_melted[df_melted["name"].isin(selected_names)]
df_melted = df_melted[df_melted["Value"].notna()]


for e in eval_on_levels:
    assert e in df_melted["Evaluated on"].unique(), (e, df_melted["Evaluated on"].unique())

num_selected_names = len(df_melted["name"].unique())
num_selected_metrics = len(df_melted["Metric"].unique())
num_eval_on_levels = len(eval_on_levels)


plot_width = col_width*num_eval_on_levels*num_selected_names
plot_height = num_selected_metrics*metric_height

fig = plt.figure(figsize=(plot_width, plot_height))

# Define GridSpec with different column widths
gs = GridSpec(num_selected_metrics, num_eval_on_levels, width_ratios=width_ratios, hspace=hspace, wspace=wspace)

# Create subplots using the GridSpec
axes = np.empty((num_selected_metrics, num_eval_on_levels), dtype=np.object_)
for i in range(num_selected_metrics):
    for j in range(num_eval_on_levels):
        if j == 0:
            ax = fig.add_subplot(gs[i, j])  # First column: no sharing
        else:
            if share_y:
                sy = axes[i, 0]
            else:
                sy = None

            ax = fig.add_subplot(gs[i, j], sharey=sy)  # Share y with the first column in the same row        axes.append(ax)

        axes[i, j] = ax

for j, eval_on in enumerate(eval_on_levels):

    for i, metric in enumerate(selected_metrics):
        _df = df_melted[df_melted["Evaluated on"] == eval_on]
        _df = _df[_df["Metric"] == metric]

        trained_on = [t for t in _df["Trained on"].unique() if t in trained_on_levels]

        x_order = [l for l in label_level_name_mapping.values() if l in trained_on]
        hue_order = [name for name in selected_names if name in _df["name"].unique()]
        c_p = [color_palette[name] for name in selected_names if name in _df["name"].unique()]

        x_positions = np.arange(len(x_order))

        for x_val in x_order:

            offset = 0.0

            x_data_subset = _df[_df["Trained on"] == x_val]

            num_hues = x_data_subset["name"].nunique()
            for idx, name in enumerate(hue_order):
                # Filter data for the current hue
                hue_data_subset = x_data_subset[x_data_subset['name'] == name]

                # Get the value for the current x and hue combination
                y_vals = hue_data_subset["Value"].values

                if len(y_vals) > 0:  # If there is a value to plot
                    # Calculate the position for the bar
                    x_pos = x_positions[x_order.index(x_val)]
                    bar_pos = x_pos + offset - (bar_width * (num_hues - 1) / 2)

                    mean_val = np.mean(y_vals)
                    std_val = np.std(y_vals)

                    # Plot the bar
                    bar = axes[i, j].bar(
                        bar_pos,
                        mean_val,
                        width=bar_width,
                        label=name,
                        color=color_palette[name],
                        capsize=4,  # Set the capsize for error bars
                        yerr=std_val  # Standard deviation as error
                    )

                    mean_str = f"{mean_val: .2f}"
                    # Add the bar label
                    axes[i, j].text(
                        bar_pos,
                        mean_val/2,
                        mean_str,
                        ha='center',
                        va='center',
                        rotation=90
                    )

                    # Update the offset for the next hue in this x category
                    offset += bar_width

        axes[i, j].set_xticks(x_positions)
        axes[i, j].set_xticklabels(x_order)

# Hide labels and titles in the grid
for a in axes.flatten():
    a.set_ylabel("")
    a.set_title("")
    a.set_xlabel("")


# Left most plots get the metric name as y label
for a, met in zip(axes[:, 0], selected_metrics):
    a.set_ylabel(met)

# Hide the y axis of in between plots
for a in axes[:, 1:].flatten():
    if share_y:
        a.get_yaxis().set_visible(False)


# same for x axis
for a in axes[:-1, :].flatten():
    a.get_xaxis().set_visible(False)


for a in axes.flatten():
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    # a.spines['left'].set_visible(False)
    # a.spines['bottom'].set_visible(False)

for a, eval_on in zip(axes[0, :], eval_on_levels):
    a.set_title(eval_on)

for a, eval_on in zip(axes[-1, :], eval_on_levels):
    a.set_xlabel("Trained on")

# Have to add them with per hand spacing...
fig.text(0.5, eval_on_offset, "Evaluated on", va='center', fontsize=16, fontweight='bold')
fig.text(metrics_offset, 0.5, "Metrics", va='center', rotation='vertical', fontsize=16, fontweight='bold')


handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one of the subplots

legend_dict = {}

for ax in axes.flatten():
    handles, labels = ax.get_legend_handles_labels()

    for l, h in zip(labels, handles):
        if not l in legend_dict:
            legend_dict[l] = h

fig.legend(list(legend_dict.values()), list(legend_dict.keys()), loc='center', ncol=len(selected_names), bbox_to_anchor=(0.5, 0.03), fontsize=12)

plt.savefig(fig_dir / "barplot_mainpaper.svg")

# %%
# Calculate mean and std, then format
formatted_df = (
    df_melted
    .set_index(["name", "version", "Trained on", "Evaluated on"])
    .pivot(columns=["Metric"], values="Value")
    .groupby(["Evaluated on", "Trained on", "name"])
    .aggregate(["mean", "std"])
)

formatted_df.columns = ['_'.join(col).strip() for col in formatted_df.columns.values]

# Combine mean and std into a formatted string for each cell
for column in formatted_df.columns:
    if 'mean' in column:
        metric_name = column.split('_mean')[0]
        std_col = f'{metric_name}_std'
        formatted_df[metric_name] = formatted_df.apply(
            lambda row: f"{row[column]:.3f}_{{\pm {row[std_col]:.3f}}}", axis=1
        )

# Keep only the formatted columns
formatted_df = formatted_df[[col for col in formatted_df.columns if 'mean' not in col and 'std' not in col]]

formatted_df.reset_index(inplace=True)
formatted_df.to_csv(results_dir / "publication_table_df.csv")

# %%
