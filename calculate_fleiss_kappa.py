import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap, norm
from statsmodels.stats import inter_rater as irr

from src.gleason_data import GleasonX

label_mapping = {
    "compressed or angular discrete glands": 0,
    "variable sized well-formed individual and discrete glands": 1,
    "Cribriform glands": 2,
    "Glomeruloid glands": 3,
    "poorly formed and fused glands": 4,
    "presence of comedonecrosis": 5,
    "cords": 6,
    "solid groups of tumor cells": 7,
    "single cells": 8,
}

label_renaming = {
    "poorly formed and fused glands": "4 - poorly formed glands",
    "variable sized well-formed individual and discrete glands": "3 - individual glands",
    "cords": "5 - cords",
    "presence of comedonecrosis": "5 - comedonecrosis",
    "solid groups of tumor cells": "5 - groups of tumor cells",
    "Cribriform glands": "4 - cribriform glands",
    "Glomeruloid glands": "4 - glomeruloid glands",
    "compressed or angular discrete glands": "3 - compressed glands",
    "single cells": "5 - single cells",
}

base_label_mapping = {
    "single, individual atypical glands separated from each other": "3.01",
    "atypical glands with an irregularly separated, ragged, poorly defined edge": "3.02",
    "atypical glands are looser than a nodule and are infiltrative": "3.03",
    "either minute or large and cyst-like atrophic atypical glands": "3.04",
    "atypical glands lying very closely together (with little stroma between adjacent atypical glands)": "3.05",
    "well-formed, relatively uniform atypical glands with evenly distributed lumina": "3.06",
    "compressed or angular atypical glands": "3.07",
    "atypical glands infiltrate between benign glands": "3.08",
    "slit-like lumina": "4.01",
    "large atypical glands": "4.02",
    "irregular contours, jagged edges of atypical glands": "4.03",
    "atypical glands fused or grown together into cords or chains": "4.04",
    "irregular distribution of lumina": "4.05",
    "atypical glands very close together (with little or no stroma)": "4.06",
    "cribriform": "4.07",
    "cribriform larger than a normal prostate gland; tends to fragmentation": "4.08",
    "cribriform confluent sheet of contiguous carcinoma cells with multiple glandular lumina that are easily visible at low power (objective magnification 10x)": "4.09",
    "cribriform single or fused glandular structures connected to each other (no intervening stroma or mucin)": "4.10",
    "hypernephroid pattern": "4.11",
    "hypernephroid pattern nests of clear cells resembling renal cell carcinoma": "4.12",
    "hypernephroid pattern small, hyperchromatic nuclei": "4.13",
    "hypernephroid pattern fusion of acini into more solid sheets with the appearance of back-to-back glands without intervening stroma": "4.14",
    "glomeruloid pattern": "4.15",
    "glomeruloid pattern rare small cribriform variant resembling glomerulus structures of kidney": "4.16",
    "glomeruloid pattern contains a tuft of cells that is largely detached from its surrounding duct space except for a single point of attachment": "4.17",
    "solid tumor cell clusters with nonpolar nuclei around a lumen": "5.01",
    "presence of definite comedonecrosis (central necrosis)": "5.02",
    "presence of definite comedonecrosis (central necrosis) with intraluminal necrotic cells": "5.03",
    "presence of definite comedonecrosis (central necrosis) with karyorrhexis within papillary, cribriform spaces": "5.04",
    "single cells": "5.05",
    "single cells forming cords": "5.06",
    "single cells with vacuoles (signet ring cells) but without glandular lumina": "5.07",
}

tmas_with_multiple_groups = ["ZT111_4_A_7_10", "ZT199_1_B_3_4", "ZT204_6_A_8_6"]


def calculate_kappa_per_group_and_label(df: pd.DataFrame, out_path: Path, use_sub_explanations: bool):
    group_grpd_df = df.groupby("group")

    out_append = "pattern" if use_sub_explanations == None else "sub-expl" if use_sub_explanations == True else "expl"
    out_path = out_path / out_append

    if not out_path.exists():
        os.makedirs(out_path, exist_ok=True)

    group_label_kappas = dict()
    group_label_kappas_CIs = dict()
    label_group_kappas_p = dict()

    total_group_kappa = dict()
    total_group_kappa_CIs = dict()
    total_group_kappa_p = dict()

    labelwise_collection = defaultdict(list)
    all_three_annotator_labelwise_CIs = defaultdict(list)

    for group_name, group_series in group_grpd_df:
        print("Group:", group_name)

        num_annotators = group_series["annotator"].nunique()
        assert num_annotators == 3
        annotators = group_series["annotator"].unique()

        labels = ["3", "4", "5"] if use_sub_explanations == None else base_label_mapping.keys() if use_sub_explanations else label_mapping.keys()
        current_grp_annotation_kappas = []
        current_grp_annotation_CIs = []
        current_grp_annotation_p = []

        all_decisions_of_current_group = []

        for an_idx, annotation in enumerate(labels):
            current_group_current_annotation_existance_list = []

            # GET RATING FOR EACH TMA AND ANNOTATOR
            k = 0
            full_annotated = []
            for tma, tma_series in group_series.groupby("TMA_identifier"):
                if tma in tmas_with_multiple_groups:
                    continue

                current_group_current_annotation_existance_list.append(
                    np.array(get_subject_description_per_rater(tma_series, annotators, annotation, use_sub_explanations))
                )
                if num_annotators == 3:
                    labelwise_collection[an_idx].append(np.array(get_subject_description_per_rater(tma_series, annotators, annotation, use_sub_explanations)))
                    k += 1

            all_decisions_of_current_group.extend(current_group_current_annotation_existance_list)
            current_group_current_annotation_existance_list = np.array(current_group_current_annotation_existance_list)

            # CALCULATE FLEISS KAPPA for current group & annotation
            fl_kappa, lower, upper, p = calculate_fleiss_k(current_group_current_annotation_existance_list, f"{group_name}_{annotation}")
            current_grp_annotation_kappas.append(fl_kappa)
            current_grp_annotation_CIs.append((lower, upper))
            current_grp_annotation_p.append(p)

        group_label_kappas[group_name] = current_grp_annotation_kappas
        group_label_kappas_CIs[group_name] = current_grp_annotation_CIs

        total_group_kappa[group_name] = [calculate_fleiss_k(all_decisions_of_current_group, f"{group_name}")[0]]
        total_group_kappa_CIs[group_name] = [calculate_fleiss_k(all_decisions_of_current_group, f"{group_name}")[1:-1]]
        total_group_kappa_p[group_name] = [calculate_fleiss_k(all_decisions_of_current_group, f"{group_name}")[-1]]

    print("- calculating kappa per label -")
    label_kappa = [calculate_fleiss_k(labelwise_collection[i], f"label_{i}")[0] for i in range(len(labels))]
    label_CI = [calculate_fleiss_k(labelwise_collection[i], f"label_{i}")[1:-1] for i in range(len(labels))]
    label_p = [calculate_fleiss_k(labelwise_collection[i], f"label_{i}")[-1] for i in range(len(labels))]

    print("- creating data frames -")
    kappas_df = pd.DataFrame.from_dict(group_label_kappas)
    kappas_CIs_df = pd.DataFrame.from_dict(group_label_kappas_CIs)
    group_kappas_df = pd.DataFrame.from_dict(total_group_kappa)
    group_kappa_CIs_df = pd.DataFrame.from_dict(total_group_kappa_CIs)
    group_kappa_p_df = pd.DataFrame.from_dict(total_group_kappa_p)

    renamed_labels = (
        labels
        if use_sub_explanations == None
        else [base_label_mapping[lbl] for lbl in labels] if use_sub_explanations else [label_renaming[lbl] for lbl in labels]
    )
    # cmap = {0:'forestgreen', 1: 'royalblue', 2: 'firebrick'} if raw== None else {i: "forestgreen" if i < 8 else "royalblue" if i < 25 else 'firebrick' for i in range(0, len(renamed_labels))} if raw else {i: "forestgreen" if i < 2 else "royalblue" if i < 5 else 'firebrick' for i in range(0, len(renamed_labels))}

    print("- creating figures -")
    plt.figure(figsize=(20, 5))
    sns.heatmap(kappas_df, annot=True, fmt=".3g", yticklabels=renamed_labels, cmap="Blues", cbar=False, annot_kws={"size": 14}, linewidths=0.5)
    # plt.title("Fleiss-Kappa per group and label")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path / "0_label_group_kappas.png", dpi=1000)

    plt.clf()
    plt.cla()
    plt.close()

    for col in kappas_CIs_df.columns:
        kappas_CIs_df[col] = kappas_CIs_df[col].apply(nan_replace_ci)
    # kappas_CIs_df = kappas_CIs_df.map(nan_replace_ci)

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    tabla = pd.plotting.table(ax, kappas_CIs_df, loc="upper right", cellLoc="center")

    # plt.title("Fleiss-Kappa per group and label")
    plt.tight_layout()
    plt.savefig(out_path / "label_group_kappa_CIs_table.png", transparent=True, dpi=500)

    plt.clf()
    plt.cla()
    plt.close()

    print(f"Boxplot for label/group kappas")
    kappas_df.to_csv(out_path / f"{'expl' if use_sub_explanations==False else 'sub-expl' if use_sub_explanations == True else 'pattern'}_kappas.csv")
    np.save(
        out_path / f"{'expl' if use_sub_explanations==False else 'sub-expl' if use_sub_explanations == True else 'pattern'}_kappas_y-lables.npy", renamed_labels
    )

    figsize = (6, 2.6) if use_sub_explanations == None else (5, 16) if use_sub_explanations == True else (10, 9)
    plt.figure(figsize=(figsize))
    ax_bp = sns.boxplot(
        data=kappas_df.T,
        color="#77b5d9",
        showfliers=False,
        width=0.5,
        native_scale=False,
        orient="h",
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "gray"},
    )  # data=kappas_df.T, palette=cmap
    plt.xlim(-0.3, 1.1)
    plt.grid(axis="x", linestyle="--", linewidth=0.5, color="lightgray")
    ax_bp.set_yticklabels(renamed_labels)
    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=22)
    sns.stripplot(data=kappas_df.T, color="#77b5d9", jitter=False, linewidth=1, orient="h")

    # plt.title("Fleiss-Kappa per label")
    plt.tight_layout()
    plt.savefig(out_path / "1_boxplot_kappas.png", dpi=1000)

    print("----")
    plt.clf()
    plt.cla()
    plt.close()

    print("Fleiss-k per group")

    plt.figure(figsize=(20, 4))
    sns.heatmap(
        group_kappas_df,
        annot=True,
        yticklabels="",
        fmt=".3g",
        vmin=-0.24,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.3},
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 20},
        linewidths=0.5,
    )
    # plt.title("Fleiss-Kappa per group")
    plt.tight_layout()
    plt.savefig(out_path / "0_group_kappas.png", dpi=1000)

    plt.clf()
    plt.cla()
    plt.close()

    merged_group_kappa = pd.concat([group_kappas_df.T, group_kappa_CIs_df.T], axis=1)
    merged_group_kappa.index.name = "group"
    merged_group_kappa.columns = ["Fleiss-K", "95% Confidence Interval"]
    merged_group_kappa["95% Confidence Interval"] = merged_group_kappa["95% Confidence Interval"].apply(nan_replace_ci)
    merged_group_kappa["Fleiss-K"] = merged_group_kappa["Fleiss-K"].apply(nan_replace_val)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    tabla = pd.plotting.table(ax, merged_group_kappa, loc="upper right", cellLoc="center", colWidths=[0.17] * len(df.columns))  # where df is your data frame
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.2)
    plt.tight_layout()
    plt.savefig(out_path / "group_kappa_table.png", transparent=True, dpi=500)

    print("---")

    plt.clf()
    plt.cla()
    plt.close()

    print("Fleiss-k per label")
    figsize = (20, 4)
    c_bar = {"shrink": 0.3}
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        pd.DataFrame(label_kappa).T,
        annot=True,
        yticklabels="",
        fmt=".3g",
        vmin=-0.24,
        vmax=1,
        cbar_kws=c_bar,
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 20},
        linewidths=0.5,
        square=True,
    )
    ax.set_xticklabels(renamed_labels)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    # plt.title("Fleiss-Kappa per label\n(rated by three raters)")
    plt.tight_layout()
    plt.savefig(out_path / "0_label_kappas_3_raters.png", dpi=1000)

    total_labels = [(l, f"{x:.3f}", f"[{y:.3f}, {z:.3f}]") for (x, (y, z), l) in zip(label_kappa, label_CI, renamed_labels)]
    total_label_df = pd.DataFrame(total_labels)
    total_label_df.columns = ["Label", "Fleiss-K", "95% Confidence Interval"]

    total_label_df["Fleiss-K"] = total_label_df["Fleiss-K"].apply(nan_replace_str_val)
    total_label_df["95% Confidence Interval"] = total_label_df["95% Confidence Interval"].apply(nan_replace_str_ci)
    total_label_df.set_index("Label", inplace=True)

    plt.clf()
    plt.cla()
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    tabla = pd.plotting.table(ax, total_label_df, loc="upper right", colWidths=[0.17] * len(df.columns), cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.2)
    plt.savefig(out_path / "label_kappa_table.png", transparent=True, dpi=500)


def nan_replace_val(x):
    if math.isnan(x):
        return f"-"
    return f"{x:.3f}"


def nan_replace_val_e(x):
    if math.isnan(x):
        return f"-"
    return f"{x:.3e}"


def nan_replace_ci(x):
    if math.isnan(x[0]):
        return f"[-,-]"
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def nan_replace_str_ci(x: str):
    if x.startswith("[na"):
        return f"[-,-]"
    return x


def nan_replace_str_val(x: str):
    if x.startswith("na"):
        return f"-"
    return x


def calculate_fleiss_k(annotation_list: list, name: str):
    aggregated = irr.aggregate_raters(annotation_list, n_cat=2)
    upper, lower, p = calculate_confidence_interval(aggregated[0])
    return irr.fleiss_kappa(aggregated[0], "fleiss"), lower, upper, p


def calculate_confidence_interval(formatted_data):
    # Number of bootstrap samples
    n_resamples = 10000

    formatted_df = pd.DataFrame(formatted_data)
    kappa = irr.fleiss_kappa(formatted_df.values, method="fleiss")

    wrapped_list = [combi[0] for combi in formatted_df.values]

    # Perform bootstrap resampling
    res = bootstrap((wrapped_list,), bootstrap_kappa, n_resamples=n_resamples, vectorized=False, axis=0)

    # Calculate the confidence interval
    lower_CI, upper_CI = res.confidence_interval
    std_error = res.standard_error
    zvalue = kappa / std_error
    pvalue = norm.sf(abs(zvalue)) * 2

    return upper_CI, lower_CI, pvalue


def calculate_fleiss_kappa(data):
    k = irr.fleiss_kappa(data, method="fleiss")
    return k


def bootstrap_kappa(*args):
    max_val = np.max(args[0])
    complete_data = [[n, max_val - n] for n in args[0]]
    return calculate_fleiss_kappa(complete_data)


def get_subject_description_per_rater(tma_series: pd.DataFrame, annotators: list, annotation: str, raw: bool):
    exists_list = np.zeros_like(annotators.copy(), dtype=np.int64)
    annotation_column = "explanation_lvl_0" if raw == None else "explanation_lvl_2" if raw else "explanation_lvl_1"

    seen_annotators = np.zeros_like(annotators.copy(), dtype=np.int64)

    for _, row in tma_series.iterrows():
        if str(row[annotation_column]) == annotation and row["annotator"] in annotators:
            exists_list[np.where(np.array(annotators) == row["annotator"])] = np.int64(1)
        if row["annotator"] in annotators:
            seen_annotators[np.where(np.array(annotators) == row["annotator"])] = np.int64(1)

    for i in range(len(exists_list)):
        if exists_list[i] != 1:
            exists_list[i] = np.int64(0)

    if seen_annotators.sum() < 3:
        print("NOT ALL ANNOTATORS SAW THE TMA!")
        print("exist:", exists_list)
        print("seenby:", seen_annotators)
        print("annotators:", annotators)
        print("TMA", tma_series["TMA_identifier"].unique())

    return exists_list


if __name__ == "__main__":
    random.seed(42)

    in_path = Path(os.environ["DATASET_LOCATION"]) / "GleasonXAI"
    out_path = Path("./figures/kappa")
    if not out_path.exists():
        os.makedirs(out_path, exist_ok=True)

    dataset = GleasonX(
        in_path,
        split="all",
        scaling="MicronsCalibrated",
        label_level=1,
        create_seg_masks=True,
        drawing_order="grade_frame_order",
        explanation_file="final_filtered_explanations_df.csv",
        data_split=[0.7, 0.15, 0.15],
        tissue_mask_kwargs={"open": False, "close": False, "flood": False},
    )

    # use_sub_explanations: None - Gleason_Pattern, True - Sub-explanations, False - Explanations
    calculate_kappa_per_group_and_label(dataset.df, out_path, use_sub_explanations=True)
